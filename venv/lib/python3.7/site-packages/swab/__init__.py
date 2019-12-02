# encoding=UTF-8

# Simple WSGI A/B testing.
#
# (c) 2010-2018 Oliver Cope.
#
# See ``README.rst`` for usage instructions etc.


import base64
import logging
import math
import os
import re
from collections import OrderedDict
from random import Random
from struct import unpack_from
from time import time
from hashlib import md5
from functools import partial
try:
    from urllib.parse import quote_plus
except ImportError:
    from urllib import quote_plus
from pkg_resources import resource_filename
from fresco import FrescoApp, Request, Response
from fresco.cookie import Cookie
from fresco_static import StaticFiles
from fresco_template import Piglet
from piglet import TemplateLoader


logger = logging.getLogger(__name__)
templateloader = TemplateLoader([resource_filename(__name__, 'templates')])
render = Piglet(templateloader)
ustr = type(u'')

__version__ = '0.2.2'

is_bot_ua = re.compile("|".join([
    r"findlinks",
    r"ia_archiver",
    r"ichiro",
    r".*bot\b",
    r".*crawler\b",
    r".*spider\b",
    r".*seeker\b",
    r".*fetch[oe]r\b",
    r".*wget",
    r".*plukkie",
    r".*Nutch",
    r".*InfoLink",
    r".*indy library",
    r".*yandex",
    r".*ezooms\b",
    r".*jeeves\b",
    r".*mediapartners-google",
    r".*jakarta Commons",
    r".*java/",
    r".*mj12",
    r".*speedy",
    r".*bot_",
    r".*pubsub",
    r".*facebookexternalhit",
    r".*feedfetcher-Google",
    r".*pflab",
    r".*metauri",
    r".*shopwiki",
    r".*libcurl",
    r".*resolver",
    r".*service",
    r".*postrank",
    r"^.{0,4}$",
]), re.I).match


class Swab(object):
    """
    Simple WSGI A/B testing
    """

    def __init__(self, datadir, wsgi_mountpoint='/swab'):

        """
        Create a new Swab test object

        :param datadir: Path to data storage directory
        :param wsgi_mountpoint: The path swab-specific views and resources will
                                be served from by the middleware.
        """
        self.datadir = datadir

        #: Mapping of {<experiment name>: <Experiment object>}
        self.experiments = {}

        #: Mapping of {<goal name>: [<Experiment object>, ...]}
        self.experiments_by_goal = {}

        self.wsgi_mountpoint = wsgi_mountpoint
        makedir(self.datadir)

    def include(self, environ, experiment):
        e = self.experiments[experiment]
        return e.include(environ) and not e.exclude(environ)

    def middleware(self, app, cookie_domain=None, cookie_path=None,
                   cache_control=True):
        """
        Middleware that sets a random identity cookie for tracking users.

        The identity can be overwritten by setting ``environ['swab.id']``
        before start_response is called. On egress this middleware will then
        reset the cookie if required.

        :param app: The WSGI application to wrap

        :param cookie_domain: The domain to use when setting cookies. If
                              ``None`` this will not be set and the browser
                              will default to the domain used for the request.

        :param cache_control: If ``True``, replace the upstream application's
                              cache control headers for any request where
                              show_variant is invoked.
        """

        swabapp = FrescoApp()

        static_files = StaticFiles()
        static_files.init_app(swabapp)
        static_files.add_package(__name__, 'static')

        swabapp.route_wsgi('/results', self.results_app)
        swabapp.route_wsgi('/r.js', self.record_trial_app)

        def middleware(environ, start_response):

            environ['swab.swab'] = self
            environ['swab.id'] = initswabid = _getswabid(environ)
            environ['swab.invoked'] = False
            environ['swab.experiments_invoked'] = set()
            if initswabid is None:
                environ['swab.id'] = generate_id()

            if environ['PATH_INFO'][:len(self.wsgi_mountpoint)] == \
                    self.wsgi_mountpoint:
                environ['SCRIPT_NAME'] += self.wsgi_mountpoint
                environ['PATH_INFO'] = \
                        environ['PATH_INFO'][len(self.wsgi_mountpoint):]
                return swabapp(environ, start_response)

            def my_start_response(status, headers, exc_info=None,
                                  _cache_headers=set(['cache-control',
                                                      'expires', 'etag',
                                                      'last-modified'])):
                if not environ['swab.invoked']:
                    return start_response(status, headers, exc_info)

                swabid = _getswabid(environ)
                if swabid == initswabid and swabid is not None:
                    return start_response(status, headers, exc_info)

                if swabid is None:
                    swabid = generate_id()
                    environ['swab.id'] = swabid

                _cookie_path = (cookie_path or
                                environ.get('SCRIPT_NAME') or
                                '/')
                cookie = Cookie('swab',
                                swabid.decode('ascii'),
                                path=_cookie_path,
                                domain=cookie_domain,
                                httponly=True,
                                max_age=365)
                if cache_control and environ.get('swab.experiments_invoked'):
                    headers = [(k, v)
                               for k, v in headers
                               if k.lower() not in _cache_headers]
                    headers.append(('Cache-Control', 'no-cache'))
                headers.append(("Set-Cookie", str(cookie)))
                return start_response(status, headers, exc_info)

            return app(environ, my_start_response)

        return middleware

    def add_goals(self, names):
        for n in names:
            self.experiments_by_goal.setdefault(n, set())

    def add_experiment(self, name, variants=None, goal=None):
        exp = self.experiments[name] = Experiment(name)
        if variants:
            exp.add(*variants)

        goal = goal if goal is not None else name
        self.add_goals({goal})
        self.experiments_by_goal[goal].add(exp)
        makedir(os.path.join(self.datadir, name))
        return exp

    def results_app(self, environ, start_response):
        request = Request(environ)
        dedupe = not bool(request.get('nodedupe'))
        data = self.collect_experiment_data(dedupe=dedupe)
        for exp in data:

            vdata = data[exp]['variants']
            control = request.get('control.' + exp,
                                  self.experiments[exp].control)
            control_trials = vdata[control]['trials']
            control_goals = vdata[control]['goals']
            data[exp]['control'] = control

            for variant in vdata:
                variant_trials = vdata[variant]['trials']
                for goal, goaldata in vdata[variant]['goals'].items():
                    control_rate = control_goals[goal]['rate']
                    control_conversions = control_goals[goal]['conversions']
                    z = zscore(goaldata['rate'],
                               variant_trials,
                               control_rate,
                               control_trials)
                    goaldata['z'] = z
                    goaldata['confidence'] = cumulative_normal_distribution(z)
                    goaldata['p_beats_control'] = probability_b_beats_a(
                        control_trials,
                        control_conversions,
                        variant_trials,
                        goaldata['conversions']
                    )

        return render.as_response(
            resource_filename('swab', 'templates/results.html'), {
                'request': request,
                'experiments': self.experiments.values(),
                'data': data,
                'dedupe': dedupe,
            }
        )(environ, start_response)

    def record_trial_app(self, environ, start_response):
        request = Request(environ)
        experiment = request.query.get('e')
        try:
            record_trial(environ, experiment)
        except KeyError:
            pass
        return Response([], cache_control="no-cache")(environ, start_response)

    def collect_experiment_data(self, dedupe=False):
        """
        Collect experiment data from the log files

        Return a dictionary of::

            {<experiment>: {
                'goals': [goal1, goal2, ...],
                'variants': {
                    'v1': {
                        'trials': 1062,
                        'goals': {
                            'goal1': {'conversions': 43, 'rate': 0.0405},
                            'goal2': {'conversions': 29, 'rate': 0.0273},
                        }
                    },
                    ...
                }
            }

        """
        data = {}

        for exp in self.experiments.values():
            expdir = os.path.join(self.datadir, exp.name)
            goals = sorted(
                [goal
                 for goal, experiments in self.experiments_by_goal.items()
                 if exp in experiments])
            data[exp.name] = {'goals': goals, 'variants': {}}

            for variant in exp.variants:
                path = partial(os.path.join, expdir, variant)
                if dedupe:
                    trial_identities = get_identities(path('__all__'))
                    trialc = len(trial_identities)
                else:
                    trialc = count_entries(path('__all__'), dedupe=False)
                data[exp.name]['variants'][variant] = vdata = {'trials': trialc, 'goals': {}}

                for goal in goals:
                    vdata['goals'][goal] = goaldata = {}
                    if dedupe:
                        conv_identities = trial_identities.intersection(
                                            get_identities(path(goal)))
                        convc = len(conv_identities)
                    else:
                        convc = count_entries(path(goal), dedupe=False)
                    goaldata['conversions'] = convc
                    goaldata['rate'] = (float(convc) / trialc
                                        if trialc
                                        else float('nan'))
        return data


class Experiment(object):
    invalid_chars = r'\\/:,'

    def __init__(self, name, seed_strategy=None):
        if not self._is_valid_name(name):
            raise ValueError("Experiment name may not contain any of {!r}"
                             .format(set(self.invalid_chars)))
        self.name = name

        #: The control variant name
        self.control = None

        #: Mapping of variant name → weight
        self.variants = OrderedDict()

        #: The default variant (typically shown to bots)
        self.default_variant = None

        #: Function returning a the RNG seed to use
        self.seed_strategy = (seed_strategy
                              if seed_strategy is not None
                              else default_seed_strategy)

    def add(self, *variants):
        """
        Add variants. Each argument may be either a variant name, or a tuple
        of (variant-name, weight). For example:

            colors = Experiment('colors')
            colors.add('blue', 'red')           # equal split between blue/red
            colors.add(('blue', 1), ('red', 3)) # red 3x more likely than blue
        """
        for v in variants:
            if isinstance(v, (str, ustr)):
                name, weight = v, 1
            else:
                name, weight = v

            if not self._is_valid_name(name):
                raise ValueError("Variant name may not contain any of {!r}"
                                 .format(set(self.invalid_chars)))
            if self.control is None:
                self.control = name
            self.variants[name] = self.variants.get(name, 0) + 1

        self._recalculate()

    def _recalculate(self):
        """
        Recalculate weighted_variants and default_variant
        """
        self.weighted_variants = [n for n in self.variants
                                  for i in range(self.variants[n])]
        default = self.control
        default_weight = self.variants[self.control]
        for name, weight in self.variants.items():
            if weight > default_weight:
                default, default_weight = name, weight
        self.default_variant = default

    def _is_valid_name(self, name):
        return not any(c in name for c in self.invalid_chars)

    def include(self, environ):
        return True

    def exclude(self, environ):
        return False


def _getswabid(environ):
    """
    Return the unique identifier from the WSGI environment if present,
    otherwise return ``None``.
    """
    try:
        return environ['swab.id']
    except KeyError:
        pass
    cookie = Request(environ).cookies.get('swab')
    if cookie:
        swabid = cookie.value.encode('ascii')
        environ['swab.id'] = swabid
        return swabid
    return None


def generate_id(urandom=os.urandom, encode=base64.b64encode):
    """
    Return a unique id
    """
    return encode(urandom(12)).strip()


def get_rng(environ, experiment, swabid):
    """
    Return a random number generator with a fixed seed based on the
    current session's swabid
    """
    r = Random()
    r.seed(experiment.seed_strategy(environ, experiment, swabid))
    return r


def get_seed_from_bytes(s):
    """
    Given a byte string, return a RNG seed value
    """
    return unpack_from('l', md5(s).digest())[0]


def default_seed_strategy(environ, experiment, swabid):
    return get_seed_from_bytes(swabid + experiment.name.encode('UTF-8'))


def show_variant(environ, experiment, record=False, variant=None):
    """
    Return the variant name that ``environ`` is assigned to within
    ``experiment``

    If ``record`` is true, write a line to the log file indicating that the
    variant was shown. (No deduping is done - the log line is always written. A
    page with ``show_variant`` might record multiple hits on reloads etc)

    :param experiment: Name of the experiment
    :param environ: WSGI environ
    :param record: If ``True``, record a trial for the experiment in the log
                   file
    :param variant: force the named variant. Use this if your application wants
                    to choose the variant based on some other criteria
                    (eg SEO a/b testing where you assign the variant based
                    on the URL)
    """
    swab = environ['swab.swab']
    exp = swab.experiments[experiment]
    variants = exp.weighted_variants
    swabid = _getswabid(environ)

    if not swab.include(environ, experiment):
        variant = variant if variant is not None else exp.control

    # Make sure bots don't receive randomized variants. Google's advice
    # (as of August 2017): show the version the majority of users see.
    # Failing that, show a consistent version (don't randomize)
    # https://www.youtube.com/watch?v=EaSyuH2D7Mw&start=1920
    if is_bot(environ):
        variant = variant if variant is not None else exp.control

    if variant is None:
        request = Request(environ)
        variant = request.query.get('swab.' + experiment)
        if variant is not None and variant in variants:
            return variant

        # Get a random int in the range 0 ≤ x < len(variants)
        #
        # We do this in preference to calling random.choice because
        # it guarantees a particular property of the sampling if the list of
        # variants changes.
        #
        # For example, given a list of variants ``['a', 'b']``
        # we will choose a variant according to the output of
        # `r = rng.random()` such that
        #
        #   0.0 ≤ r < 0.5  →  'a'
        #   0.5 ≤ r < 1.0  →  'b'
        #
        # if later we find that 'a' is winning and want to exploit that
        # variant, we could change the list of variants to
        # ``['a', 'a', 'a', 'b']``, and the mapping would become:
        #
        #   0.0  ≤ r < 0.75  →  'a'
        #   0.75 ≤ r < 1.0   →  'b'
        #
        # Notice that the range corresponding to variant 'a' completely
        # contains the old values - ie users who previously saw the winning
        # variant 'a' will continue to see that variant.
        r = int(get_rng(environ, exp, swabid).random() * len(variants))
        variant = variants[r]

    if variant not in variants:
        raise ValueError("Invalid variant {!r}. Choices are: {!r}"
                         .format(variant, variants))

    environ['swab.invoked'] = True

    invoked = environ.setdefault('swab.experiments_invoked', set())
    if experiment in invoked:
        return variant
    invoked.add(experiment)

    if not record or is_bot(environ):
        return variant

    path = os.path.join(swab.datadir, experiment, variant, '__all__')
    try:
        f = open(path, 'a')
    except IOError:
        makedir(os.path.dirname(path))
        f = open(path, 'a')

    try:
        f.write(_logline(swabid))
    finally:
        f.close()
    return variant


record_trial = partial(show_variant, record=True)


def record_trial_tag(environ, experiment=None):
    experiments = environ.get('swab.experiments_invoked', frozenset())
    if experiment is None:
        if len(experiments) != 1:
            raise ValueError("record_trial_tag can't guess the experiment "
                             "name, as show_variant has been called multiple "
                             "times (or not at all). "
                             "Fix this by passing an experiment name. ")
        experiment = next(iter(experiments))
    if experiment not in experiments:
        logger.warn(
            "record_trial_tag called without a corresponding show_variant "
            "(experiment=%r)", experiment)
    request = Request(environ)
    if 'swab.' + experiment in request.query:
        return ''
    swab = environ['swab.swab']
    return ('<script>'
            '(function(f,o,x){{'
            'x=f.getElementsByTagName(o)[0],'
            'o=f.createElement(o),'
            'o.async=1,'
            'o.src="{0}?e={1};s={2}";'
            'x.parentNode.insertBefore(o,x)}})(document,"script")'
            '</script>').format(
                request.make_url(path=swab.wsgi_mountpoint + '/' + 'r.js',
                                 query=''),
                quote_plus(experiment),
                _getswabid(environ))


def is_bot(environ, is_bot_ua=is_bot_ua):
    """
    Return True if the request is from a bot.
    Uses rather simplistic tests based on user agent and header signatures, but
    should still catch most well behaved bots.
    """
    if is_bot_ua(environ.get('HTTP_USER_AGENT', '')):
        return True
    if 'HTTP_ACCEPT_LANGUAGE' not in environ:
        return True
    return False


def _logline(swabid):
    return '%-14.2f:%s\n' % (time(), swabid)


def _logentries(path):
    with open(path, 'r') as f:
        for line in f:
            try:
                t, id = line.strip().split(':')
            except ValueError:
                continue
            yield float(t.strip()), id


def record_goal(environ, goal, experiment=None):
    """
    Record a goal conversion by adding a record to the file at
    ``swab-path/<experiment>/<variant>/<goal>``.

    If experiment is not specified, all experiments linked to the named goal
    are looked up.

    This doesn't use any file locking, but we should be safe on any posix
    system as we are appending each time to the file.
    See http://www.perlmonks.org/?node_id=486488 for a discussion of the issue.
    """

    if is_bot(environ):
        return

    swab = environ['swab.swab']
    if experiment is None:
        try:
            experiments = swab.experiments_by_goal[goal]
        except KeyError:
            raise ValueError("Invalid goal: {!r}".format(goal))
    else:
        experiments = [swab.experiments[experiment]]
    for experiment in experiments:
        if not swab.include(environ, experiment.name):
            continue

        variant = show_variant(environ, experiment.name, record=False)
        path = os.path.join(swab.datadir, experiment.name, variant, goal)
        try:
            f = open(path, 'a')
        except IOError:
            makedir(os.path.dirname(path))
            f = open(path, 'a')

        try:
            f.write(_logline(_getswabid(environ)))
        finally:
            f.close()


def makedir(path):
    """
    Create a directory at ``path``. Unlike ``os.makedirs`` don't raise an error
    if ``path`` already exists.
    """
    try:
        os.makedirs(path)
    except OSError:
        # Path already exists or cannot be created
        if not os.path.isdir(path):
            raise


def count_entries(path, dedupe=True):
    """
    Count the number of entries in ``path``.

    :param dedupe: if True, dedupe Entries so only one conversion is counted per
                  identity.
    """
    if dedupe:
        return len(get_identities(path))
    else:
        if not os.path.exists(path):
            return 0
        with open(path, 'rb') as f:
            return sum(1 for line in f)


def get_identities(path):
    """
    Return a Counter for identity entries in ``path``
    """
    if not os.path.isfile(path):
        return set()

    return set(identity for t, identity in _logentries(path))


def zscore(p, n, pc, nc):
    """
    Calculate the zscore of probability ``p`` over ``n`` tests, compared to
    control probability ``pc`` over ``nc`` tests

    See http://20bits.com/articles/statistical-analysis-and-ab-testing/.
    """
    from math import sqrt
    try:
        return (p - pc) / sqrt(
            (p * (1 - p) / n) +
            (pc * (1 - pc) / nc))
    except (ZeroDivisionError, ValueError):
        return float('nan')


def cumulative_normal_distribution(z):
    """
    Return the confidence level from calculating of the cumulative normal
    distribution for the given zscore.

    See http://abtester.com/calculator/ and
    http://www.sitmo.com/doc/Calculating_the_Cumulative_Normal_Distribution
    """
    from math import exp

    b1 = +0.319381530
    b2 = -0.356563782
    b3 = +1.781477937
    b4 = -1.821255978
    b5 = +1.330274429
    p = +0.2316419
    c = +0.39894228

    if z >= 0.0:
        t = 1.0 / (1.0 + p * z)
        return 1.0 - c * exp(-z * z / 2.0) * t * \
                (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1)
    else:
        t = 1.0 / (1.0 - p * z)
        return c * exp(-z * z / 2.0) * t * \
                (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1)


def probability_b_beats_a(trials_a, goals_a, trials_b, goals_b):
    a_A = 1 + goals_a
    b_A = 1 + trials_a - goals_a
    a_B = 1 + goals_b
    b_B = 1 + trials_b - goals_b

    # Happens when goals > trials.
    # In this case the gamma function isn't defined and the remaining code
    # errors.
    if b_A <= 0 or b_B <= 0:
        return float('nan')

    total = 0.0
    log = math.log
    exp = math.exp

    # Define a log beta function in terms of gamma
    lbeta = lambda a, b, lgamma=math.lgamma: (lgamma(a) +
                                              lgamma(b) -
                                              lgamma(a + b))

    for i in range(a_B):
        total += exp(
            lbeta(a_A + i, b_B + b_A) -
            log(b_B + i) -
            lbeta(1 + i, b_B) -
            lbeta(a_A, b_A)
        )
    return total
