from flea import Agent
import pytest

import string
from shutil import rmtree
from tempfile import mkdtemp

from swab import (Swab,
                  generate_id,
                  record_goal,
                  record_trial,
                  record_trial_tag,
                  show_variant,
                  )

Agent.environ_defaults.update(
    HTTP_USER_AGENT="A real web browser, honest guv",
    HTTP_ACCEPT_LANGUAGE='en,q=0.9'
)


def make_env(swab, *args, **kwargs):
    env = {'swab.id': generate_id(),
           'swab.swab': swab,
           'QUERY_STRING': '',
           'HTTP_USER_AGENT': 'A real web browser, honest guv',
           'HTTP_ACCEPT_LANGUAGE': 'en,q=0.9',
           'SERVER_NAME': 'localhost',
           'SERVER_PORT': '80',
           'SCRIPT_NAME': '',
           'PATH_INFO': '/'}
    for d in list(args) + [kwargs]:
        env.update(d)
    return env


class SwabTestBase(object):

    def setup(self):
        self.datadir = mkdtemp()

    def teardown(self):
        rmtree(self.datadir)


class TestSwab(SwabTestBase):

    def test_identity_set_and_preserved(self):

        def app(environ, start_response):
            show_variant(environ, 'exp', record=True)
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return []

        s = Swab(self.datadir)
        s.add_experiment('exp', 'yn')

        agent = Agent(s.middleware(app))
        r = agent.get('/')

        assert 'swab=' in r.response.get_header('Set-Cookie'), \
                "Swab cookie not set on first request"

        r = r.get('/')
        assert 'swab=' not in r.response.get_header('Set-Cookie'), \
                "Swab cookie reset on subsequent request"

    def test_override_identity(self):

        def app(environ, start_response):
            show_variant(environ, 'exp')
            environ['swab.id'] = b'1234567890'
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return []

        s = Swab(self.datadir)
        s.add_experiment('exp', 'yn')
        agent = Agent(s.middleware(app))
        assert 'swab=1234567890;' in \
                agent.get('/').response.get_header('Set-Cookie')

    def test_show_variants_produces_all_variants(self):

        def app(environ, start_response):
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return [show_variant(environ, 'exp').encode('ascii')]

        s = Swab(self.datadir)
        s.add_experiment('exp', string.digits, 'goal')

        variants = set()
        for i in range(100):
            agent = Agent(s.middleware(app))
            variants.add(agent.get('/').body)
        assert len(variants) == 10

    def test_show_variant_returns_requested_variant(self):

        def app(environ, start_response):
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return [show_variant(environ, 'exp').encode('ascii')]

        s = Swab(self.datadir)
        s.add_experiment('exp', ['a', 'b'], 'goal')

        variants = set()
        for i in range(100):
            agent = Agent(s.middleware(app))
            variants.add(agent.get('/?swab.exp=a').body)
        assert variants == set('a')

    def test_show_variant_does_not_error_if_called_before_start_response(self):

        def app(environ, start_response):
            response = [show_variant(environ, 'exp').encode('ascii')]
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return response

        s = Swab(self.datadir)
        s.add_experiment('exp', ['a', 'b'], 'goal')

        Agent(s.middleware(app)).get('/').body

    def test_show_variant_returns_default_to_bots(self):

        def app(environ, start_response):
            response = [show_variant(environ, 'exp').encode('ascii')]
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return response

        s = Swab(self.datadir)
        s.add_experiment('exp', ['a', 'b'], 'goal')

        for i in range(10):
            r = Agent(s.middleware(app)).get('/', HTTP_USER_AGENT='Googlebot')
            assert r.body == 'a'

    def test_record_trial_tag_returns_script_for_bots(self):
        """
        Calling record_trial_tag should return the same javascript code for
        bots as for a regular user-agent.
        """

        def app(environ, start_response):
            start_response('200 OK', [('Content-Type', 'text/plain')])
            show_variant(environ, 'exp').encode('ascii')
            return [
                record_trial_tag(environ).encode('ascii')
            ]

        s = Swab(self.datadir)
        s.add_experiment('exp', ['a', 'b'], 'goal')
        r = Agent(s.middleware(app)).get('/', HTTP_USER_AGENT='Googlebot')
        assert 'r.js?e=exp' in r.body

    def test_default_variant_is_most_common_or_control(self):
        """
        Test that the default variation is either that which most users see,
        or failing this, the control variant.

        See google webmaster hangout for discussion:
        https://www.youtube.com/watch?v=EaSyuH2D7Mw&start=1920
        """
        s = Swab(self.datadir)
        s.add_experiment('foo', ['v1', 'v2'], 'goal')
        assert s.experiments['foo'].default_variant == 'v1'

        s.add_experiment('bar', ['v1', 'v2', 'v2'], 'goal')
        assert s.experiments['bar'].default_variant == 'v2'

    def test_variants_are_chosen_independently(self):
        s = Swab(self.datadir)
        s.add_experiment('foo', ['v1', 'v2'], 'goal')
        s.add_experiment('bar', ['v1', 'v2'], 'goal')
        for i in range(100):
            env = make_env(s)
            a = show_variant(env, 'foo', record=False)
            b = show_variant(env, 'bar', record=False)
            if a != b:
                break
        else:
            raise AssertionError("Expected different variants to be chosen")

    def test_record_goal_fails_with_invalid_name(self):
        s = Swab(self.datadir)
        s.add_experiment('foo', ['v1', 'v2'], 'foo_goal')

        def good_app(environ, start_response):
            record_goal(environ, 'foo_goal')
            start_response('204 No Content', [])
            return []

        def bad_app(environ, start_response):
            record_goal(environ, 'oof')
            start_response('204 No Content', [])
            return []

        Agent(s.middleware(good_app)).get('/')
        with pytest.raises(ValueError):
            Agent(s.middleware(bad_app)).get('/')

    def test_caching_headers_added(self):

        def app(environ, start_response):
            record_trial(environ, 'exp')
            start_response('200 OK', [
                ('Content-Type', 'text/plain'),
                ('Last-Modified', 'x'),
                ('ETag', 'x'),
                ('Expires', 'x'),
                ('Cache-Control', 'cache me!'),
            ])
            return []

        s = Swab(self.datadir)
        s.add_experiment('exp', ['a', 'b'], 'goal')
        r = Agent(s.middleware(app)).get('/')
        headers = dict(r.response.headers)

        assert 'Content-Type' in headers
        assert 'Last-Modified' not in headers
        assert 'Etag' not in headers
        assert 'Expires' not in headers
        assert headers['Cache-Control'] == 'no-cache'


class TestResults(SwabTestBase):

    def test_results_page_renders(self):

        def app(environ, start_response):
            response = [show_variant(environ, 'exp').encode('ascii')]
            record_goal(environ, 'goal')
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return response

        s = Swab(self.datadir)
        s.add_experiment('exp', ['variant1', 'variant2'], 'goal')
        r = Agent(s.middleware(app))
        r.get('/')

        assert 'A/B test results summary' in r.get('/swab/results').body
        assert 'variant1' in r.get('/swab/results').body

    def test_goals_without_trials_ignored(self):
        s = Swab(self.datadir)
        s.add_experiment('exp', ['variant1', 'variant2'], 'goal')
        env = make_env(s)
        record_goal(env, 'goal')
        r = Agent(s.middleware(None))
        r = r.get('/swab/results')
        for row in r('table tr'):
            try:
                assert row('td')[0].text == '0'
            except IndexError:
                # Header row
                continue


class TestSwabJS(SwabTestBase):

    def test_record_trial_tag(self):
        s = Swab(self.datadir)
        env = make_env(s)
        s.add_experiment('exp', ['a', 'b'], 'goal')
        assert record_trial_tag(env, 'exp') == (
            '<script>(function(f,o,x){{x=f.getElementsByTagName(o)[0],'
            'o=f.createElement(o),o.async=1,'
            'o.src="http://localhost/swab/r.js?e=exp;s={0}";'
            'x.parentNode.insertBefore(o,x)}})(document,"script")'
            '</script>').format(env['swab.id'])

    def test_tag_not_generated_if_variant_forced(self):
        s = Swab(self.datadir)
        env = make_env(s, QUERY_STRING='swab.exp=a')
        s.add_experiment('exp', ['a', 'b'], 'goal')
        assert record_trial_tag(env, 'exp') == ''

    def test_tag_infers_experiment_name(self):
        s = Swab(self.datadir)
        env = make_env(s, QUERY_STRING='swab.exp=a')
        s.add_experiment('my-experiment', ['a', 'b'], 'goal')
        show_variant(env, 'my-experiment')
        assert 'my-experiment' in record_trial_tag(env)

    def test_tag_requires_experiment_name_if_ambiguous(self):
        s = Swab(self.datadir)
        env = make_env(s, QUERY_STRING='swab.exp=a')
        s.add_experiment('my-experiment', ['a', 'b'], 'goal')
        s.add_experiment('your-experiment', ['a', 'b'], 'goal')

        # No experiments previously invoked via show_variant()
        with pytest.raises(ValueError):
            record_trial_tag(env)

        # Multiple experiments previously invoked via show_variant()
        show_variant(env, 'my-experiment')
        show_variant(env, 'your-experiment')
        with pytest.raises(ValueError):
            record_trial_tag(env)

    def test_javascript_response_not_cachable(self):
        def app(environ, start_response):
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return []

        s = Swab(self.datadir)
        s.add_experiment('exp', ['a', 'b'], 'goal')
        agent = Agent(s.middleware(app))
        r = agent.get('/swab/r.js?e=exp')
        s.add_experiment('exp', ['a', 'b'], 'goal')
        assert r.response.get_header('Cache-Control') == 'no-cache'
