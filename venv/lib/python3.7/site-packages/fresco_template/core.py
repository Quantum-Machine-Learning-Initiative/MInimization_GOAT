# Copyright 2015 Oliver Cope
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import unicode_literals

from collections import Mapping
from functools import wraps

from fresco import context, Response


def _make_decorator_or_filter(render_method, template, *args, **kwargs):
    """
    Return a function that can act either as a decorator around a view function
    or a filter function called on the function's output.

    The decorated/filtered function must return a data dict that will be
    rendered with `` template``
    """
    def _render(data):
        if template is None:
            try:
                local_template, data = data
            except (TypeError, ValueError):
                return data
        else:
            local_template = template

        if data is None:
            return {}

        # Pass through any response that is not a mapping
        if not isinstance(data, Mapping):
            return data

        return render_method(
            local_template, data, _in_decorator=True, *args, **kwargs)

    def decorate_or_filter(func_or_data):

        if callable(func_or_data) and not isinstance(func_or_data, Response):
            func = func_or_data

            @wraps(func)
            def decorated(*fargs, **fkwargs):
                data = func(*fargs, **fkwargs)
                return _render(data)

            return decorated

        else:
            data = func_or_data
            return _render(data)

    return decorate_or_filter


class TemplateContent(object):
    """\
    Wrap template rendering information (template, template vars, filters etc)
    so that it can be used as a WSGI response iterator.

    The template is not rendered until the WSGI response is iterated, giving
    downstream functions a chance to modify the templated response, eg by
    injecting new template variables.
    """

    def __init__(self, environment, template, template_context, filters,
                 stream=False):

        #: Template plugin
        self.environment = environment
        self.render = environment.plugin.render

        #: Template filename
        self.template = template

        #: Vars passed into template rendering context
        self.template_context = template_context

        #: If true, the template backend should stream the result instead of
        #: rendering it as a single string
        self.stream = stream

        # Call get_context upfront to ensure any context processors are called
        # before rendering starts and headers are sent.
        # This is necessary for context processors that need to coordinate with
        # middleware (eg CSRF middleware, which typically injects a csrf token
        # into the template context *and* sets a cookie).
        self.full_context = dict(environment.get_context(), **template_context)

    def as_string(self):
        """
        Return the rendered template as a string
        """
        return self.render(self.template, self.full_context, stream=False)

    def as_stream(self):
        """
        Return the rendered template as a stream
        """
        return self.render(self.template, self.full_context, stream=True)

    def __iter__(self):
        """
        Return the rendered template as a string
        """
        if self.stream:
            return self.as_stream()
        return iter([self.as_string()])


class TemplateEnvironment(object):
    """
    An rendering environment not tied to any specific templating system
    """

    #: Registry of TemplateEnvironment instances
    __registry__ = {}

    #: Default plugin class to use.
    #: Subclasses should override this to provide plugin-specific environments
    plugin_class = None

    def __init__(self, *args, **kwargs):
        app = kwargs.pop('app', None)
        self.name = kwargs.pop('name', 'default')
        self.plugin = self.plugin_class(*args, **kwargs)
        if app:
            self.init_app(app)

        #: Variables accessible from all templates
        self.context_sources = [self.default_context]

    @classmethod
    def of(cls, app, name='default'):
        try:
            return cls.__registry__[(cls, app, name)]
        except KeyError:
            names = [n for n in cls.__registry__.keys() if n[:2] == (cls, app)]
            if not names:
                raise ValueError(
                    "No {} registered for {}".format(cls.__name__, app))

            raise ValueError(
                "No {envclass} named {name} registered for {app} (try {names})"
                .format(envclass=cls.__name__,
                        app=app,
                        name=name,
                        names=', '.join(names)))

    def default_context(self, request):
        return {'context': context,
                'request': request,
                'urlfor': context.app.urlfor}

    def init_app(self, app):
        self.__registry__[(self.__class__, app, self.name)] = self

    def get_context(self):
        """
        Merge all template context sources into a single dict
        """
        result = {}
        merge = result.update
        request = context.request
        for source in self.context_sources:
            if callable(source):
                merge(source(request))
            else:
                merge(source)
        return result

    def contextprocessor(self, func):
        """
        Register a function as a context source
        """
        self.context_sources.append(func)
        return func

    def as_string(self, template=None, data=None, _in_decorator=False):
        """
        Return the string output of the rendered template.
        Can also work as a function decorator
        """
        if data is None and not _in_decorator:
            return _make_decorator_or_filter(self.to_string, template)

        ns = self.get_context()
        ns.update(data)
        return self.plugin.render(template, ns, stream=False)

    def as_stream(self, template=None, data=None, _in_decorator=False):
        """
        Return the string output of the rendered template.
        Can also work as a function decorator
        """
        if data is None and not _in_decorator:
            return _make_decorator_or_filter(self.as_string, template)

        ns = self.get_context()
        ns.update(data)
        return self.plugin.render(template, ns, stream=True)

    def as_response(self, template=None, data=None, _in_decorator=False,
                    _stream=False, **response_kwargs):
        """
        Return a response object for the rendered template::

            >>> from jinja2 import PackageLoader
            >>> from fresco import FrescoApp
            >>> app = FrescoApp()
            >>> render = Jinja2(app, loader=PackageLoader('myapp', 'tmpls'))
            >>> response = render.as_response('my_template.html',
            ...                               {'foo': 'bar'}) #doctest: +SKIP

        Can also be used as a decorator. The decorated function will merge the
        original function's return value (a dict) with the specified template::

            >>> render = Jinja2(app, loader=PackageLoader('myapp', 'tmpls'))
            >>>
            >>> @render.as_response('my_template.html') #doctest: +SKIP
            ... def handler(request):
            ...     return {'foo': 'bar'}
            ...
        """
        if data is None and not _in_decorator:
            return _make_decorator_or_filter(self.as_response, template,
                                   **response_kwargs)

        return Response(
            TemplateContent(self, template, data, None, stream=_stream),
            **response_kwargs)

    render = as_response
    __call__ = as_response


class BasePlugin(object):
    """
    Base template plugin. Subclass this for each supported templating system.
    """

    def render(self, template_name, namespace, stream=False):
        raise NotImplementedError()


def get_template_environment(app=None,
                             name='default',
                             cls=TemplateEnvironment):
    app = app or context.app
    return cls.of(app)
