# Copyright 2015 Oliver Cope
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
import sys
from functools import wraps

from fresco import Response

_marker = object()


def onerror(exceptions, handler):
    """\
    Return a decorator that can replace or update the return value of the
    function if an exception is raised
    """

    try:
        if isinstance(exceptions, Exception):
            exceptions = (exceptions,)
    except TypeError:
        pass

    def decorator(func):
        @wraps(func)
        def decorated(*fargs, **fkwargs):
            try:
                return func(*fargs, **fkwargs)
            except exceptions:
                exc_info = sys.exc_info()
                return handler(exc_info, *fargs, **fkwargs)
        return decorated
    return decorator


def json_response(data=_marker,
                  indent=None,
                  separators=(',', ':'),
                  content_type='application/json',
                  **kwargs):
    """
    JSON encode the function's result and return a response object with the
    content-type ``application/json``.

    May also be used as a regular function to directly create a JSON encoded
    response, however it's preferrable to use
    :meth:`~fresco.response.Response.json` for this.

    :param data: The data to json encode.
                 If left unspecified, ``json_response``
                 will act as a function decorator.
    :param indent: The indent level. Defaults to ``None`` (no pretty printing)
    :param separators: Defaults to ``(',', ':')`` for the most compact JSON
                       representation
    :param kwargs: Other keyword arguments are passed to ``json.dumps``. These
                   may be used to change encoding paramters, for example
                   overriding the default ``JSONEncoder`` class.


    """
    # Called as a decorator or decorator factory
    if data is _marker or callable(data):
        def json_response_decorator(func):
            @wraps(func)
            def json_response_decorated(*fa, **fkw):
                return Response.json(
                    func(*fa, **fkw), indent, separators, **kwargs)
            return json_response_decorated

        # Called as a decorator factory (``@json_response()``)
        if data is _marker:
            return json_response_decorator

        # Called as a decorator (ie no parens: ``@json_response``)
        return json_response_decorator(data)

    # Called as a regular function (or via ``Route(...).filter``)
    else:
        if isinstance(data, Response):
            return data

        return Response.json(data,
                             indent=indent,
                             separators=separators,
                             content_type=content_type,
                             **kwargs)
