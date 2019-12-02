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

from collections import OrderedDict
from posixpath import isfile
from posixpath import isdir
from posixpath import join
import logging

from pkg_resources import ResourceManager
from pkg_resources import get_provider

from fresco import GET
from fresco import HEAD
from fresco import context
from fresco.static import serve_static_file
from fresco.util.urls import normpath

__version__ = '0.2'

logger = logging.getLogger(__name__)


class StaticFiles(object):

    #: Registry of instances
    __registry__ = {}

    def __init__(self, app=None, prefix='/static',
                 cache_max_age=60, route_name='static'):
        self.__class__.__registry__[app] = self
        self.app = app
        self.prefix = prefix
        self.route_name = route_name
        self.sources = OrderedDict()
        self.resource_manager = ResourceManager()

        #: Default ``Cache-Control: max-age`` value
        self.cache_max_age = cache_max_age

        #: Add "Access-Control-Allow-Origin: *" header?
        self.access_control_allow_origin = '*'

        if app is not None:
            self.init_app(app)

    @classmethod
    def of(cls, app):
        return cls.__registry__[app]

    @classmethod
    def active(cls, context=context):
        return cls.of(context.app)

    def init_app(self, app):
        app.route(self.prefix + '/<path:path>', [GET, HEAD], self.serve,
                  name=self.route_name)

    def add_package(self, package_name, directory, cache_max_age=None):
        """
        Add static files served from within a python package.

        Only one directory per python package may be configured using this
        method. For more flexibility use
        :meth:`fresco_static.StaticFiles.add_source`.

        :param package_name: The python package name
        :param directory: The directory within the package containing the
                          static files
        :param cache_max_age: Optional duration in seconds for the
                              Cache-Control max-age header. If omitted the
                              default value is used
        """
        self.add_source(package_name, package_name, directory, cache_max_age)

    def add_directory(self, name, directory, cache_max_age=None):
        """
        Add a directory for static files not associated with any python
        package.

        :param name: The (unique) name used to identify this source
        :param directory: Absolute path to the directory containing the
                          static files
        :param cache_max_age: Optional duration in seconds for the
                              Cache-Control max-age header. If omitted the
                              default value is used
        """
        self.add_source(name, None, directory, cache_max_age)

    def add_source(self, name, package_name, directory, cache_max_age=None):
        """
        Add a static files source directory, optionally associated with a
        python package.

        :param name: The (unique) name used to identify this source.
        :param package_name: The name of the python package containing the
                             files
        :param directory: Path to the directory containing the
                          static files. Should be relative if package_name is
                          specified, otherwise absolute.
        :param cache_max_age: Optional duration in seconds for the
                              Cache-Control max-age header. If omitted the
                              default value is used
        """
        if name in self.sources:
            raise ValueError("StaticFiles source %r is already used" % (name,))

        if package_name:
            map_path = get_provider(package_name).get_resource_filename
        else:
            def map_path(resource_manager, path):
                return path

        static_root = map_path(self.resource_manager, directory)
        if not isdir(static_root):
            raise ValueError("%r is not a directory" % (static_root,))

        cache_max_age = (self.cache_max_age
                         if cache_max_age is None
                         else cache_max_age)

        self.sources[name] = (map_path, directory, cache_max_age)

    def resolve_path(self, path, normpath=normpath):
        """
        Resolve ``path`` to a source and physical path if possible.

        :param path: a path of the form `<source>/<fspath>` or `<fspath>`
        :returns: a tuple of `(source, mapped_path, cache_max_age)`.
                  `source` may be `None` if no source was identified.
        """
        path = normpath(path.lstrip('/'))
        try:
            source, remaining = path.split('/', 1)
        except ValueError:
            source = None
            remaining = path

        if source and source in self.sources:
            map_path, d, cache_max_age = self.sources[source]
            path = map_path(self.resource_manager, join(d, remaining))
            return source, path, cache_max_age

        for s, p, cache_max_age in self.map_path_all_sources(path):
            if isfile(p):
                return s, p, cache_max_age

        return None, path, self.cache_max_age

    def map_path_all_sources(self, path):
        """
        Try to map ``path`` to a file in the list of configured sources.
        :return: a generator yielding mapped filesystem paths.
        """
        for s in reversed(self.sources):
            map_path, d, cache_max_age = self.sources[s]
            yield s, map_path(self.resource_manager, join(d, path)), cache_max_age

    def serve(self, path, serve_static_file=serve_static_file):
        source, fspath, cache_max_age = self.resolve_path(path)
        logger.info("Serving %r from %r", path, fspath)
        response = serve_static_file(fspath)

        headers = []

        if cache_max_age:
            headers.append(('Cache-Control', 'max-age: %d' % cache_max_age))

        if self.access_control_allow_origin:
            headers.append(('Access-Control-Allow-Origin',
                            self.access_control_allow_origin))

        if headers:
            response = response.replace(headers=response.headers + headers)
        return response

    def pathfor(self, path):
        return self.prefix + '/' + path
