# Copyright 2018 Oliver Cope
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
from __future__ import unicode_literals
from os.path import join as pjoin
from os.path import dirname
from tempfile import mkdtemp
import contextlib
import io
import os
import shutil

from fresco_static import StaticFiles


@contextlib.contextmanager
def tmpdir():
    t = mkdtemp()
    yield t
    shutil.rmtree(t)


@contextlib.contextmanager
def create_tree(paths):
    """
    Write files to a temporary directory

    :param paths: list of (path, content) tuples
    :returns: path to temporary directory
    """
    if isinstance(paths, dict):
        paths = paths.items()

    with tmpdir() as d:
        paths = [(pjoin(d, *p.split('/')), content) for p, content in paths]
        for p, c in paths:
            try:
                os.makedirs(dirname(p))
            except OSError:
                pass
            with io.open(p, 'w', encoding='UTF-8') as f:
                f.write(c)

        try:
            yield d
        except Exception:
            import traceback
            traceback.print_exc()
            raise


def test_it_resolves_source_prefix():

    sf = StaticFiles(cache_max_age=0)
    with create_tree({'x/foo': '', 'y/foo': ''}) as d:
        sf.add_directory('x', '{}/x'.format(d))
        sf.add_directory('y', '{}/y'.format(d))
        assert sf.resolve_path('x/foo') == ('x', '{}/x/foo'.format(d), 0)
        assert sf.resolve_path('y/foo') == ('y', '{}/y/foo'.format(d), 0)


def test_it_resolves_without_source_prefix():

    sf = StaticFiles(cache_max_age=0)
    with create_tree({'x/foo': '', 'y/foo/bar': ''}) as d:
        sf.add_directory('x', '{}/x'.format(d))
        sf.add_directory('y', '{}/y'.format(d))
        assert sf.resolve_path('foo') == ('x', '{}/x/foo'.format(d), 0)
        assert sf.resolve_path('foo/bar') == ('y', '{}/y/foo/bar'.format(d), 0)


def test_it_sets_per_source_cache_age():
    sf = StaticFiles(cache_max_age=10)
    with create_tree({'x/foo': '', 'y/foo': '', 'z/foo': ''}) as d:
        sf.add_directory('x', '{}/x'.format(d))
        sf.add_directory('y', '{}/y'.format(d), cache_max_age=20)
        sf.add_directory('z', '{}/z'.format(d), cache_max_age=30)
        assert sf.resolve_path('x/foo') == ('x', '{}/x/foo'.format(d), 10)
        assert sf.resolve_path('y/foo') == ('y', '{}/y/foo'.format(d), 20)
        assert sf.resolve_path('z/foo') == ('z', '{}/z/foo'.format(d), 30)
