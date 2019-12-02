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

from .core import BasePlugin, TemplateEnvironment


class KajikiPlugin(BasePlugin):

    def __init__(self, loader):
        self.loader = loader

    def render(self, template_name, context, stream):
        tmpl = self.loader.import_(template_name)(context)
        if stream:
            return tmpl
        return tmpl.render()


class Kajiki(TemplateEnvironment):
    plugin_class = KajikiPlugin
