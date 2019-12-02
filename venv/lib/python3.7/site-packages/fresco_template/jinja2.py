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

from .core import TemplateEnvironment, BasePlugin


class Jinja2Plugin(BasePlugin):

    def __init__(self, *args, **kwargs):
        from jinja2 import Environment
        if (not kwargs and len(args) == 1 and
                isinstance(args[0], Environment)):
            self.loader = args[0]
        else:
            self.loader = Environment(**kwargs)

    def _get_template(self, template):
        return

    def _render(self, template_name, data, stream):
        template = self.loader.get_template(template_name)
        if stream:
            return template.generate(data)
        return template.render(data)


class Jinja2(TemplateEnvironment):
    plugin_class = Jinja2Plugin
