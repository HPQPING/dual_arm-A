"""从任意工作目录运行时，都只检查当前功能包的文档字符串."""

# Copyright 2015 Open Source Robotics Foundation, Inc.
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


from pathlib import Path

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257() -> None:
    """用测试文件所在位置定位功能包，避免依赖当前终端目录."""
    package_root = Path(__file__).resolve().parents[1]
    return_code = main(argv=[str(package_root)])
    assert return_code == 0, '发现文档字符串样式问题，请查看上面的详细输出'
