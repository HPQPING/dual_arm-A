"""仅检查当前功能包的 Python 样式，并处理检查器初始化失败."""

# Copyright 2017 Open Source Robotics Foundation, Inc.
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

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8() -> None:
    """显式区分错误码和正常结果，避免对整数执行解包."""
    package_root = Path(__file__).resolve().parents[1]
    result = main_with_errors(argv=[str(package_root)])
    # Humble 的检查器在配置不存在时只返回整数，不能直接解包。
    if isinstance(result, int):
        raise AssertionError(
            f'flake8 未完成检查，返回码 {result}，请检查其配置文件',
        )
    return_code, errors = result
    assert return_code == 0, (
        f'发现 {len(errors)} 项代码样式问题：\n' + '\n'.join(errors)
    )
