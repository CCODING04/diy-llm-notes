"""Pytest 配置文件

添加项目根目录到 sys.path，以便导入 scripts/ 中的模块。
"""
import sys
from pathlib import Path

# 将 homework/assignment2/ 添加到 sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
