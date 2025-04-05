__version__ = "1.0.0"

import os
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_MODULES = ["simple_nodes", "tweaked_nodes", "private_nodes"]  # 全部模块列表

for module in NODE_MODULES:
    try:
        # 动态导入模块
        imported = importlib.import_module(f".nodes.{module}", __name__)
        # 静默更新节点映射，即使imported没有这些属性也不会报错（会跳过）
        NODE_CLASS_MAPPINGS.update(getattr(imported, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(imported, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception:  # 捕获任何异常
        pass  # 空操作，相当于完全静默

# Web资源目录配置
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']