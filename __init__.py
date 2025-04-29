__version__ = "1.0.0"

import os
import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_MODULES = ["simple_nodes", "tweaked_nodes", "private_nodes"]

for module in NODE_MODULES:
    try:
        imported = importlib.import_module(f".nodes.{module}", __name__)
        NODE_CLASS_MAPPINGS.update(getattr(imported, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(imported, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as e:
        ##############################################
        # 仅当非 private_nodes 时才显示错误
        if module != "private_nodes":
            print(f"⚠️ 加载模块 '{module}' 时出错:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            print("   堆栈跟踪:")
            traceback.print_exc()
            print("-" * 50)
        ##############################################

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
