import folder_paths as comfy_paths
import glob
import json
import os
import random
from PIL import ImageOps
from .func import *
import numpy as np
import torch


# 该节点用于微调他人节点，用于适配某些工作流场景使用
# 大部分代码来源于：https://github.com/WASasquatch/was-node-suite-comfyui
# 其他代码来源于：

#GLOBALS
#获取当前文件的绝对路径
NODE_FILE = os.path.abspath(__file__)
#NODE_FILE的父目录路径
WAS_SUITE_ROOT = os.path.dirname(NODE_FILE)
#环境变量的值
WAS_CONFIG_DIR = os.environ.get('WAS_CONFIG_DIR', WAS_SUITE_ROOT)
#was_suite_settings.json路径
WAS_DATABASE = os.path.join(WAS_CONFIG_DIR, 'was_suite_settings.json')
#was_history.json路径
WAS_HISTORY_DATABASE = os.path.join(WAS_CONFIG_DIR, 'was_history.json')
WAS_CONFIG_FILE = os.path.join(WAS_CONFIG_DIR, 'was_suite_config.json')
MODELS_DIR =  comfy_paths.models_dir
#文件扩展名的元组
ALLOWED_EXT = ('.jpeg', '.jpg', '.png',
                        '.tiff', '.gif', '.bmp', '.webp')

#! WAS SUITE CONFIG
was_conf_template = {
                    "run_requirements": True,
                    "suppress_uncomfy_warnings": True,
                    "show_startup_junk": True,
                    "show_inspiration_quote": True,
                    "text_nodes_type": "STRING",
                    "webui_styles": None,
                    "webui_styles_persistent_update": True,
                    "blip_model_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth",
                    "blip_model_vqa_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth",
                    "sam_model_vith_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "sam_model_vitl_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "sam_model_vitb_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    "history_display_limit": 36,
                    "use_legacy_ascii_text": False,
                    "ffmpeg_bin_path": "/path/to/ffmpeg",
                    "ffmpeg_extra_codecs": {
                        "avc1": ".mp4",
                        "h264": ".mkv",
                    },
                    "wildcards_path": os.path.join(WAS_SUITE_ROOT, "wildcards"),
                    "wildcard_api": True,
                }

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# Our any instance wants to be a wildcard string
ANY = AnyType("*")

# Create, Load, or Update Config
def getSuiteConfig():
    global was_conf_template
    try:
        with open(WAS_CONFIG_FILE, "r") as f:
            was_config = json.load(f)
    except OSError as e:
        cstr(f"Unable to load conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        return was_conf_template
    except Exception as e:
        cstr(f"Unable to load conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        return was_conf_template
    return was_config

def updateSuiteConfig(conf):
    try:
        with open(WAS_CONFIG_FILE, "w", encoding='utf-8') as f:
            json.dump(conf, f, indent=4)
    except OSError as e:
        print(e)
        return False
    except Exception as e:
        print(e)
        return False
    return True

if not os.path.exists(WAS_CONFIG_FILE):
    if updateSuiteConfig(was_conf_template):
        cstr(f'Created default conf file at `{WAS_CONFIG_FILE}`.').msg.print()
        was_config = getSuiteConfig()
    else:
        cstr(f"Unable to create default conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        was_config = was_conf_template

else:
    was_config = getSuiteConfig()

    update_config = False
    for sett_ in was_conf_template.keys():
        if not was_config.__contains__(sett_):
            was_config.update({sett_: was_conf_template[sett_]})
            update_config = True

    if update_config:
        updateSuiteConfig(was_config)

# WAS Suite Locations Debug
if was_config.__contains__('show_startup_junk'):
    if was_config['show_startup_junk']:
        print(f"Running At: {NODE_FILE}")
        print(f"Running From: {WAS_SUITE_ROOT}")

# Check Write Access
if not os.access(WAS_SUITE_ROOT, os.W_OK) or not os.access(MODELS_DIR, os.W_OK):
    print(f"There is no write access to `{WAS_SUITE_ROOT}` or `{MODELS_DIR}`. Write access is required!")
    exit

# WAS SETTINGS MANAGER
class WASDatabase:
    """
    WAS Suite数据库类提供了一个简单的键值数据库，它使用JSON格式将数据存储在一个平面文件中。每个键值对都与一个类别关联。

    属性：
        filepath (str)：存储数据的JSON文件的路径。
        data (dict)：从JSON文件读取的数据的字典。

    方法：
        insert(category, key, value)：将键值对插入到指定类别的数据库中。
        get(category, key)：从数据库中检索与指定键和类别关联的值。
        update(category, key)：从数据库中更新与指定键和类别关联的值。
        delete(category, key)：从数据库中删除与指定键和类别关联的键值对。
        _save()：将数据库的当前状态保存到JSON文件中。
    """
    def __init__(self, filepath):
        self.filepath = filepath
        try:
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}
    #检查指定的类别是否存在。
    def catExists(self, category):
        return category in self.data
    #检查指定的类别和键是否存在。
    def keyExists(self, category, key):
        return category in self.data and key in self.data[category]
    #将键值对插入到指定类别的数据库中
    def insert(self, category, key, value):
        if not isinstance(category, str) or not isinstance(key, str):
            print("Category and key must be strings")
            return

        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()
    #更新指定类别和键关联的值。
    def update(self, category, key, value):
        if category in self.data and key in self.data[category]:
            self.data[category][key] = value
            self._save()
    #更新指定类别的所有键值对。
    def updateCat(self, category, dictionary):
        self.data[category].update(dictionary)
        self._save()
    #从数据库中检索指定类别和键关联的值。
    def get(self, category, key):
        return self.data.get(category, {}).get(key, None)
    #返回整个数据库的字典。
    def getDB(self):
        return self.data
    #插入一个新的类别。
    def insertCat(self, category):
        if not isinstance(category, str):
            print("Category must be a string")
            return

        if category in self.data:
            print(f"The database category '{category}' already exists!")
            return
        self.data[category] = {}
        self._save()
    #返回指定类别的所有键值对。
    def getDict(self, category):
        if category not in self.data:
            print(f"The database category '{category}' does not exist!")
            return {}
        return self.data[category]
    #从数据库中删除指定类别和键关联的键值对。
    def delete(self, category, key):
        if category in self.data and key in self.data[category]:
            del self.data[category][key]
            self._save()
    #将数据库的当前状态保存到JSON文件中。
    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except FileNotFoundError:
            print(f"Cannot save database to file '{self.filepath}'. "
                 "Storing the data in the object instead. Does the folder and node file have write permissions?")
        except Exception as e:
            print(f"Error while saving JSON data: {e}")

# 初始化settings数据库
WDB = WASDatabase(WAS_DATABASE)

#将新的图片路径添加到历史记录中。
def update_history_images(new_paths):

    HDB = WASDatabase(WAS_HISTORY_DATABASE)
    if HDB.catExists("History") and HDB.keyExists("History", "Images"):
        saved_paths = HDB.get("History", "Images")
        for path_ in saved_paths:
            if not os.path.exists(path_):
                saved_paths.remove(path_)
        if isinstance(new_paths, str):
            if new_paths in saved_paths:
                saved_paths.remove(new_paths)
            saved_paths.append(new_paths)
        elif isinstance(new_paths, list):
            for path_ in new_paths:
                if path_ in saved_paths:
                    saved_paths.remove(path_)
                saved_paths.append(path_)
        HDB.update("History", "Images", saved_paths)
    else:
        if not HDB.catExists("History"):
            HDB.insertCat("History")
        if isinstance(new_paths, str):
            HDB.insert("History", "Images", [new_paths])
        elif isinstance(new_paths, list):
            HDB.insert("History", "Images", new_paths)

# LOAD IMAGE BATCH
class Load_Image_Batch:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "number_image", "random"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": 'Batch 001', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
                "allow_RGBA_output": (["false", "true"],),
                "rename_images": (["false", "true"],),
                "image_number": ("INT",{"default": 0, "min": 0, "max": 150000, "step": 1,"forceInput": False}),

            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING","INT",)
    RETURN_NAMES = ("image","filename_text","image_path","isTrue")
    FUNCTION = "load_batch_images"
    CATEGORY = "Sanmi Nodes/Tweaked Nodes"

    def load_batch_images(self, path, pattern='*', index=0, mode="single_image", label='Batch 001', allow_RGBA_output='false', rename_images='false',filename_text_extension='true', image_number=None):
        single_image_path = ''
        allow_RGBA_output = (allow_RGBA_output == 'true')

        if path == '':
            path = 'C:'
        if not os.path.exists(path):
                return (None,)

        # 创建BatchImageLoader对象并获取图像路径
        fl = self.BatchImageLoader(path, label, pattern)
        # 符合规则的图像升序的绝对路径列表
        new_paths = fl.image_paths

        if mode == 'number_image' and path != 'C:' and rename_images == 'true':
            fl.rename_images_with_sequence(path)

        # 根据加载模式选择加载图像的方式
        if mode == 'single_image':
            image, filename = fl.get_image_by_id(index)
            if image == None:
                print(f"No valid image was found for the inded `{index}`")
                return (None, None)
        if mode == 'incremental_image':
            image, filename = fl.get_next_image()
            if image == None:
                print(f"No valid image was found for the next ID. Did you remove images from the source directory?")
                return (None, None,None)
        if mode == 'number_image':
            image, filename, single_image_path ,isTrue= fl.get_image_by_number(image_number)
            if image == None:
                print(f"No valid image was found for the next ID. Did you remove images from the source directory?")
                return (self.create_black_image(), None, None, isTrue)
        else:
            newindex = int(random.random() * len(fl.image_paths))
            image, filename = fl.get_image_by_id(newindex)
            if image == None:
                print(f"No valid image was found for the next ID. Did you remove images from the source directory?")
                return (None, None)

        # 更新历史图像
        update_history_images(new_paths)

        if not allow_RGBA_output:

            image = image.convert("RGB")

        # 如果不保留文件名的文本扩展名，则去除文件名的扩展名部分
        if filename_text_extension == "false":

            filename = os.path.splitext(filename)[0]

        # 返回将图像转换为张量后的图像和文件名
        return (pil2tensor(image), filename, single_image_path, isTrue)


    class BatchImageLoader:
        def __init__(self, directory_path, label, pattern):
            # 初始化BatchImageLoader对象
            self.WDB = WDB
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()
            stored_directory_path = self.WDB.get('Batch Paths', label)
            stored_pattern = self.WDB.get('Batch Patterns', label)

            # 如果存储的路径或模式与当前路径或模式不匹配，则重置索引和存储的路径和模式
            if stored_directory_path != directory_path or stored_pattern != pattern:
                self.index = 0
                self.WDB.insert('Batch Counters', label, 0)
                self.WDB.insert('Batch Paths', label, directory_path)
                self.WDB.insert('Batch Patterns', label, pattern)

            else:
                self.index = self.WDB.get('Batch Counters', label)

            self.label = label

        def load_images(self, directory_path, pattern):

            # 加载指定路径下的图像文件
            for file_name in glob.glob(os.path.join(glob.escape(directory_path), pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

        def get_image_by_number(self, image_number):
            isTrue = 2
            for file_name in self.image_paths:
                single_image_path = file_name
                file_name_only = os.path.basename(file_name)
                # 提取图像名称中第一个逗号前的字符串
                file_number = file_name_only.split(',')[0]
                # 提取数字部分
                file_number = ''.join(filter(str.isdigit, file_number))
                if file_number == "":
                    continue
                if int(image_number) == int(file_number):
                    i = Image.open(file_name)
                    i = ImageOps.exif_transpose(i)
                    isTrue = 1
                    return (i, os.path.basename(file_name),single_image_path,isTrue)
            return (self.create_black_image(), f"编号{image_number}对应图像不存在，输出512*512黑色图像" , None , isTrue,)

        def get_image_by_id(self, image_id):

            # 根据图像ID获取图像和文件名
            if image_id < 0 or image_id >= len(self.image_paths):
                print(f"Invalid image index `{image_id}`")
                return
            i = Image.open(self.image_paths[image_id])
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(self.image_paths[image_id]))

        def get_next_image(self):

            # 获取下一张图像
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            self.index += 1
            if self.index == len(self.image_paths):
                self.index = 0
            print(f'{self.label} Index: {self.index}')
            self.WDB.insert('Batch Counters', self.label, self.index)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(image_path))

        def get_current_image(self):

            # 获取当前图像的文件名
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            return os.path.basename(image_path)

        def create_black_image(self):
            # Creates a 512x512 black image
            return Image.fromarray(np.zeros((512, 512), dtype=np.uint8))

        def rename_images_with_sequence(self, folder_path):

            # 获取文件夹下所有文件
            files = os.listdir(folder_path)
            # 检查文件是否为图片文件
            def is_valid_image(file):
                return file.lower().endswith(ALLOWED_EXT)
            # 获取第一张图片文件
            first_image = next((file for file in files if is_valid_image(file)), None)
            # 如果没有图片文件，则直接返回
            if not first_image:
                print("没有图片文件")
                return

            # 检查所有图片文件的前缀名是否为纯数字
            all_prefixes_are_digits = all(os.path.splitext(file)[0].isdigit() for file in files if is_valid_image(file))
            if all_prefixes_are_digits:
                print("所有图片文件的前缀名都为纯数字，放弃重命名")
                return

            # 重命名图片文件
            for i, file in enumerate(files):
                if is_valid_image(file):
                    ext = os.path.splitext(file)[1]
                    new_name = f"{i:03d}{ext}"
                    old_path = os.path.join(folder_path, file)
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(old_path, new_path)
            print("图片文件已成功重命名")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 打印文件名和行号

        if kwargs['mode'] != 'single_image':
            return float("NaN")
        else:
            fl = Load_Image_Batch.BatchImageLoader(kwargs['path'], kwargs['label'], kwargs['pattern'])
            filename = fl.get_current_image()
            image = os.path.join(kwargs['path'], filename)
            sha = get_sha256(image)
            return sha

class CounterNode:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_number": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "end_number": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "label": ("STRING", {"default": 'Number 001', "multiline": False}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("number", "now_number_text")
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Tweaked Nodes/Counter"

    def gogo(self, start_number, end_number,label='Number 001',index=0):
        # 获取数字列表
        number_list = list(range(start_number, end_number + 1))
        # 创建NumberListLoader对象
        Nu = self.NumberListLoader(number_list,label)
        # 获取当前数字,更新索引,并添加到WDB数据库中
        number = Nu.get_next_number()
        # 获取文本，提示当前数字
        now_number_text = f"现在输出的编号是：{number}"
        print(now_number_text)
        return (number, now_number_text)

    class NumberListLoader:
        def __init__(self, number_list,label):
            self.WDB = WDB
            # 初始化BatchImageLoader对象
            self.number_list = number_list
            stored_number_list = self.WDB.get('List', label)
            # 如果列表不匹配，则重置列表，并获取当前索引值
            if stored_number_list != number_list:
                self.index = 0
                self.WDB.insert('Counters', label, 0)
                self.WDB.insert('List', label, number_list)
            else:
                self.index = self.WDB.get('Counters', label)
            self.label = label


        def get_next_number(self):
            # 获取下一个数字
            if self.index >= len(self.number_list):
                self.index = 0
            number = self.number_list[self.index]
            self.index += 1
            if self.index == len(self.number_list):
                self.index = 0
            self.WDB.insert('Counters', self.label, self.index)
            return number
        def get_current_number(self):

            # 获取当前图像的文件名
            if self.index >= len(self.number_list):
                self.index = 0
            number = self.number_list[self.index]
            return number
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 打印文件名和行号
            Nu = CounterNode.gogo(kwargs['start_number'], kwargs['label'], kwargs['end_number'])
            number = Nu.get_current_number()
            now_number_text = f"现在输出的数字是：{number}"
            return (number, now_number_text)


class SpecialCounterNode:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_list": ("STRING", {"default": '0,1,2,3,4,5,6,7,8,9,10', "multiline": False}),
                "label": ("STRING", {"default": 'Number 001', "multiline": False}),
            }
        }

    RETURN_TYPES = ("INT","FLOAT","STRING")
    RETURN_NAMES = ("integer_number", "float_number","now_number_text")
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Tweaked Nodes/Counter"

    def gogo(self, number_list, label='Number 001', index=0):
        # 将中文逗号替换为英文逗号
        number_list = number_list.replace('，', ',')
        # 将字符串转换为浮点数列表
        number_list = [float(num) for num in number_list.split(',')]
        # 创建NumberListLoader对象
        Nu = self.NumberListLoader(number_list, label)
        # 获取当前数字,更新索引,并添加到WDB数据库中
        float_number = Nu.get_next_number()
        # 获取整数部分
        integer_number = int(float_number)
        # 获取文本，提示当前数字
        now_number_text = f"现在输出的编号是：{float_number}"
        return (integer_number, float_number, now_number_text)

    class NumberListLoader:
        def __init__(self, number_list, label):
            self.WDB = WDB
            # 初始化BatchImageLoader对象
            self.number_list = number_list
            stored_number_list = self.WDB.get('List', label)
            # 如果列表不匹配，则重置列表，并获取当前索引值
            if stored_number_list != number_list:
                self.index = 0
                self.WDB.insert('Counters', label, 0)
                self.WDB.insert('List', label, number_list)
            else:
                self.index = self.WDB.get('Counters', label)
            self.label = label

        def get_next_number(self):
            # 获取下一个数字
            if self.index >= len(self.number_list):
                self.index = 0
            number = self.number_list[self.index]
            self.index += 1
            if self.index == len(self.number_list):
                self.index = 0
            self.WDB.insert('Counters', self.label, self.index)
            return number

        def get_current_number(self):
            # 获取当前图像的文件名
            if self.index >= len(self.number_list):
                self.index = 0
            number = self.number_list[self.index]
            return number

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 打印文件名和行号
        Nu = CounterNode.gogo(kwargs['number_list'], kwargs['label'])
        number = Nu.get_current_number()
        now_number_text = f"现在输出的数字是：{number}"
        return (int(number), number, now_number_text)

class StringCounterNode:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"default": '小明,小红,小蓝,任意字符串', "multiline": False}),
                "label": ("STRING", {"default": 'Label 001', "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("current_string",)
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Tweaked Nodes/Counter"

    def gogo(self, string_list, label='Label 001', index=0):
        # 将中文逗号替换为英文逗号
        string_list = string_list.replace('，', ',')
        # 将字符串转换为字符串列表
        string_list = string_list.split(',')
        # 创建StringListLoader对象
        loader = self.StringListLoader(string_list, label)
        # 获取当前字符串,更新索引,并添加到WDB数据库中
        current_string = loader.get_next_string()
        return (current_string,)

    class StringListLoader:
        def __init__(self, string_list, label):
            self.WDB = WDB
            # 初始化列表
            self.string_list = string_list
            stored_string_list = self.WDB.get('List', label)
            # 如果列表不匹配，则重置列表，并获取当前索引值
            if stored_string_list != string_list:
                self.index = 0
                self.WDB.insert('Counters', label, 0)
                self.WDB.insert('List', label, string_list)
            else:
                self.index = self.WDB.get('Counters', label)
            self.label = label

        def get_next_string(self):
            # 获取下一个字符串
            if self.index >= len(self.string_list):
                self.index = 0
            current_string = self.string_list[self.index]
            self.index += 1
            if self.index == len(self.string_list):
                self.index = 0
            self.WDB.insert('Counters', self.label, self.index)
            return current_string

        def get_current_string(self):
            # 获取当前字符串
            if self.index >= len(self.string_list):
                self.index = 0
            current_string = self.string_list[self.index]
            return current_string

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 打印文件名和行号
        loader = SpecialCounterNode.StringListLoader(kwargs['string_list'], kwargs['label'])
        current_string = loader.get_current_string()
        return (current_string,)

class StringCounterNode_V2:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING",
                                  {
                                      "default": "任意字符串，每行代表一个单位",
                                      "multiline": True, "dynamicPrompts": False
                                  }),
                "label": ("STRING", {"default": 'Label 001', "multiline": False}),
            }
        }

    RETURN_TYPES = (ANY, "INT")
    RETURN_NAMES = ("current_string", "line_count")  # 添加返回名称
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Tweaked Nodes/Counter"

    def gogo(self, string_list, label='Label 001', index=0):
        # 将字符串分割为列表并计算行数
        lines = string_list.split('\n')
        line_count = len(lines)

        # 创建 StringListLoader 对象
        loader = self.StringListLoader(lines, label)
        # 获取当前字符串
        current_string = loader.get_next_string()
        return (current_string, line_count)  # 返回两个值

    class StringListLoader:
        def __init__(self, string_list, label):
            self.WDB = WDB
            # 初始化列表
            self.string_list = string_list
            stored_string_list = self.WDB.get('List', label)
            # 如果列表不匹配，则重置列表，并获取当前索引值
            if stored_string_list != string_list:
                self.index = 0
                self.WDB.insert('Counters', label, 0)
                self.WDB.insert('List', label, string_list)
            else:
                self.index = self.WDB.get('Counters', label)
            self.label = label

        def get_next_string(self):
            # 获取下一个字符串
            if self.index >= len(self.string_list):
                self.index = 0
            current_string = self.string_list[self.index]
            self.index += 1
            if self.index == len(self.string_list):
                self.index = 0
            self.WDB.insert('Counters', self.label, self.index)
            return current_string

        def get_current_string(self):
            # 获取当前字符串
            if self.index >= len(self.string_list):
                self.index = 0
            current_string = self.string_list[self.index]
            return current_string

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 打印文件名和行号
        loader = SpecialCounterNode.StringListLoader(kwargs['string_list'], kwargs['label'])
        current_string = loader.get_current_string()
        return (current_string,)

class SanmiFlorence2toCoordinates:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("JSON",),
                "index": ("STRING", {"default": "0"}),
                "batch": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "BBOX")
    RETURN_NAMES = ("center_coordinates", "bboxes")
    FUNCTION = "segment"
    CATEGORY = "Sanmi Nodes/Tweaked Nodes"

    def segment(self, data, index, batch=False):
        print(data)
        try:
            coordinates = coordinates.replace("'", '"')
            coordinates = json.loads(coordinates)
        except:
            coordinates = data
        #print("Type of data:", type(data))
        #print("Data:", data)
        if len(data) == 0:
            return (json.dumps([{'x': 0, 'y': 0}]),)
        center_points = []

        if index.strip():  # Check if index is not empty
            indexes = [int(i) for i in index.split(",")]
        else:  # If index is empty, use all indices from data[0]
            indexes = list(range(len(data[0])))

        #print("Indexes:", indexes)
        bboxes = []

        if batch:
            for idx in indexes:
                if 0 <= idx < len(data[0]):
                    for i in range(len(data)):
                        bbox = data[i][idx]
                        min_x, min_y, max_x, max_y = bbox
                        center_x = int((min_x + max_x) / 2)
                        center_y = int((min_y + max_y) / 2)
                        center_points.append({"x": center_x, "y": center_y})
                        bboxes.append(bbox)
                else:
                    # Default to index 0 if idx is out of range
                    for i in range(len(data)):
                        bbox = data[i][0]
                        min_x, min_y, max_x, max_y = bbox
                        center_x = int((min_x + max_x) / 2)
                        center_y = int((min_y + max_y) / 2)
                        center_points.append({"x": center_x, "y": center_y})
                        bboxes.append(bbox)
        else:
            for idx in indexes:
                if 0 <= idx < len(data[0]):
                    bbox = data[0][idx]
                    min_x, min_y, max_x, max_y = bbox
                    center_x = int((min_x + max_x) / 2)
                    center_y = int((min_y + max_y) / 2)
                    center_points.append({"x": center_x, "y": center_y})
                    bboxes.append(bbox)
                else:
                    # Default to index 0 if idx is out of range
                    bbox = data[0][0]
                    min_x, min_y, max_x, max_y = bbox
                    center_x = int((min_x + max_x) / 2)
                    center_y = int((min_y + max_y) / 2)
                    center_points.append({"x": center_x, "y": center_y})
                    bboxes.append(bbox)

        coordinates = json.dumps(center_points)
        #print("Coordinates:", coordinates)
        return (coordinates, bboxes)

class SANMIN_Adapt_Coordinates:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {"multiline": False, "forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coord_str",)
    FUNCTION = "adapt_coordinates"
    CATEGORY = "Sanmi Nodes/Tweaked Nodes"

    def adapt_coordinates(self, coordinates, width, height):
        scale_factor_x = round(width / 512)
        scale_factor_y = round(height / 512)

        coordinates_out = [
            {
                "x": coord["x"] * scale_factor_x if coord["x"] != 0 else 0,
                "y": coord["y"] * scale_factor_y if coord["y"] != 0 else 0,
            }
            for coord in eval(coordinates)
        ]

        coordinates_out = str(coordinates_out)
        # print("coordinates_out1:", type(coordinates_out), coordinates_out)
        return (coordinates_out,)

# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "sanmi Load Image Batch": Load_Image_Batch,
    "SANMIN Adapt Coordinates": SANMIN_Adapt_Coordinates,
    "sanmi Florence2toCoordinates": SanmiFlorence2toCoordinates,

    "sanmi Counter": CounterNode,
    "sanmi Special Counter": SpecialCounterNode,
    "sanmi String Counter": StringCounterNode,
    "sanmi String Counter V2": StringCounterNode_V2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "sanmi Load Image Batch": "Sanmi Load Image Batch",
    "SANMIN Adapt Coordinates": "Sanmi Adapt_Coordinates",
    "sanmi Florence2toCoordinates": "Sanmi Florence2toCoordinates",

    "sanmi Counter": "Counter",
    "sanmi Special Counter": "Special Counter",
    "sanmi String Counter": "String Counter",
    "sanmi String Counter V2": "String Counter V2",
}