import os
import re
import cv2
import json
import copy
import random
import string
import ast
import pilgram
from datetime import datetime
from collections import Counter

import pandas as pd
import piexif
import piexif.helper
import scipy.ndimage
import torchvision.transforms as transforms
from PIL import ImageOps, ImageSequence, ImageColor, PngImagePlugin
from pypinyin import lazy_pinyin
from stegano import lsb

import model_management
import folder_paths
import node_helpers
from nodes import SaveImage
from .func import *

import comfy.utils
# import comfy.sample
# import comfy.samplers
# import comfy.sd
# import comfy.latent_formats

def shape(image):
    image_shape = image.size()
    print("image_shape:", image_shape)
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.1})},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Input"

    def execute(self, value):
        return (value,)

class Int90:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 90}),  # 旋转角度，默认为0，范围0到360，步长90
                 },

        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Input"

    def execute(self, rotation):
        return (rotation,)

class AdjustTransparencyByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "mask": ("MASK",),    # 输入遮罩
                "invert": ("BOOLEAN", {"default": False}),  # 是否反转遮罩
                "opacity": ("FLOAT", {"default": 100, "min": 0, "max": 100, "step": 5}),  # 透明度调整值
            },
        }

    RETURN_TYPES = ("IMAGE",)  # 返回调整后的图像
    FUNCTION = "adjust_transparency"  # 主函数
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"  # 节点分类

    def adjust_transparency(self, image, mask, opacity, invert=False):
        # 转换张量到PIL图像
        image = tensor2pil(image)
        mask = tensor2pil(mask).convert('L')

        # 确保图像为RGBA格式
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        if invert:
            mask = ImageOps.invert(mask)
        # 调整遮罩尺寸匹配图像
        mask = mask.resize(image.size)

        # 转换为numpy数组并归一化
        original_alpha = np.array(image.getchannel('A'), dtype=np.float32) / 255.0
        mask_array = np.array(mask, dtype=np.float32) / 255.0

        # 计算透明度因子
        opacity_factor = opacity / 100.0

        # 计算新alpha通道（保留非遮罩区域原始透明度）
        new_alpha = original_alpha * (1 - mask_array) + opacity_factor * mask_array
        new_alpha = (new_alpha * 255).clip(0, 255).astype(np.uint8)

        # 更新图像alpha通道
        image.putalpha(Image.fromarray(new_alpha, mode='L'))

        return (pil2tensor(image),)

class ChineseToCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chinese_name": ("STRING", {"multiline": False, "default": "填写角色中文名",}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "find_character"
    CATEGORY = "Sanmi Nodes/Special Nodes"
    DESCRIPTION = "填写角色中文名自动匹配最接近的英文提示词，用于搭配animagine-xl模型使用，也可自行查看animagine-xl-3.1-characterfull-zh.txt."

    @staticmethod
    # 读取角色英文名
    def find_character(chinese_name):
        # 获取当前代码文件的目录路径
        code_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建文件的完整路径
        file_path = os.path.join(code_dir, "animagine-xl-3.1-characterfull-zh.txt")

        # Read the file
        df = pd.read_csv(file_path, sep=",|#", engine='python', header=None)

        # Rename columns
        df.columns = ["Gender", "Character_EN", "Anime_EN", "Character_CN", "Anime_CN"]

        # Find character
        result = df[df['Character_CN'].str.contains(chinese_name)]

        # If result is found, return the corresponding information
        if not result.empty:
            gender = result['Gender'].values[0]
            character_en = result['Character_EN'].values[0]
            anime_en = result['Anime_EN'].values[0]
            character = f"{gender}, {character_en}, {anime_en}"

            return (character,)
        else:
            return ("Character not found",)

class ReadImagePrompt:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
            {
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    # 类别定义
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"

    # 返回值类型定义
    RETURN_TYPES = ("IMAGE","STRING","STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT")

    # 返回值名称定义
    RETURN_NAMES = ("image","all_prompt","positive", "negative", "seed", "steps", "cfg", "width", "height")

    # 函数名称定义
    FUNCTION = "get_image_data"

    def get_image_data(self, image,):
        # 获取图像的路径
        image_path = folder_paths.get_annotated_filepath(image)

        # 获取文件扩展名
        extension = image_path.split('.')[-1]

        # 使用 pillow 打开图像
        img = node_helpers.pillow(Image.open, image_path)

        # 初始化输出图像和掩码的列表
        output_images = []
        output_masks = []
        w, h = None, None

        # 排除的图像格式
        excluded_formats = ['MPO']

        # 迭代图像序列
        for i in ImageSequence.Iterator(img):
            # 处理图像的 EXIF 信息
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            # 如果图像模式为 'I'，则进行转换
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            # 初始化图像的宽度和高度
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            # 如果图像大小不一致，则跳过
            if image.size[0] != w or image.size[1] != h:
                continue

            # 转换图像为 numpy 数组并归一化
            image = np.array(image).astype(np.float32) / 255.0
            # 转换为 PyTorch 张量并添加维度
            image = torch.from_numpy(image)[None,]
            # 检查图像中是否有 'A' 通道
            if 'A' in i.getbands():
                # 获取 'A' 通道的掩码并归一化
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                # 反转掩码
                mask = 1. - torch.from_numpy(mask)
            else:
                # 如果没有 'A' 通道，则创建一个全零的掩码
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            # 添加图像和掩码到输出列表
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        # 如果有多个图像并且格式不在排除列表中，则合并图像和掩码
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            # 否则只使用第一个图像和掩码
            output_image = output_images[0]
            output_mask = output_masks[0]


        parameters = ""
        comfy = False
        if extension.lower() == 'png':
            try:
                # 尝试从 PNG 图像的元信息中提取参数
                parameters = img.info['parameters']
                if not parameters.startswith("Positive prompt"):
                    parameters = "Positive prompt: " + parameters
            except:
                parameters = ""
                print("Error loading prompt info from png")
        elif extension.lower() in ("jpg", "jpeg", "webp"):
            try:
                # 尝试从 JPEG 图像的 EXIF 信息中提取参数
                exif = piexif.load(img.info["exif"])
                parameters = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
                parameters = piexif.helper.UserComment.load(parameters)
                if not parameters.startswith("Positive prompt"):
                    parameters = "Positive prompt: " + parameters
            except:
                try:
                    # 尝试从 JPEG 图像的评论信息中提取参数
                    parameters = str(img.info['comment'])
                    comfy = True
                    # 修复旧版本中的问题
                    parameters = parameters.replace("Positive Prompt", "Positive prompt")
                    parameters = parameters.replace("Negative Prompt", "Negative prompt")
                    parameters = parameters.replace("Start at Step", "Start at step")
                    parameters = parameters.replace("End at Step", "End at step")
                    parameters = parameters.replace("Denoising Strength", "Denoising strength")
                except:
                    parameters = ""
                    print("Error loading prompt info from jpeg")

        # 根据图像类型调整参数中的换行符处理
        if (comfy and extension.lower() == 'jpeg'):
            parameters = parameters.replace('\\n', ' ')
        else:
            parameters = parameters.replace('\n', ' ')

        # 定义需要匹配的参数模式
        patterns = [
            "Positive prompt: ",
            "Negative prompt: ",
            "Steps: ",
            "Start at step: ",
            "End at step: ",
            "Sampler: ",
            "Scheduler: ",
            "CFG scale: ",
            "Seed: ",
            "Size: ",
            "Model: ",
            "Model hash: ",
            "Denoising strength: ",
            "Version: ",
            "ControlNet 0",
            "Controlnet 1",
            "Batch size: ",
            "Batch pos: ",
            "Hires upscale: ",
            "Hires steps: ",
            "Hires upscaler: ",
            "Template: ",
            "Negative Template: ",
        ]
        if (comfy and extension.lower() == 'jpeg'):
            parameters = parameters[2:]
            parameters = parameters[:-1]

        # 使用正则表达式匹配参数键和值
        keys = re.findall("|".join(patterns), parameters)
        values = re.split("|".join(patterns), parameters)
        values = [x for x in values if x]
        results = {}
        result_string = ""
        for item in range(len(keys)):
            result_string += keys[item] + values[item].rstrip(', ')
            result_string += "\n"
            results[keys[item].replace(": ", "")] = values[item].rstrip(', ')

        # 提取正面提示
        try:
            positive = results['Positive prompt']
        except:
            positive = ""

        # 提取负面提示
        try:
            negative = results['Negative prompt']
        except:
            negative = ""

        # 提取种子值
        try:
            seed = int(results['Seed'])
        except:
            seed = -1

        # 提取步骤数
        try:
            steps = int(results['Steps'])
        except:
            steps = 20

        # 提取 CFG 值
        try:
            cfg = float(results['CFG scale'])
        except:
            cfg = 8.0

        # 提取图像宽度和高度
        try:
            width, height = img.size
        except:
            width, height = 512, 512

        ''' 返回值顺序：
            所有字符串, 正面提示（字符串），负面提示（字符串），种子（整数），步骤（整数），CFG（浮点数），
            宽度（整数），高度（整数）
        '''

        return (output_image, result_string, positive, negative, seed, steps, cfg, width, height)

def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

class SaveImageToLocal:

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                  {
                    "images": ("IMAGE",),
                    "filename": ("STRING", {"default": "ComfyUI"}),
                    "file_path": ("STRING", {"multiline": False, "default": "", "dynamicPrompts": False}),
                    "isTrue": ("INT", {"default": 1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,}),
                    "generate_txt": ("BOOLEAN", {"default": False}),
                    "overwrite_image": ("BOOLEAN", {"default": False}),
                    "txt_content": ("STRING", {"multiline": True}),  # 使用box
                  },
                  "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "str",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"

    def save(self, images, file_path, isTrue=1, seed=None, filename="ComfyUI", prompt=None, extra_pnginfo=None, generate_txt=False, overwrite_image=False, txt_content=""):
        file_path = file_path.strip('"')
        if isTrue == 0:
            return (images,"isTrue为0时，不做任何处理",)
        if file_path == "":
            SaveImage().save_images(images, filename, prompt, extra_pnginfo)
            return (images,"确保路径不为空",)

        if not os.path.exists(file_path):
            # 使用os.makedirs函数创建新目录
            os.makedirs(file_path)

        def generate_unique_filename(base_name, extension, path):
            counter = 1
            unique_name = f"{base_name}{extension}"
            while os.path.exists(os.path.join(path, unique_name)):
                unique_name = f"{base_name}({counter}){extension}"
                counter += 1
            return unique_name

        for idx, image in enumerate(images):
            # 将图像张量从GPU转移到CPU，并转换为NumPy数组，然后将其缩放到0-255范围
            i = 255. * image.cpu().numpy()
            # 将NumPy数组转换为PIL图像，并确保像素值在0到255之间
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 根据索引确定文件名
            if idx == 0:
                file = f"{filename}.png"
            else:
                file = f"{filename}_{idx:03d}.png"

            # 组合文件路径
            fp = os.path.join(file_path, file)

            # 检查文件是否存在且不允许覆盖
            if os.path.exists(fp) and not overwrite_image:
                base_name, extension = os.path.splitext(file)
                file = generate_unique_filename(base_name, extension, file_path)
                fp = os.path.join(file_path, file)

            # 保存图像到文件
            img.save(fp)

            if generate_txt:
                txt_filename = os.path.splitext(file)[0] + ".txt"
                txt_filepath = os.path.join(file_path, txt_filename)
                with open(txt_filepath, "w") as f:
                    f.write(txt_content)

        return (images, "完成",)

class Upscale_And_Keep_Original_Size:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale_by": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 9.99, "step": 0.01}),
                "image": ("IMAGE", ),
                "fill_color": (["Custom", "white", "yellow", "black", "red", "gray", "blue", "green"], {"default": "white"}),
                "color_hex": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_and_original_size"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"

    def upscale_and_original_size(self, scale_by, image, fill_color, color_hex):
        # 将输入图像转换为 PIL 格式
        image_pil = tensor2pil(image)
        original_size = image_pil.size

        # 计算缩放后的新尺寸
        new_width = int(original_size[0] * scale_by)
        new_height = int(original_size[1] * scale_by)

        # 调整图像大小
        resized_image = image_pil.resize((new_width, new_height), resample=Image.LANCZOS)

        # 选择背景颜色
        background_color = color_hex if fill_color == "Custom" else fill_color
        background = Image.new("RGB", original_size, ImageColor.getrgb(background_color))

        # 计算居中粘贴位置
        paste_x = (original_size[0] - new_width) // 2
        paste_y = (original_size[1] - new_height) // 2

        # 将缩放后的图像粘贴到背景中心
        background.paste(resized_image, (paste_x, paste_y))

        # 返回张量格式结果
        return (pil2tensor(background),)

def binarize_mask(mask, threshold=0.5):
    """
    将遮罩进行二值化处理。

    参数：
    mask (Tensor): 输入的遮罩。
    threshold (float): 二值化的阈值，默认为0.5。

    返回：
    Tensor: 二值化后的遮罩。
    """
    return (mask > threshold).float()

class BinarizeMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/Mask"
    FUNCTION = "binarize_mask"

    def binarize_mask(self,mask, threshold=0.5):
        """
        将遮罩进行二值化处理。

        参数：
        mask (Tensor): 输入的遮罩。
        threshold (float): 二值化的阈值，默认为0.5。

        返回：
        Tensor: 二值化后的遮罩。
        """
        return ((mask > threshold).float(),)

class GetWhiteRegionSize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width","height",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/Mask"
    FUNCTION = "get_mask_size"

    def get_mask_size(self, mask: torch.Tensor):
        # 先将遮罩二值化
        mask = binarize_mask(mask)
        # 获取遮罩处白色区域的宽高
        white_mask_indices = torch.nonzero(mask == 1, as_tuple=True)
        if len(white_mask_indices[0]) == 0 or len(white_mask_indices[1]) == 0:
            mask_height, mask_width = 0, 0
        else:
            mask_height = white_mask_indices[2].max().item() - white_mask_indices[2].min().item() + 1
            mask_width = white_mask_indices[1].max().item() - white_mask_indices[1].min().item() + 1

        return (mask_height,mask_width,)

class MaskToBox:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "mask": ("MASK",),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("rectangular_mask",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/To"
    FUNCTION = "get_rectangular_mask"
    DESCRIPTION = "将遮罩白色区域转化为矩形"

    def get_rectangular_mask(self, mask: torch.Tensor, top_reserve: int, bottom_reserve: int, left_reserve: int, right_reserve: int):
        # 遮罩二值化
        mask = binarize_mask(mask)
        # 获取遮罩处白色区域的宽高和位置
        white_mask_indices = torch.nonzero(mask == 1, as_tuple=True)
        if len(white_mask_indices[0]) == 0 or len(white_mask_indices[1]) == 0:
            return mask  # 如果没有白色区域，返回原蒙版

        min_y = white_mask_indices[1].min().item()
        max_y = white_mask_indices[1].max().item()
        min_x = white_mask_indices[2].min().item()
        max_x = white_mask_indices[2].max().item()

        # 扩展边界
        min_y = max(0, min_y - top_reserve)
        max_y = min(mask.shape[1] - 1, max_y + bottom_reserve)
        min_x = max(0, min_x - left_reserve)
        max_x = min(mask.shape[2] - 1, max_x + right_reserve)

        # 创建一个全黑的蒙版
        rectangular_mask = torch.zeros_like(mask)

        # 将矩形区域设置为白色
        rectangular_mask[:, min_y:max_y + 1, min_x:max_x + 1] = 1

        return (rectangular_mask,)

class PathChange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "pattern": (["add", "sub"],),
                "path": ("STRING", {"multiline": False}),
                "add_file": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/Path"
    FUNCTION = "gogo"
    DESCRIPTION = "输入路径，给路径增加或者减少一级"

    def gogo(self, pattern, path, add_file):
        path = path.strip('"')
        # 统一正反斜杠
        normalized_path = os.path.normpath(path)
        if pattern == "sub":
            # 返回上一级路径
            path = os.path.dirname(normalized_path)
        elif pattern == "add":
            # 在原有路径上增加add_file文件夹
            path = os.path.join(normalized_path, add_file)
        return (path,)

# 将布尔蒙版应用到原始遮罩，保留目标区域的原始强度信息。
def apply_mask_with_intensity(
    original_mask: torch.Tensor,  # 原始遮罩，包含强度信息
    mask_area: np.ndarray,        # 布尔蒙版，表示目标区域
) -> torch.Tensor:
    """
    将布尔蒙版应用到原始遮罩，保留目标区域的原始强度信息。

    参数:
        original_mask (torch.Tensor): 原始遮罩，形状为 (H, W) 或 (B, H, W)，包含强度信息。
        mask_area (np.ndarray): 布尔蒙版，形状为 (H, W)，表示目标区域。

    返回:
        torch.Tensor: 应用蒙版后的遮罩，保留目标区域的原始强度。
    """
    # 将布尔蒙版转换为与原始遮罩相同类型和设备的张量
    mask_tensor = torch.from_numpy(mask_area).to(
        device=original_mask.device,
        dtype=original_mask.dtype
    )

    # 应用蒙版到原始遮罩
    return original_mask * mask_tensor

class SortTheMasksSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "num_masks": ("INT", {"default": 1, "min": 0, "step": 1}),
                "return_choice": (
                    ["black", "all"],
                    {"default": "black", "tooltip": "找不到目标遮罩时返回黑色或原图"}
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/Mask"
    FUNCTION = "process_masks"
    DESCRIPTION = "按白色区域面积降序排列并输出指定排名的遮罩（保留原图强度）"

    def process_masks(self, mask: torch.Tensor, num_masks: int, return_choice: str):
        # 保留原始遮罩的强度信息
        original_mask = mask.clone()

        # 转换为CPU上的布尔数组处理
        mask_np = mask.bool().cpu().numpy()

        # 寻找连通区域
        labeled, regions = scipy.ndimage.label(mask_np)
        if regions == 0:
            return (original_mask,)

        # 计算区域面积并排序
        areas = scipy.ndimage.sum(mask_np, labeled, range(1, regions + 1))
        sorted_idx = np.argsort(areas)[::-1]

        # 处理无效序号
        if not (0 < num_masks <= len(sorted_idx)):
            if return_choice == "black":
                return (torch.zeros_like(original_mask),)
            return (original_mask,)

        # 生成目标遮罩的布尔蒙版
        target_label = sorted_idx[num_masks - 1] + 1
        mask_area = (labeled == target_label)

        # 将布尔蒙版转换为与原遮罩相同设备和类型的强度遮罩
        mask_tensor = torch.from_numpy(mask_area).to(
            device=original_mask.device,
            dtype=original_mask.dtype
        )

        # 应用蒙版到原始遮罩（保留原强度）
        result = original_mask * mask_tensor

        return (result,)

class PathCaptioner:
    """路径标注处理器，用于为图像文件生成/更新文本标注"""

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数格式"""
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),  # 图像文件/文件夹路径
                "txt_content": ("STRING", {  # 要写入的文本内容
                    "multiline": True,
                    "default": ""
                }),
                "ignore_if_exists": ("BOOLEAN", {"default": False}),  # 是否跳过已有标注文件
                "mode": ([  # 写入模式
                             "overwrite",  # 覆盖模式
                             "append_at_the_beginning",  # 开头追加
                             "append_at_the_end"  # 末尾追加
                         ], {"default": "overwrite"})
            }
        }

    RETURN_TYPES = ("STRING",)  # 返回类型为字符串
    RETURN_NAMES = ("string",)  # 返回参数名称为string
    CATEGORY = "Sanmi Nodes/Basics Nodes/Path"
    FUNCTION = "path_captioner"  # 主处理函数

    def path_captioner(self, folder_path, txt_content, ignore_if_exists, mode):
        folder_path = folder_path.strip('"')
        # 支持的图像文件扩展名
        ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

        """主处理函数"""
        # 验证路径类型并获取图像列表
        if os.path.isfile(folder_path):
            # 单个文件处理
            if not folder_path.lower().endswith(ALLOWED_EXTENSIONS):
                return ("错误：文件格式不支持",)
            image_paths = [folder_path]
        elif os.path.isdir(folder_path):
            # 文件夹处理（扫描所有图像文件）
            image_paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(ALLOWED_EXTENSIONS)
            ]
        else:
            return ("错误：路径无效",)

        success_count = 0
        # 遍历处理每个图像文件
        for img_path in image_paths:
            txt_path = os.path.splitext(img_path)[0] + ".txt"

            # 跳过已存在的标注文件
            if ignore_if_exists and os.path.exists(txt_path):
                continue

            # 根据模式执行不同写入操作
            if mode == "overwrite":
                with open(txt_path, "w") as f:
                    f.write(txt_content)
            elif mode == "append_at_the_beginning":
                # 读取原有内容并前置新内容
                existing = ""
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        existing = f.read()
                with open(txt_path, "w") as f:
                    f.write(txt_content + existing)
            elif mode == "append_at_the_end":
                # 在文件末尾追加内容
                with open(txt_path, "a") as f:
                    f.write(txt_content)

            success_count += 1

        # 生成结果反馈信息
        if os.path.isfile(folder_path):
            return (f"成功标注文件：{os.path.basename(folder_path)}",)
        return (f"成功处理 {success_count} 个文件",)

class SanmiNothing:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "nothing_boolean": ("BOOLEAN", {"default": False}),
            },
            # 可选项
            "optional": {
                "nothing1": (AlwaysEqualProxy("*"), {}),
                "nothing2": (AlwaysEqualProxy("*"), {}),
                "nothing3": (AlwaysEqualProxy("*"), {}),
                "nothing4": (AlwaysEqualProxy("*"), {}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),AlwaysEqualProxy("*"),AlwaysEqualProxy("*"),AlwaysEqualProxy("*"),"STRING",)
    RETURN_NAMES = ("nothing1","nothing2","nothing3","nothing4","string",)
    CATEGORY = "Sanmi Nodes/Basics Nodes"
    FUNCTION = "Nothing"
    DESCRIPTION = "用于节点排序运行，避免部分节点提前运行，输入什么就输出什么，并无其他改变"

    def Nothing(self, nothing1=None, nothing2=None,nothing3=None,nothing4=None,nothing_boolean=False, **kwargs):
        return (nothing1,nothing2,nothing3,nothing4,"nothing")

def get_white_mask_dimensions(mask):
    """
    获取遮罩处白色区域的宽高。

    参数:
    mask (torch.Tensor): 输入的遮罩张量。

    返回:
    tuple: (mask_height, mask_width)，表示白色区域的高度和宽度。
    """
    mask = binarize_mask(mask)
    white_mask_indices = torch.nonzero(mask == 1, as_tuple=True)
    if len(white_mask_indices[0]) == 0 or len(white_mask_indices[1]) == 0:
        return 0, 0
    else:
        mask_height = white_mask_indices[1].max().item() - white_mask_indices[1].min().item() + 1
        mask_width = white_mask_indices[2].max().item() - white_mask_indices[2].min().item() + 1
        return mask_height, mask_width

def get_white_mask_bounding_box(mask):
    mask = binarize_mask(mask)
    white_mask_indices = torch.nonzero(mask == 1, as_tuple=True)
    """
    获取遮罩处白色区域的边界框。

    参数:
    mask (torch.Tensor): 输入的遮罩张量。

    返回:
    tuple: (x_min, y_min, width, height)，表示白色区域的左上角坐标和宽高。
    """

    if len(white_mask_indices[0]) == 0 or len(white_mask_indices[1]) == 0:
        return 0, 0, 0, 0
    else:
        # 找到白色区域在每个维度上的最小和最大索引
        y_min = white_mask_indices[1].min().item()
        y_max = white_mask_indices[1].max().item()
        x_min = white_mask_indices[2].min().item()
        x_max = white_mask_indices[2].max().item()

        # 计算宽度和高度
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        # 返回的值是一个元组(x, y, w, h)，其中x和y是边界框的左上角坐标，w和h是宽度和高度。
        return x_min, y_min, width, height

class AlignImageswithMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),  # 图像
                "mask_target": ("MASK",),  # 蒙版
                "mask_ref": ("MASK",),  # 蒙版
                "method_mode": (
                ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest'], {"default": "lanczos"}),
                "scaling_mode": (["High_Consistency", "Width_Consistency"], {"default": "Width_Consistency","tooltip": "遮罩缩放参考，高缩放一致，宽缩放一致."}),  # 默认字符串
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,"tooltip": "缩放百分比，1时缩放至相同的大小，0时不缩放."}),  # 浮点数
                "align_mode": (["Y_Align", "X_Align", "All_Align"], {"default": "All_Align","tooltip": "参考遮罩的轴进行对齐，Y轴对齐，X轴对齐，XY轴同时对齐."}),  # 默认字符串
                "center_align": ("BOOLEAN", {"default": False,"tooltip": "次要对齐优先级.参考图像中心进行对齐"}),  # 按钮
                "fill_color": (
                ["Custom", "background", "white", "yellow", "black", "red", "gray", "blue", "green"], {"default": "white"}),  # 默认字符串
                "color_hex": ("STRING", {"multiline": False, "default": "#FFFFFF"}),  # 默认字符串
            },
            "optional": {
                "mask_any": ("MASK",{"tooltip": "不参与参考，直接获得mask_target相同的处理结果"}),  # 蒙版
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "JSON",)
    RETURN_NAMES = ("new_image", "new_mask_target", "new_mask_any", "json",)
    FUNCTION = "align_images_with_mask"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"
    DESCRIPTION = "对比target和ref两张遮罩，将图像裁切缩放至参考图像相似的构图，mask_any获得相同的处理结果。可以使用AlignRestoreJson来恢复原图"

    def dynamic_mirror_extend(self, image_pil, target_size, offset_x, offset_y):
        h, w = image_pil.height, image_pil.width
        target_w, target_h = target_size
        offset_x =  abs(offset_x)
        offset_y =  abs(offset_y)
        # 计算各方向的填充量，确保偏移后仍能覆盖画布
        pad_left = max((target_w - w) + offset_x, 0)
        pad_right = pad_left

        pad_top = max((target_h - h)  + offset_y, 0)
        pad_bottom = pad_top

        # 执行镜像填充
        image_np = np.array(image_pil)
        mirrored = cv2.copyMakeBorder(
            image_np,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )

        return Image.fromarray(mirrored), pad_left, pad_top

    def align_images_with_mask(self, background_image, mask_target, mask_ref, method_mode, scaling_mode, scale_factor,
                               align_mode, center_align, fill_color, color_hex, mask_any=None):
        """
        基于蒙版对齐将图像缩放裁切至相似构图。

        参数:
        - background_image: 背景图像
        - mask_target: 目标蒙版
        - mask_ref: 参考蒙版
        - method_mode: 缩放方法模式
        - scaling_mode: 缩放模式
        - scale_factor: 缩放因子
        - align_mode: 对齐模式
        - center_align: 是否中心对齐
        - fill_color: 填充颜色
        - color_hex: 自定义颜色（当fill_color为"Custom"时使用）
        - mask_any: 可选蒙版（默认为mask_target）

        返回:
        - 对齐后的图像、目标蒙版、可选蒙版以及恢复信息
        """
        # 尺寸一致性校验
        if background_image.shape[1:3] != mask_target.shape[1:3]:
            print("尺寸校验未通过，请尽量保证background_image和mask_target大小一致")
            # 将mask_target缩放至background_image相同大小
            mask_target = mask_scale_as(background_image, mask_target)

        # 初始化可选蒙版
        mask_any = mask_target if mask_any is None else mask_any

        # 张量转PIL
        image_pil = tensor2pil(background_image)  # 将背景图像转换为PIL图像
        mask_ref_pil = tensor2pil(mask_ref).convert('L')  # 将参考蒙版转换为PIL灰度图像
        mask_tgt_pil = tensor2pil(mask_target).convert('L')  # 将目标蒙版转换为PIL灰度图像
        mask_any_pil = tensor2pil(mask_any).convert('L')  # 将可选蒙版转换为PIL灰度图像

        # 获取原始蒙版边界框
        # tuple: (x_min, y_min, width, height)，表示白色区域的最里坐标和宽高
        ref_bbox = get_white_mask_bounding_box(mask_ref)  # 获取参考蒙版的边界框
        tgt_bbox = get_white_mask_bounding_box(mask_target)  # 获取目标蒙版的边界框
        print(f"参考蒙版边界框:{ref_bbox} 目标蒙版边界框:{tgt_bbox}")

        # 动态计算缩放因子
        ref_w, ref_h = ref_bbox[2], ref_bbox[3]  # 参考蒙版的宽度和高度
        tgt_w, tgt_h = tgt_bbox[2], tgt_bbox[3]  # 目标蒙版的宽度和高度
        scale = ref_h / tgt_h if scaling_mode == "High_Consistency" else ref_w / tgt_w  # 根据缩放模式计算缩放因子
        effective_scale = scale_factor * scale + (1 - scale_factor)  # 计算实际缩放系数
        print(f"实际缩放系数:{effective_scale:.2f} (模式:{scaling_mode})")

        # 执行图像缩放
        new_size = (int(image_pil.width * effective_scale), int(image_pil.height * effective_scale))  # 计算新尺寸
        resize_method = getattr(Image, method_mode.upper())  # 获取缩放方法
        image_pil = image_pil.resize(new_size, resize_method)  # 缩放背景图像
        mask_tgt_pil = mask_tgt_pil.resize(new_size, resize_method)  # 缩放目标蒙版
        mask_any_pil = mask_any_pil.resize(new_size, resize_method)  # 缩放可选蒙版
        print(f"缩放完成 新尺寸:{new_size}")

        # 计算对齐偏移量
        new_bbox = get_white_mask_bounding_box(pil2tensor(mask_tgt_pil))  # 获取缩放后目标蒙版的边界框
        a_center = (new_bbox[0] + new_bbox[2] // 2, new_bbox[1] + new_bbox[3] // 2)  # 缩放后目标蒙版的中心点
        b_center = (ref_bbox[0] + ref_bbox[2] // 2, ref_bbox[1] + ref_bbox[3] // 2)  # 参考蒙版的中心点
        offset_x = 0
        offset_y = 0
        if align_mode == "X_Align":
            offset_x = b_center[0] - a_center[0]  # 计算X轴偏移量
        if align_mode == "Y_Align":
            offset_y = b_center[1] - a_center[1]  # 计算Y轴偏移量
        if align_mode == "All_Align":
            offset_x = b_center[0] - a_center[0]  # 计算X轴偏移量
            offset_y = b_center[1] - a_center[1]  # 计算Y轴偏移量
        if center_align:  # 如果启用中心对齐
            offset_x = mask_ref_pil.width // 2 - a_center[0] if "Y" in align_mode else offset_x  # 计算X轴中心对齐偏移量
            offset_y = mask_ref_pil.height // 2 - a_center[1] if "X" in align_mode else offset_y  # 计算Y轴中心对齐偏移量
        print(f"最终偏移量 X:{offset_x} Y:{offset_y} (对齐模式:{align_mode})")

        # 修改后的画布创建逻辑
        if fill_color == "background":
            fill = "background"
            canvas = Image.new("RGB", mask_ref_pil.size)
            # 动态扩展原图尺寸
            extended_image, pad_left, pad_top = self.dynamic_mirror_extend(
                image_pil,
                mask_ref_pil.size,
                offset_x,
                offset_y
            )

            # 计算最终粘贴坐标（核心修复）
            paste_x = offset_x - pad_left
            paste_y = offset_y - pad_top

            # 确保坐标不越界（二次保护）
            # paste_x = max(paste_x, 0)
            # paste_y = max(paste_y, 0)

            canvas.paste(extended_image, (paste_x, paste_y))
        else:
            # 创建新画布
            fill = color_hex if fill_color == "Custom" else fill_color  # 确定填充颜色
            canvas = Image.new("RGB", mask_ref_pil.size, fill)  # 创建新画布
            canvas.paste(image_pil, (offset_x, offset_y)) # 将缩放后的背景图像粘贴到画布上

        final_mask = Image.new("L", mask_ref_pil.size, 0)  # 创建最终目标蒙版画布
        final_any_mask = Image.new("L", mask_ref_pil.size, 0)  # 创建最终可选蒙版画布

        # 合成图像
        # canvas.paste(image_pil, (offset_x, offset_y))  # 将缩放后的背景图像粘贴到画布上
        final_mask.paste(mask_tgt_pil, (offset_x, offset_y))  # 将缩放后的目标蒙版粘贴到画布上
        final_any_mask.paste(mask_any_pil, (offset_x, offset_y))  # 将缩放后的可选蒙版粘贴到画布上
        print("图像合成完成")

        # 构建恢复信息
        restore_info = {
            "offset": (offset_x, offset_y),  # 偏移量
            "original_size": mask_ref_pil.size,  # 原始尺寸
            "scaled_size": new_size,  # 缩放后尺寸
            "bounding_box": new_bbox,  # 缩放后目标蒙版的边界框
            "fill_color": fill,  # 填充颜色
            "effective_scale": effective_scale,
            "method_mode": method_mode
        }

        return (
            pil2tensor(canvas),  # 返回对齐后的图像
            pil2tensor(final_mask),  # 返回目标蒙版
            pil2tensor(final_any_mask),  # 返回可选蒙版
            json.dumps(restore_info)  # 返回恢复信息（JSON格式）
        )

class AlignRestoreJson:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "new_image": ("IMAGE",),
                "json_str": ("JSON",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("new_background_image",)
    FUNCTION = "restore_crop_json"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"

    def restore_crop_json(self, background_image, new_image, json_str):
        # 解析JSON
        restore_info = json.loads(json_str)
        print(f"恢复信息: {restore_info}")

        original_width, original_height = restore_info["original_size"]
        print("original_width, original_height:",original_width, original_height)
        offset_x, offset_y = restore_info["offset"]
        method_mode = restore_info["method_mode"]
        effective_scale = restore_info["effective_scale"]

        # 张量转PIL
        background_pil = tensor2pil(background_image)
        new_image_pil = tensor2pil(new_image)

        new_size = (int(background_pil.width * effective_scale), int(background_pil.height * effective_scale))  # 计算新尺寸
        resize_method = getattr(Image, method_mode.upper())  # 获取缩放方法
        new_background_pil = background_pil.resize(new_size, resize_method)  # 缩放背景图像

        # 创建画布
        canvas = new_background_pil.convert('RGB')
        print(f"创建画布: 尺寸={background_pil.size}")

        # 计算粘贴位置
        paste_x = -offset_x
        paste_y = -offset_y
        print(f"粘贴位置: X={paste_x} Y={paste_y}")

        # 粘贴图像
        canvas.paste(new_image_pil, (paste_x, paste_y))
        print("图像粘贴完成")

        # 缩小回去
        # 计算缩小回去的尺寸（即原始尺寸）
        original_size = (background_pil.width, background_pil.height)

        # 缩小回去
        restored_background_pil = canvas.resize(original_size, resize_method)

        return (pil2tensor(restored_background_pil),)

class GetContentFromExcel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "Excel路径","multiline": False}),  # 不使用box
                "target_name": ("STRING", {"default": "对象名","multiline": False}),  # 不使用box
                "target_category": ( "STRING", {"default": "类别名","multiline": False}),  # 不使用box
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("string", )
    FUNCTION = "get_content_from_excel"
    CATEGORY = "Sanmi Nodes/Basics Nodes"
    def get_content_from_excel(self,file_path, target_name, target_category):
        file_path = file_path.strip('"')
        # 读取 Excel 文件
        df = pd.read_excel(file_path, index_col=0)

        # 检查输入的类别名称是否在列中
        if target_category not in df.columns:
            raise ValueError(f"类别名称 '{target_category}' 不存在于 Excel 文件中。")

        # 检查输入的名称是否在索引中
        if target_name not in df.index:
            raise ValueError(f"名称 '{target_name}' 不存在于 Excel 文件中。")

        # 获取对应的内容
        content = df.at[target_name, target_category]

        return (content,)

class GetFilePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "填写文件夹路径", "multiline": False}),
                "extension": ("STRING", {"default": "填写所需的后缀名，逗号间隔", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("path", "number")
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Basics Nodes"

    def gogo(self, file_path, extension):
        # 去除路径首尾的引号并验证有效性
        cleaned_path = file_path.strip("'\"")
        if not os.path.isdir(cleaned_path):
            return ('', 0)

        # 处理多语言逗号和空格
        normalized_ext = extension.replace('，', ',').replace(' ', '')
        ext_list = [ext.strip().lower() for ext in normalized_ext.split(',') if ext.strip()]

        # 遍历目录收集匹配文件
        result = []
        for root, _, files in os.walk(cleaned_path):
            for filename in files:
                # 获取无点后缀并转小写
                file_ext = os.path.splitext(filename)[1][1:].lower()
                if file_ext in ext_list:
                    result.append(os.path.join(root, filename))

        return ('\n'.join(result), len(result))

class ReduceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),# 蒙版
                "direction": (["up", "down", "left", "right"], {"default": "up"}),  # 不使用box
                "percentage": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01,"tooltip": "减少的比例，0时完全不减少."}), # 浮点数
            },
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "reduce_mask"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Mask"

    def reduce_mask(self, mask, direction, percentage):
        # Make a copy of the mask to avoid modifying the original
        new_mask = mask.clone()

        # 使用新的函数获取白色区域的边界框
        x_min, y_min, width, height = get_white_mask_bounding_box(mask)

        # 根据方向和百分比调整边界
        if direction == 'up':
            new_height = int(height * percentage)
            new_mask[:, y_min:y_min + new_height, :] = mask[:, y_min:y_min + new_height, :] * 0
        elif direction == 'down':
            new_height = int(height * percentage)
            new_mask[:, y_min + height - new_height:y_min + height, :] = mask[:,
                                                                         y_min + height - new_height:y_min + height,
                                                                         :] * 0
        elif direction == 'left':
            new_width = int(width * percentage)
            new_mask[:, :, x_min:x_min + new_width] = mask[:, :, x_min:x_min + new_width] * 0
        elif direction == 'right':
            new_width = int(width * percentage)
            new_mask[:, :, x_min + width - new_width:x_min + width] = mask[:, :,
                                                                      x_min + width - new_width:x_min + width] * 0

        return (new_mask,)

class AdjustHexBrightness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_color": ("STRING",),  # 不使用box
                "brightness": ("FLOAT", {"default": 0.00, "min": -1.00, "max": 1.00, "step": 0.01}), # 浮点数
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hex",)
    FUNCTION = "adjust_brightness"
    CATEGORY = "Sanmi Nodes/Basics Nodes"

    def adjust_brightness(self,hex_color, brightness):
        # 确保输入的颜色是有效的十六进制格式
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]

        # 将十六进制颜色转换为RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # 调整RGB值
        def adjust_channel(channel, brightness):
            return max(0, min(255, int(channel * (1 + brightness))))

        r = adjust_channel(r, brightness)
        g = adjust_channel(g, brightness)
        b = adjust_channel(b, brightness)

        # 将调整后的RGB值转换回十六进制格式
        new_hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return (new_hex_color,)

class SanmiTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nothing_boolean": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "nothing": (AlwaysEqualProxy("*"), {}),
                "time_format": (["%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S"], {"default": "%Y%m%d%H%M%S"}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"), "STRING",)
    RETURN_NAMES = ("nothing", "string",)
    CATEGORY = "Sanmi Nodes/Basics Nodes"
    FUNCTION = "get_current_datetime"

    def get_current_datetime(self, time_format,nothing = None, **kwargs):
        # 获取当前日期和时间
        now = datetime.now()

        # 根据选择的时间格式化类型进行格式化
        if time_format == "%Y%m%d%H%M%S":
            formatted_datetime = now.strftime("%Y%m%d%H%M%S")
        else:
            formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

        return (nothing, formatted_datetime, )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        now = datetime.now()
        return now

class IntToBOOLEAN:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "int": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),# 整数
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/To"
    FUNCTION = "gogo"

    def gogo(self, int):
        # Convert the integer to a boolean
        boolean_value = bool(int)
        return (boolean_value,)

class LoadImagesanmi:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {
            "image": (sorted(files), {"image_upload": True}),
            "filename_text_extension": (["true", "false"],),
            "A_Key": ("STRING", {"multiline": False}),  # 不使用box
            "B_Key": ("STRING", {"multiline": False}),  # 不使用box
        }
        }

    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"
    RETURN_TYPES = ("IMAGE", "STRING","STRING","STRING")
    RETURN_NAMES = ("IMAGE", "filename","A_Value","B_Value",)
    FUNCTION = "load_image"

    def load_image(self, image, filename_text_extension, A_Key, B_Key):
        # 获取图像的路径
        image_path = folder_paths.get_annotated_filepath(image)

        # 使用 pillow 打开图像
        img = node_helpers.pillow(Image.open, image_path)

        # 初始化输出图像和掩码的列表
        output_images = []
        output_masks = []
        w, h = None, None

        # 排除的图像格式
        excluded_formats = ['MPO']

        # 迭代图像序列
        for i in ImageSequence.Iterator(img):
            # 处理图像的 EXIF 信息
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            # 如果图像模式为 'I'，则进行转换
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            # 初始化图像的宽度和高度
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            # 如果图像大小不一致，则跳过
            if image.size[0] != w or image.size[1] != h:
                continue

            # 转换图像为 numpy 数组并归一化
            image = np.array(image).astype(np.float32) / 255.0
            # 转换为 PyTorch 张量并添加维度
            image = torch.from_numpy(image)[None,]
            # 检查图像中是否有 'A' 通道
            if 'A' in i.getbands():
                # 获取 'A' 通道的掩码并归一化
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                # 反转掩码
                mask = 1. - torch.from_numpy(mask)
            else:
                # 如果没有 'A' 通道，则创建一个全零的掩码
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            # 添加图像和掩码到输出列表
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        # 如果有多个图像并且格式不在排除列表中，则合并图像和掩码
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            # 否则只使用第一个图像和掩码
            output_image = output_images[0]
            output_mask = output_masks[0]

        # 提取文件名并根据 filename_text_extension 参数决定是否添加后缀名
        filename = os.path.basename(image_path)
        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]  # 去掉文件扩展名

        # 自动检测并读取三种隐藏数据格式
        A_Value, B_Value = self.extract_metadata(image_path, A_Key, B_Key)
        # 返回输出图像、掩码、文件名和提取的值
        return (output_image, filename, A_Value, B_Value)

    def extract_metadata(self, image_path, A_Key, B_Key):
        """自动检测并读取三种隐藏数据格式"""
        data = None
        file_ext = os.path.splitext(image_path)[1].lower()

        # 优先尝试PNG参数读取
        if file_ext == '.png':
            try:
                with Image.open(image_path) as img:
                    data = img.text.get('parameters', '')
                    if data: print(f"[DEBUG] 从PNG参数读取到数据")
            except:
                pass

        # 如果未找到参数，尝试LSB隐写
        if not data and file_ext == '.png':
            try:
                data = lsb.reveal(image_path)
                if data: print(f"[DEBUG] 从LSB读取到数据")
            except:
                pass

        # 如果是JPEG尝试EXIF读取
        if not data and file_ext in ('.jpg', '.jpeg'):
            try:
                exif_dict = piexif.load(image_path)
                data = exif_dict["Exif"].get(piexif.ExifIFD.UserComment, b'').decode('utf-8')
                if data: print(f"[DEBUG] 从EXIF读取到数据")
            except:
                pass

        # 如果未找到有效数据
        if not data:
            return ("", "")

        # 统一解析逻辑
        try:
            parsed_data = {}
            pairs = data.strip().split(';')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    parsed_data[key.strip()] = value.strip()

            A_Value = parsed_data.get(A_Key, "")
            B_Value = parsed_data.get(B_Key, "")
        except Exception as e:
            print(f"[WARNING] 数据解析失败: {str(e)}")
            A_Value = "ParseError"
            B_Value = "ParseError"

        return (A_Value, B_Value)

class LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image_path": ("STRING", {"multiline": False, "default": "填写单张图片的绝对路径"}),
            "filename_text_extension": (["true", "false"],),
        }
        }

    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "filename_text",)
    FUNCTION = "load_image"

    def load_image(self, image_path, filename_text_extension):
        # 图像处理部分保持不变
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        # 新增文件名处理逻辑
        filename = os.path.basename(image_path)
        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (image_tensor, filename)

class SanmiTextConcatenate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ", "}),  # 分隔符，默认为逗号和空格
                "clean_whitespace": (["true", "false"],),  # 是否清除多余空格
                "newline_mode": (["true", "false"], {"default": "false"}),  # 是否启用换行模式
            },
            "optional": {
                "text_a": ("STRING", ),  # 可选文本输入A
                "text_b": ("STRING", ),  # 可选文本输入B
                "text_c": ("STRING", ),  # 可选文本输入C
                "text_d": ("STRING", ),  # 可选文本输入D
            }
        }

    RETURN_TYPES = ("STRING",)  # 返回类型为字符串
    FUNCTION = "text_concatenate"  # 函数名
    CATEGORY = "Sanmi Nodes/Basics Nodes"  # 类别

    def text_concatenate(self, delimiter, clean_whitespace, newline_mode, **kwargs):
        text_inputs: list[str] = []  # 用于存储有效的文本输入

        # 处理特殊情况，如果分隔符是"\n"（字面上的换行符）。
        if delimiter == "\\n":
            delimiter = "\n"

        # 按键名排序遍历接收到的输入。
        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            # 仅处理字符串输入端口。
            if isinstance(v, str):
                if clean_whitespace == "true":
                    # 去除输入的前后空格。
                    v = v.strip()

                # 仅在输入为非空字符串时使用，因为连接完全空的输入没有意义。
                # 注意：如果禁用空格清理，包含100%空格的输入将被视为非空输入。
                if v != "":
                    text_inputs.append(v)

        # 检查是否启用换行模式
        if newline_mode == "true":
            merged_text = "\n".join(text_inputs)  # 使用换行符连接文本
        else:
            # 合并输入。即使为空也会生成输出。
            merged_text = delimiter.join(text_inputs)  # 使用指定的分隔符连接文本

        return (merged_text,)  # 返回合并后的文本

class SanmiCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cmp": (['a = b', 'a <> b', 'a > b', 'a < b', 'a >= b', 'a <= b'],),
                "a": (AlwaysEqualProxy("*"), {}),
                "b": (AlwaysEqualProxy("*"), {}),
                "ture": ("FLOAT", {"default": [1.0],}),
                "false": ("FLOAT", {"default": [1.0],}),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Logic"
    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float",)
    DESCRIPTION = "比较a和b的值，如果是真，则返回ture的值，否则返回false的值。a和b可能是字符串、整数、浮点数，但必须格式一致。"

    def doit(self, cmp, a, b,ture,false):
    # 比较a和b的值，如果是真，则返回ture的值，否则返回false的值。a和b可能是字符串、整数、浮点数，但必须格式一致。
        if cmp == 'a = b':
            result = a == b
        elif cmp == 'a <> b':
            result = a != b
        elif cmp == 'a > b':
            result = a > b
        elif cmp == 'a < b':
            result = a < b
        elif cmp == 'a >= b':
            result = a >= b
        elif cmp == 'a <= b':
            result = a <= b
        else:
            raise ValueError("Invalid comparison operator")

        return (ture if result else false,)

class SanmiCompareV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cmp": (['a = b', 'a <> b', 'a > b', 'a < b', 'a >= b', 'a <= b'],),
                "a": (AlwaysEqualProxy("*"), {}),
                "b": (AlwaysEqualProxy("*"), {}),
                "ture": (AlwaysEqualProxy("*"), {}),
                "false": (AlwaysEqualProxy("*"), {}),
            },
        }

    FUNCTION = "doit"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Logic"
    RETURN_TYPES = (AlwaysEqualProxy("*"), )
    RETURN_NAMES = ("any",)
    DESCRIPTION = "比较a和b的值，如果是真，则返回ture的值，否则返回false的值。a和b可能是字符串、整数、浮点数，但必须格式一致。"

    def doit(self, cmp, a, b,ture,false):
    # 比较a和b的值，如果是真，则返回ture的值，否则返回false的值。a和b可能是字符串、整数、浮点数，但必须格式一致。
        if cmp == 'a = b':
            result = a == b
        elif cmp == 'a <> b':
            result = a != b
        elif cmp == 'a > b':
            result = a > b
        elif cmp == 'a < b':
            result = a < b
        elif cmp == 'a >= b':
            result = a >= b
        elif cmp == 'a <= b':
            result = a <= b
        else:
            raise ValueError("Invalid comparison operator")

        return (ture if result else false,)

class SanmiImageRotate:
    def __init__(self):
        pass  # 初始化方法

    @classmethod
    def INPUT_TYPES(cls):
        # 定义输入参数类型和默认值
        return {
            "required": {
                "images": ("IMAGE",),  # 输入图像列表
                "rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 90}),  # 旋转角度，默认为0，范围0到360，步长90
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK","INT",)  # 返回类型为图像
    RETURN_NAMES = ("images","mask","rotation")  # 返回的名称
    FUNCTION = "image_rotate"  # 功能名称
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image" # 类别名称

    def image_rotate(self, images, rotation, mask=None):
        batch_tensor = []
        rotated_mask = None

        if mask is not None:
            mask = tensor2pil(mask)  # Convert the mask tensor to a PIL Image
            mask = mask.convert('L')  # Convert the mask to grayscale

        for image in images:
            # Convert tensor to PIL Image
            image = tensor2pil(image)

            # Normalize rotation angle
            if rotation > 360 or rotation < -360:
                rotation = rotation % 360
            if rotation % 90 != 0:
                rotation = int((rotation // 90) * 90)

            # Calculate the number of 90-degree rotations
            rot = int(rotation / 90)

            # Rotate the image
            if rot > 0:
                for _ in range(rot):
                    image = image.transpose(Image.ROTATE_270)  # Clockwise 90 degrees
                    if mask is not None:
                        mask = mask.transpose(Image.ROTATE_270)
            elif rot < 0:
                for _ in range(-rot):
                    image = image.transpose(Image.ROTATE_90)  # Counter-clockwise 90 degrees
                    if mask is not None:
                        mask = mask.transpose(Image.ROTATE_90)

            batch_tensor.append(pil2tensor(image))

        batch_tensor = torch.cat(batch_tensor, dim=0)

        if mask is not None:
            rotated_mask = pil2tensor(mask)

        return (batch_tensor, rotated_mask, rotation)

class CreateTxtForImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": './database'}),
                "prompt": ("STRING", {"multiline": True}),
                "Overwrite": ("BOOLEAN", {"default": False}),  # 按钮
            },
            # 可选项
            "optional": {
                "nothing": (AlwaysEqualProxy("*"), {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"

    def gogo(self, folder_path, prompt, Overwrite=False, **kwargs):
        text_content = prompt  # 将提示文本存储在变量中

        if os.path.isfile(folder_path):
            # 如果路径是一个单独的图像文件
            if folder_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                self.process_image(folder_path, Overwrite, text_content)  # 处理单个图像文件
            else:
                print(f"Not a valid image file: {folder_path}")  # 打印无效图像文件的信息
        elif os.path.isdir(folder_path):
            # 如果路径是一个目录
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                        image_path = os.path.join(root, file)  # 获取完整的图像文件路径
                        self.process_image(image_path, Overwrite, text_content)  # 处理目录中的每个图像文件
        else:
            print(f"Invalid path: {folder_path}")  # 打印无效路径的信息

        return ("创建文本完成",)  # 返回完成信息

    def process_image(self, image_path, Overwrite, text_content):
        # 生成与图像文件同名的文本文件路径
        txt_file_path = os.path.splitext(image_path)[0] + '.txt'
        # 如果文本文件不存在或允许覆盖，则创建或覆盖文本文件
        if not os.path.exists(txt_file_path) or Overwrite:
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text_content)  # 写入提示文本内容
                print(f"Created or Overwritten: {txt_file_path}")  # 打印创建或覆盖的文件路径
        else:
            print(f"Skipped (already exists): {txt_file_path}")  # 打印跳过的信息

class AddTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img_path": ("STRING", {"multiline": False}),  # 图片路径（支持目录和文件）
                "A_Key": ("STRING", {"multiline": False, "tooltip": "写一个健名，例如parameters."}),  # 第一个键名
                "A_Value": ("STRING", {"multiline": False,
                                       "tooltip": "如果相同路径下存在同名的txt文本，会自动将其替换A_Value的值，便于填写目录时批量处理图像。"}),
                # 第一个键值
                "B_Key": ("STRING", {"multiline": False}),  # 第二个键名
                "B_Value": ("STRING", {"multiline": False}),  # 第二个键值
                "method": (["lsb", "piexif", "parameters"], {"default": "parameters"}),  # 写入方法选择
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"
    DESCRIPTION = "支持处理目录和单个文件, 用于将文本写入图像中，当相同路径下存在同名的txt文本，会自动将其替换为A_Value的值。选择lsb或parameters会自动转换为PNG格式，选择piexif会自动转换为JPEG格式。lsb模式不支持中文和非ASCII字符。"

    def gogo(self, img_path, A_Key, A_Value, B_Key, B_Value, method):
        img_path = img_path.strip('"')
        """主处理函数，支持处理目录和单个文件"""
        if os.path.isdir(img_path):
            # 遍历目录处理所有图片文件
            for filename in os.listdir(img_path):
                file_path = os.path.join(img_path, filename)
                if self.is_image_file(file_path):
                    self.process_entry(file_path, A_Key, A_Value, B_Key, B_Value, method)
        else:
            # 处理单个文件
            self.process_entry(img_path, A_Key, A_Value, B_Key, B_Value, method)
        return ("Processing complete.",)

    def process_entry(self, file_path, A_Key, A_Value, B_Key, B_Value, method):
        """处理单个文件入口"""
        # 自动读取同目录txt文件内容（如果存在）
        txt_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                A_Value = f.read().strip()

        # 执行元数据处理
        self.process_metadata(file_path, A_Key, A_Value, B_Key, B_Value, method)

    def process_metadata(self, img_path, A_Key, A_Value, B_Key, B_Value, method):
        """元数据处理核心逻辑"""
        # 格式转换预处理
        img_path = self.convert_format(img_path, method)

        if method == "lsb":
            # 文本规范化处理
            A_Value = self.normalize_text(A_Value)
            B_Value = self.normalize_text(B_Value)

        # 生成组合文本
        combined_text = f"{A_Key}:{A_Value};{B_Key}:{B_Value}"

        try:
            # 根据选择的方法写入数据
            if method == "lsb":
                self.write_lsb(img_path, combined_text)
            elif method == "piexif":
                self.write_exif(img_path, combined_text)
            elif method == "parameters":
                self.write_png_parameters(img_path, combined_text)
            print(f"[Success] {os.path.basename(img_path)} 数据写入成功")
        except Exception as e:
            print(f"[Error] {os.path.basename(img_path)} 处理失败: {str(e)}")

    def convert_format(self, img_path, method):
        """格式转换处理（根据方法自动转换）"""
        # piexif方法需要JPEG格式，自动转换
        if method == "piexif":
            if not img_path.lower().endswith(('.jpg', '.jpeg')):
                # 如果当前不是JPEG格式，则转换为JPEG
                with Image.open(img_path) as img:
                    jpeg_path = os.path.splitext(img_path)[0] + ".jpg"
                    img.convert("RGB").save(jpeg_path, "JPEG", quality=95)
                    if img_path != jpeg_path:  # 避免删除新文件
                        os.remove(img_path)
                return jpeg_path
            return img_path  # 已经是JPEG格式，直接返回

        # 其他方法统一转换为PNG
        if not img_path.lower().endswith('.png'):
            with Image.open(img_path) as img:
                png_path = os.path.splitext(img_path)[0] + ".png"
                img.save(png_path, "PNG")
            if img_path != png_path:  # 避免删除新文件
                os.remove(img_path)
            return png_path
        return img_path  # 已经是PNG格式，直接返回

    def normalize_text(self, text):
        """文本规范化处理（替换符号+去除非ASCII字符）"""
        symbol_map = {
            '，': ',', '。': '.', '！': '!', '？': '?', '：': ':', '；': ';',
            '（': '(', '）': ')', '【': '[', '】': ']', '“': '"', '”': '"',
            '‘': "'", '’': "'", '《': '<', '》': '>', '、': ',', '——': '-',
            '…': '...',
        }
        for zh, en in symbol_map.items():
            text = text.replace(zh, en)
        return re.sub(r'[\u4e00-\u9fff]', '', text)

    def write_lsb(self, img_path, text):
        """LSB隐写写入"""
        lsb.hide(img_path, text).save(img_path)

    def write_exif(self, img_path, text):
        """EXIF元数据写入"""
        exif_dict = piexif.load(img_path)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = text.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, img_path)

    def write_png_parameters(self, img_path, text):
        """PNG参数写入"""
        img = Image.open(img_path)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text('parameters', text)
        img.save(img_path, "PNG", pnginfo=png_info)

    def is_image_file(self, filename):
        """判断是否为支持的图片格式"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

class GetMostCommonColor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "image": ("IMAGE",),  # 图像
            },
            "optional": {
                "mask": ("MASK",),    # 蒙版
            },
        }

    RETURN_TYPES = ("STRING",)  # 返回类型为字符串
    RETURN_NAMES = ("string",)  # 返回值名称为字符串
    FUNCTION = "get_most_common_color"  # 定义函数名称
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"
    DESCRIPTION = "识别图像（或遮罩区域）中的主色，返回HEX值"

    def get_most_common_color(self, image, mask=None):
        batch_size, height, width, channels = image.shape
        most_frequent_colors = []

        for b in range(batch_size):
            img = image[b]  # [H, W, C]

            # 处理蒙版逻辑
            if mask is not None:
                msk = mask[b]  # [H, W]
                # 获取蒙版区域的像素
                indices = torch.nonzero(msk, as_tuple=True)
                masked_colors = img[indices]
            else:
                # 当没有蒙版时处理整个图像
                masked_colors = img.reshape(-1, channels)  # [H*W, C]

            # 处理空蒙版情况（当有蒙版但全为False时）
            if masked_colors.nelement() == 0:
                most_frequent_colors.append("#000000")
                continue

            # 转换颜色到0-255整数
            if masked_colors.is_floating_point():
                masked_colors = (masked_colors * 255).to(torch.int32)

            # 统计颜色频率
            color_counts = Counter(
                tuple(color.tolist())
                for color in masked_colors
            )

            # 获取最常用颜色并转换HEX
            if color_counts:
                most_common = color_counts.most_common(1)[0][0]
                hex_color = "#{:02x}{:02x}{:02x}".format(*most_common)
            else:
                hex_color = "#000000"

            most_frequent_colors.append(hex_color)

        # 返回所有批次的颜色（当batch_size=1时自动解包）
        return (most_frequent_colors[0] if batch_size == 1 else most_frequent_colors,)

class MaskToBboxes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "mask": ("MASK",),# 蒙版
            },
            "optional": {
                "bboxe": ("BBOX",),
            },
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "find_bboxes"
    CATEGORY = "Sanmi Nodes/Basics Nodes/To"
    DESCRIPTION = "用于搭配Sam2Segmentation节点使用,允许你手动绘制遮罩识别成框"

    def find_bboxes(self,mask: torch.Tensor, bboxe=None):
        batch_size, height, width = mask.shape
        bboxes = []

        for b in range(batch_size):
            # 获取当前batch的mask
            current_mask = mask[b].numpy()  # 转换为numpy数组以便处理

            # 使用scipy.ndimage.label来找到mask中的所有独立连通区域
            labeled_mask, num_features = scipy.ndimage.label(current_mask)
            print("num_features", num_features)

            for label_idx in range(1, num_features + 1):
                # 找到当前label的区域
                region = (labeled_mask == label_idx)

                # 找到区域的坐标
                coords = np.argwhere(region)
                if coords.size == 0:
                    continue

                # 获取左上角和右下角的坐标
                y1, x1 = coords.min(axis=0)
                y2, x2 = coords.max(axis=0)

                # 添加到bboxes列表
                bboxes.append([x1, y1, x2, y2])

        # 如果提供了额外的bboxe，添加它们
        if bboxe is not None:
            bboxes.extend(bboxe)
        return (bboxes,)

class StringToBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "string": ("STRING", {"multiline": False}),# 不使用box
            },
        }

    RETURN_TYPES = ("BOX",)
    RETURN_NAMES = ("box",)
    FUNCTION = "do"
    CATEGORY = "Sanmi Nodes/Basics Nodes/To"
    DESCRIPTION = "将字符串转换为BOX格式"

    def do(self,string):
        # Attempt to convert the string representation of a list into an actual list
        box = ast.literal_eval(string)
        if isinstance(box, list):
            return (box,)
        return (None,)

class SimpleWildcards:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
        }

    CATEGORY = "Sanmi Nodes/Basics Nodes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "doit"

    # 查找输入的通配符文本，并返回文本列表。
    def find_matching_files(self, wildcard_text):
        wildcard_names = re.findall(r"__(.*?)__", wildcard_text)
        matching_files = []
        # 获取当前脚本文件的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取wildcards文件夹的路径
        wildcards_dir = os.path.join(script_dir, "..", "wildcards")
        for wildcard_name in wildcard_names:
            file_path = os.path.join(wildcards_dir, f"{wildcard_name}.txt")
            if os.path.exists(file_path):
                matching_files.append(file_path)
            else:
                matching_files.append(f"未查到该路径: {file_path}")
        return matching_files

    # 查找通配符为对应的文本，并返回随机一行。
    def replace_wildcards(self, wildcard_text):
        matching_files = self.find_matching_files(wildcard_text)
        for file_path in matching_files:
            if file_path.startswith("未查到该路径:"):
                continue
            file_content = self.get_file_content(file_path)
            wildcard_name = os.path.basename(file_path).split(".")[0]
            wildcard_lines = file_content.splitlines()
            wildcard_text = re.sub(re.escape(f"__{wildcard_name}__"), random.choice(wildcard_lines), wildcard_text)
        return wildcard_text

    # 读取通配符文本内容，并返回字符串。
    def get_file_content(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content.strip()

    def doit(self, wildcard_text):
        matching_files = self.find_matching_files(wildcard_text)
        for file_path in matching_files:
            print("SimpleWildcards查找到的文件路径：", file_path)
        prompt = self.replace_wildcards(wildcard_text)
        # 去除属性部分，只返回正文部分
        return (prompt.strip(),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        Nu = SimpleWildcards.doit(kwargs['wildcard_text'])
        prompt = Nu.replace_wildcards()
        return (prompt,)

class StrToPinYin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "str": ("STRING", {"multiline": False}),  # 不使用box
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Basics Nodes/To"

    def gogo(self, str, **kwargs):
        result = []
        # 获取每个汉字的完整拼音
        pinyin_list = lazy_pinyin(str)

        for char, pinyin_char in zip(str, pinyin_list):
            if 'a' <= char.lower() <= 'z':
                # 英文字符保持不变
                result.append(char)
            else:
                # 中文字符转换为拼音，首字母大写
                capitalized_pinyin = pinyin_char.capitalize()
                result.append(capitalized_pinyin)

        # 将结果转换为字符串并返回元组
        return (''.join(result),)

class GetLastPathComponent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PATH": ("STRING", {"multiline": False}),  # 不使用box
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Path"
    def gogo(self, PATH, **kwargs):
        import os
        # 获取路径的最后一个部分
        last_component = os.path.basename(PATH)
        # 返回一个包含该字符串的元组
        return (last_component,)

class MaskWhiteRatioAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "calculate_ratio"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Mask"

    def calculate_ratio(self, mask,):
        mask = binarize_mask(mask)
        ratios = mask.mean(dim=(1, 2))
        # 使用 PyTorch 操作将小数点保留到3位
        ratios_rounded = torch.round(ratios * 1000) / 1000
        return (ratios_rounded,)

class ImageBatchSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "image0", "image1", "image2", "image3")
    FUNCTION = "gogo"
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"
    DESCRIPTION = "将批次图像张量拆分为单个图像输出，不足时回退到第一个图像"

    def gogo(self, image_list):

        batch_size = image_list.shape[0]

        # 自动填充逻辑
        default_image = image_list[0:1]  # 保持四维形状

        image0 = image_list[0:1] if batch_size >= 1 else default_image
        image1 = image_list[1:2] if batch_size >= 2 else default_image
        image2 = image_list[2:3] if batch_size >= 3 else default_image
        image3 = image_list[3:4] if batch_size >= 4 else default_image

        return (
            image_list,  # 原始输入
            image0,  # 第1张图像 (四维)
            image1,  # 第2张图像 (四维)
            image2,  # 第3张图像 (四维)
            image3  # 第4张图像 (四维)
        )

class SortTheMasksLeftRight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "num_masks": ("INT", {"default": 1, "min": 0, "step": 1}),
                "return_choice": (
                    ["black", "all"],
                    {"default": "black", "tooltip": "找不到目标遮罩时返回黑色或原图"}
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/Mask"
    FUNCTION = "process_masks"
    DESCRIPTION = "按水平位置从左到右排序并输出指定排名的遮罩（保留原图强度）"


    def process_masks(self, mask: torch.Tensor, num_masks: int, return_choice: str):
        # 保留原始遮罩的强度信息
        original_mask = mask.clone()

        # 转换为CPU上的布尔数组处理
        mask_np = mask.bool().cpu().numpy()

        # 寻找连通区域
        labeled, regions = scipy.ndimage.label(mask_np)
        if regions == 0:
            return (original_mask,)

        # 使用自定义函数计算边界框中心点
        x_centers = []
        for label in range(1, regions + 1):
            # 提取单个区域的mask
            region_mask = (labeled == label)
            # 转换为PyTorch张量
            region_tensor = torch.from_numpy(region_mask).to(device=mask.device)
            # 获取边界框坐标
            x_min, y_min, width, height = get_white_mask_bounding_box(region_tensor)
            # 计算中心点x坐标
            if width > 0:  # 有效区域
                x_center = x_min + (width - 1) / 2
                x_centers.append(x_center)

        # 处理无有效区域的情况
        if not x_centers:
            return (original_mask,)

        # 按中心点x坐标排序
        sorted_idx = np.argsort(x_centers)

        # 处理无效序号
        if not (0 < num_masks <= regions):
            if return_choice == "black":
                return (torch.zeros_like(original_mask),)
            return (original_mask,)

        # 生成目标遮罩的布尔蒙版
        target_label = sorted_idx[num_masks - 1] + 1  # 标签从1开始
        mask_area = (labeled == target_label)

        # 将布尔蒙版转换为与原遮罩相同设备和类型的强度遮罩
        mask_tensor = torch.from_numpy(mask_area).to(
            device=original_mask.device,
            dtype=original_mask.dtype
        )

        # 应用蒙版到原始遮罩（保留原强度）
        result = original_mask * mask_tensor

        return (result,)

def fit_resize_image(image:Image, target_width:int, target_height:int, resize_sampler:str) -> Image:
    image = image.convert('RGB')
    ret_image = image.resize((target_width, target_height), resize_sampler)
    return  ret_image

# 将image缩放至scale_as相同大小
def image_scale_as(scale_as, image):
    # 提取目标尺寸（兼容单图或批量输入）
    _asimage = tensor2pil(scale_as[0] if scale_as.shape[0] > 0 else scale_as)
    target_size = _asimage.size  # (width, height)

    # 处理输入图像
    input_img = tensor2pil(image[0]).convert('RGB')

    # 执行缩放（固定fill模式 + lanczos采样）
    resized = fit_resize_image(input_img, *target_size, Image.LANCZOS)

    return pil2tensor(resized)

def mask_scale_as(scale_as, mask):
    # 提取目标尺寸（兼容单图或批量输入）
    _asimage = tensor2pil(scale_as[0] if scale_as.shape[0] > 0 else scale_as)
    target_size = _asimage.size  # (width, height)

    # 处理输入掩码（转换为单通道灰度图）
    input_mask = tensor2pil(mask[0]).convert('L')  # 使用'L'模式处理单通道

    # 执行缩放（使用最近邻插值保持掩码的离散值）
    resized = fit_resize_image(input_mask, *target_size, Image.NEAREST)

    # 返回tensor格式，并确保正确的维度
    return pil2tensor(resized)

class BlendICLight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ic_light_image": ("IMAGE",),
                "original_image": ("IMAGE", ),
                "blend_factor": ("FLOAT", {"default": 0.75, "min": 0.00, "max": 1.00, "step": 0.01}),
                "blend_mode": (["normal","brighter","softer"],{"default": "softer"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "Sanmi Nodes/Basics Nodes/Image"
    FUNCTION = "gogo"
    DESCRIPTION = "提取IC_Light光效混合于原图中,系数为0的时候输出原图"

    def gogo(self, ic_light_image, original_image, blend_factor, blend_mode):
        # 将ic_light_images缩放至original_image相同大小
        ic_light_image = image_scale_as(original_image,ic_light_image)

        # 将ic_light_image与original_image混合
        blend_image = self.blend_images(original_image,ic_light_image,blend_factor,blend_mode)

        # 恢复原始颜色和风格
        blend_image = self.batch_normalize(blend_image,original_image,0.5)

        # 阴影混合
        blend_image = self.blending_darken(blend_image,original_image,0.5)

        # 恢复原始颜色和风格
        blend_image = self.batch_normalize(blend_image,original_image,0.5)

        # 混合色相和饱和度
        blend_image = self.blending_Color(blend_image,original_image,0.5)

        # 还原细节
        blend_image = self.detail_transfer(blend_image,original_image)

        return (blend_image,)


    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        image2 = image2.to(image1.device)
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)

        blended_image = self.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return blended_image

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "brighter":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "softer":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

    def batch_normalize(self, images, reference, factor):
        t = copy.deepcopy(images)  # [B x H x W x C]

        t = t.movedim(-1, 0)  # [C x B x H x W]
        for c in range(t.size(0)):
            for i in range(t.size(1)):
                r_sd, r_mean = torch.std_mean(reference[i, :, :, c], dim=None)  # index by original dim order
                i_sd, i_mean = torch.std_mean(t[c, i], dim=None)

                t[c, i] = ((t[c, i] - i_mean) / i_sd) * r_sd + r_mean

        t = torch.lerp(images, t.movedim(0, -1), factor)  # [B x H x W x C]
        return t
    def blending_darken(self, image_a, image_b, blend_percentage=1.0):

        # Convert images to PIL
        img_a = tensor2pil(image_a)
        img_b = tensor2pil(image_b)

        out_image = pilgram.css.blending.darken(img_a, img_b)
        out_image = out_image.convert("RGB")

        # Blend image
        blend_mask = Image.new(mode="L", size=img_a.size,
                               color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        out_image = Image.composite(img_a, out_image, blend_mask)

        return pil2tensor(out_image)

    def blending_Color(self, image_a, image_b, blend_percentage=1.0):

        # Convert images to PIL
        img_a = tensor2pil(image_a)
        img_b = tensor2pil(image_b)

        out_image = pilgram.css.blending.color(img_a, img_b)
        out_image = out_image.convert("RGB")

        # Blend image
        blend_mask = Image.new(mode="L", size=img_a.size,
                               color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        out_image = Image.composite(img_a, out_image, blend_mask)

        return pil2tensor(out_image)

    def adjust_mask(self, mask, target_tensor):
        # Add a channel dimension and repeat to match the channel number of the target tensor
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)  # Add a channel dimension
            target_channels = target_tensor.shape[1]
            mask = mask.expand(-1, target_channels, -1,
                               -1)  # Expand the channel dimension to match the target tensor's channels

        return mask

    def detail_transfer(self, target, source):
        B, H, W, C = target.shape
        device = model_management.get_torch_device()
        target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
        source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

        if target.shape[1:] != source.shape[1:]:
            source_tensor = comfy.utils.common_upscale(source_tensor, W, H, "bilinear", "disabled")

        if source.shape[0] < B:
            source = source[0].unsqueeze(0).repeat(B, 1, 1, 1)

        kernel_size = int(6 * int(1) + 1)

        gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size),
                                                sigma=(1, 1))

        blurred_target = gaussian_blur(target_tensor)
        blurred_source = gaussian_blur(source_tensor)

        # 默认add
        tensor_out = (source_tensor - blurred_source) + blurred_target
        tensor_out = torch.lerp(target_tensor, tensor_out, 1)
        tensor_out = torch.clamp(tensor_out, 0, 1)
        tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()

        return tensor_out

NODE_CLASS_MAPPINGS = {
    "sanmi AddTextToImage": AddTextToImage,
    "sanmi AdjustHexBrightness": AdjustHexBrightness,
    "sanmi Adjust Transparency By Mask": AdjustTransparencyByMask,
    "sanmi Align Images with Mask": AlignImageswithMask,
    "sanmi RestoreJson": AlignRestoreJson,

    "sanmi BinarizeMask": BinarizeMask,
    "sanmi BlendICLight": BlendICLight,

    "sanmi Chinese To Character": ChineseToCharacter,
    "sanmi CreateTxtForImages": CreateTxtForImages,

    "sanmi Float": Float,
    "sanmi Int90": Int90,

    "sanmi Get Content From Excel": GetContentFromExcel,
    "sanmi Get LastPathComponent": GetLastPathComponent,
    "sanmi Get Mask White Region Size": GetWhiteRegionSize,
    "sanmi GetMostCommonColor": GetMostCommonColor,
    "sanmi GetFilePath": GetFilePath,

    "sanmi ImageBatchSplitter": ImageBatchSplitter,

    "sanmi Image_Rotate": SanmiImageRotate,
    "sanmi IntToBOOLEAN": IntToBOOLEAN,

    "sanmi MaskToBboxes": MaskToBboxes,
    "sanmi MaskWhiteRatioAnalyzer": MaskWhiteRatioAnalyzer,

    "sanmi Path Captioner": PathCaptioner,
    "sanmi Path Change": PathChange,

    "sanmi Read Image Prompt": ReadImagePrompt,
    "sanmi Reduce Mask": ReduceMask,

    "sanmi Compare": SanmiCompare,
    "sanmi CompareV2": SanmiCompareV2,
    "sanmi LoadImagesanmi": LoadImagesanmi,
    "sanmi LoadImageFromPath": LoadImageFromPath,
    "sanmi Mask To Box": MaskToBox,
    "sanmi Sanmi_Text_Concatenate": SanmiTextConcatenate,
    "sanmi Time": SanmiTime,

    "sanmi Save Image To Local": SaveImageToLocal,
    "sanmi SimpleWildcards": SimpleWildcards,
    "sanmi SortTheMasksSize": SortTheMasksSize,
    "sanmi SortTheMasksLeftRight":SortTheMasksLeftRight,
    "sanmi StrToPinYin": StrToPinYin,
    "sanmi StringToBox": StringToBox,

    "sanmi Upscale And Keep Original Size": Upscale_And_Keep_Original_Size,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "sanmi AddTextToImage": "Add Text To Image",
    "sanmi AdjustHexBrightness": "Adjust Hex Brightness",
    "sanmi Adjust Transparency By Mask": "Adjust Transparency By Mask",
    "sanmi Align Images with Mask": "Align Images with Mask",
    "sanmi RestoreJson": "Align Restore Json",

    "sanmi BinarizeMask": "Binarize Mask",
    "sanmi BlendICLight": "Blend ICLight",

    "sanmi Chinese To Character": "Chinese To Character",
    "sanmi CreateTxtForImages": "Create Txt For Images",

    "sanmi ImageBatchSplitter": "Image Batch Splitter",

    "sanmi Float": "Float",
    "sanmi Int90": "Int 90",

    "sanmi Get Content From Excel": "Get Content From Excel",
    "sanmi Get LastPathComponent": "Get Last Path Component",
    "sanmi Get Mask White Region Size": "Get Mask White Region Size",
    "sanmi GetMostCommonColor": "Get Most Common Color",
    "sanmi GetFilePath": "Get File Path",

    "sanmi Image_Rotate": "Image Rotate",
    "sanmi IntToBOOLEAN": "Int To BOOLEAN",

    "sanmi MaskToBboxes": "Mask To Bboxes",
    "sanmi MaskWhiteRatioAnalyzer": "Mask White Ratio Analyzer",

    "sanmi Path Captioner": "Path Image Captioner",
    "sanmi Path Change": "Path Change",

    "sanmi Read Image Prompt": "Read Image Prompt",
    "sanmi Reduce Mask": "Reduce Mask",

    "sanmi Compare": "Sanmi Compare",
    "sanmi CompareV2": "Sanmi Compare V2",
    "sanmi LoadImagesanmi": "Sanmi Load Image",
    "sanmi LoadImageFromPath": "Load Image From Path",
    "sanmi Mask To Box": "Sanmi Mask To Box",
    "sanmi Sanmi_Text_Concatenate": "Sanmi Text Concatenate",
    "sanmi Time": "Sanmi Time",

    "sanmi Save Image To Local": "Save Image To Local",
    "sanmi SimpleWildcards": "Simple Wildcards",
    "sanmi SortTheMasksSize": "Sort The Masks Size",
    "sanmi SortTheMasksLeftRight": "Sort The Masks Left Right",
    "sanmi StrToPinYin": "Str To PinYin",
    "sanmi StringToBox": "String To Box",

    "sanmi Upscale And Keep Original Size": "Upscale And Keep Original Size",

}