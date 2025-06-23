import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def calculate_center_position(image, text, font, position):
    """
    计算文本的居中位置
    
    Args:
        image: PIL Image对象
        text (str): 要渲染的文本
        font: PIL ImageFont对象
        position: 位置参数，可以是'tuple'或'center'
    
    Returns:
        tuple: 计算后的位置坐标 (x, y)
    """
    if position == 'center':
        # 获取图片尺寸
        img_width, img_height = image.size
        
        # 获取文本边界框
        bbox = ImageDraw.Draw(image).textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算居中位置
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
        
        return (x, y)
    else:
        return position

def render_text_on_image(image_path, text, output_path=None, font_size=30, font_color=(255, 255, 255), 
                        position=(50, 50), background_color=None, font_path=None):
    """
    在图片上渲染文本
    
    Args:
        image_path (str): 输入图片路径
        text (str): 要渲染的文本
        output_path (str): 输出图片路径，如果为None则覆盖原图
        font_size (int): 字体大小
        font_color (tuple): 字体颜色，RGB格式
        position (tuple or str): 文本位置 (x, y) 或 'center' 表示居中
        background_color (tuple): 背景颜色，如果为None则无背景
        font_path (str): 字体文件路径，如果为None则使用默认字体
    """
    
    # 打开图片
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    
    # 转换为RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 设置字体
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 尝试使用系统字体
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                try:
                    # 尝试其他常见字体路径
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                except:
                    # 如果都失败，使用默认字体
                    font = ImageFont.load_default()
                    print("警告: 无法加载系统字体，将使用默认字体")
    except Exception as e:
        font = ImageFont.load_default()
        print(f"字体加载失败: {e}")
    
    # 计算实际渲染位置（支持居中）
    actual_position = calculate_center_position(image, text, font, position)
    
    # 获取文本边界框
    bbox = draw.textbbox(actual_position, text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 如果需要背景
    if background_color:
        # 绘制背景矩形
        background_bbox = (
            actual_position[0] - 10,
            actual_position[1] - 5,
            actual_position[0] + text_width + 10,
            actual_position[1] + text_height + 5
        )
        draw.rectangle(background_bbox, fill=background_color)
    
    # 绘制文本
    draw.text(actual_position, text, font=font, fill=font_color)
    
    # 保存图片
    if output_path:
        image.save(output_path)
    else:
        # 如果没有指定输出路径，覆盖原图
        if isinstance(image_path, str):
            image.save(image_path)
    
    return image

def render_text_with_multiline(image_path, text, output_path=None, font_size=30, font_color=(255, 255, 255),
                              position=(50, 50), max_width=None, line_spacing=5, background_color=None, font_path=None):
    """
    在图片上渲染多行文本，支持自动换行
    
    Args:
        image_path (str): 输入图片路径
        text (str): 要渲染的文本
        output_path (str): 输出图片路径
        font_size (int): 字体大小
        font_color (tuple): 字体颜色
        position (tuple or str): 起始位置 (x, y) 或 'center' 表示居中
        max_width (int): 最大行宽，超过则换行
        line_spacing (int): 行间距
        background_color (tuple): 背景颜色
        font_path (str): 字体文件路径
    """
    
    # 打开图片
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    
    # 转换为RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 设置字体
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 尝试使用系统字体
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                try:
                    # 尝试其他常见字体路径
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                except:
                    # 如果都失败，使用默认字体
                    font = ImageFont.load_default()
                    print("警告: 无法加载系统字体，将使用默认字体")
    except Exception as e:
        font = ImageFont.load_default()
        print(f"字体加载失败: {e}")
    
    # 分割文本为多行
    if max_width:
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
    else:
        lines = [text]
    
    # 计算总高度
    total_height = len(lines) * font_size + (len(lines) - 1) * line_spacing
    
    # 计算实际渲染位置（支持居中）
    if position == 'center':
        # 获取图片尺寸
        img_width, img_height = image.size
        
        # 计算多行文本的总宽度（取最长的一行）
        max_line_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            max_line_width = max(max_line_width, line_width)
        
        # 计算居中位置
        x = (img_width - max_line_width) // 2
        y = (img_height - total_height) // 2
        actual_position = (x, y)
    else:
        actual_position = position
    
    # 绘制背景（如果需要）
    if background_color:
        background_bbox = (
            actual_position[0] - 10,
            actual_position[1] - 5,
            actual_position[0] + max_width + 10 if max_width else actual_position[0] + 500,
            actual_position[1] + total_height + 5
        )
        draw.rectangle(background_bbox, fill=background_color)
    
    # 绘制每一行文本
    current_y = actual_position[1]
    for line in lines:
        draw.text((actual_position[0], current_y), line, font=font, fill=font_color)
        current_y += font_size + line_spacing
    
    # 保存图片
    if output_path:
        image.save(output_path)
    else:
        if isinstance(image_path, str):
            image.save(image_path)
    
    return image

def render_text_with_effects(image_path, text, output_path=None, font_size=30, font_color=(255, 255, 255),
                           position=(50, 50), effect='shadow', effect_color=(0, 0, 0), effect_offset=2, font_path=None):
    """
    在图片上渲染带特效的文本
    
    Args:
        image_path (str): 输入图片路径
        text (str): 要渲染的文本
        output_path (str): 输出图片路径
        font_size (int): 字体大小
        font_color (tuple): 字体颜色
        position (tuple or str): 文本位置 (x, y) 或 'center' 表示居中
        effect (str): 特效类型 ('shadow', 'outline')
        effect_color (tuple): 特效颜色
        effect_offset (int): 特效偏移量
        font_path (str): 字体文件路径
    """
    
    # 打开图片
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    
    # 转换为RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 设置字体
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 尝试使用系统字体
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                try:
                    # 尝试其他常见字体路径
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                except:
                    # 如果都失败，使用默认字体
                    font = ImageFont.load_default()
                    print("警告: 无法加载系统字体，将使用默认字体")
    except Exception as e:
        font = ImageFont.load_default()
        print(f"字体加载失败: {e}")
    
    # 计算实际渲染位置（支持居中）
    actual_position = calculate_center_position(image, text, font, position)
    
    if effect == 'shadow':
        # 绘制阴影
        shadow_position = (actual_position[0] + effect_offset, actual_position[1] + effect_offset)
        draw.text(shadow_position, text, font=font, fill=effect_color)
        # 绘制主文本
        draw.text(actual_position, text, font=font, fill=font_color)
    
    elif effect == 'outline':
        # 绘制轮廓
        for dx in [-effect_offset, 0, effect_offset]:
            for dy in [-effect_offset, 0, effect_offset]:
                if dx != 0 or dy != 0:
                    outline_position = (actual_position[0] + dx, actual_position[1] + dy)
                    draw.text(outline_position, text, font=font, fill=effect_color)
        # 绘制主文本
        draw.text(actual_position, text, font=font, fill=font_color)
    
    # 保存图片
    if output_path:
        image.save(output_path)
    else:
        if isinstance(image_path, str):
            image.save(image_path)
    
    return image

# 使用示例
if __name__ == "__main__":
    # 基本文本渲染
    render_text_on_image(
        image_path="/home/jjc/codebase/connector/asset/tokyo.png",
        text="Hello World!",
        # text="早安，北京！",
        output_path="output_basic.jpg",
        font_size=100,
        # font_color=(0, 89, 158),
        font_color=(0, 0, 0),
        position='center',
        background_color=None  # 设置为None实现透明背景
    )
    
    # 多行文本渲染
    # render_text_with_multiline(
    #     image_path="/home/jjc/codebase/connector/asset/sky.jpeg",
    #     text="这是一段很长的文本，需要自动换行显示。这样可以更好地控制文本的布局和显示效果。",
    #     output_path="output_multiline.jpg",
    #     font_size=200,
    #     max_width=300,
    #     position=(50, 50)
    # )
    
    # 带特效的文本渲染
    # render_text_with_effects(
    #     image_path="/home/jjc/codebase/connector/asset/sky.jpeg",
    #     text="特效文本",
    #     output_path="output_effects.jpg",
    #     font_size=200,
    #     effect='shadow',
    #     effect_color=(0, 0, 0)
    # )
