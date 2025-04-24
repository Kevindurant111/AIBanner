import requests
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import io
import base64
import numpy as np
import re

# Stability AI API 配置
API_KEY = ""
API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# 输出路径
OUTPUT_PATHS = {
    "1536x700_en": "generated_banner_1536x700_en.png",
    "1536x700_cn": "generated_banner_1536x700_cn.png",
    "1152x408_en": "generated_banner_1152x408_en.png",
    "1152x408_cn": "generated_banner_1152x408_cn.png"
}

# 尺寸配置
API_SIZE = (1536, 640)  # API 允许的尺寸
SIZE_1 = (1536, 700)    # 第一尺寸
SIZE_2 = (1152, 408)    # 第二尺寸（缩放生成）

# 艺术字颜色
TEXT_COLOR = (255, 165, 0)  # 橙黄色
SHADOW_COLOR = (50, 50, 50)  # 阴影颜色（深灰）
STROKE_COLOR = (100, 100, 100)  # 描边颜色（中灰）

# 艺术字文本
TEXT_EN = "ANTPOOL\nLaunches SHIC Merged-mining"
TEXT_CN = "ANTPOOL\n上线 SHIC 合并挖矿"  # 已确认中文文本

# 判断是否为中文文本
def is_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)

# 获取边缘平均颜色
def get_edge_average_color(image):
    img_array = np.asarray(image.convert("RGB"))
    height, width = img_array.shape[:2]
    top_edge = img_array[0:10, :, :].mean(axis=(0, 1))
    bottom_edge = img_array[-10:, :, :].mean(axis=(0, 1))
    left_edge = img_array[:, 0:10, :].mean(axis=(0, 1))
    right_edge = img_array[:, -10:, :].mean(axis=(0, 1))
    avg_color = np.mean([top_edge, bottom_edge, left_edge, right_edge], axis=0).astype(int)
    return tuple(avg_color)

# 添加艺术字（自适应字体大小，加粗和立体效果，增加行间距）
def add_art_text(image, text, font_path=None, font_size=60, text_color=TEXT_COLOR, shadow_color=SHADOW_COLOR, stroke_color=STROKE_COLOR):
    draw = ImageDraw.Draw(image)
    target_width = image.width * 4 / 5  # 文字宽度为画面的 4/5

    # 增加中文字体大小
    if is_chinese(text):
        font_size = 70  # 中文字体初始大小更大

    # 根据文本语言选择字体路径
    if is_chinese(text):
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
            "arial.ttf"
        ]

    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            print(f"成功加载字体: {path}")
            break
        except IOError:
            continue

    if font is None:
        print(f"错误: 未找到支持的字体，请安装 Noto Sans CJK（sudo apt-get install fonts-noto-cjk）或 WenQuanYi（sudo apt-get install fonts-wqy-zenhei）")
        font = ImageFont.load_default()

    lines = text.split("\n")
    max_width = 0
    total_height = 0
    line_heights = []

    # 计算初始文本尺寸
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        max_width = max(max_width, text_width)
        total_height += text_height
        line_heights.append(text_height)

    # 自适应调整字体大小
    while max_width > target_width and font_size > 10:
        font_size -= 5
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()
        max_width = 0
        total_height = 0
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            max_width = max(max_width, text_width)
            total_height += text_height
            line_heights.append(text_height)

    print(f"最终字体大小: {font_size} 像素，文本宽度: {max_width} 像素")

    # 增加行间距
    line_spacing = 15  # 额外行间距（像素）
    total_height_with_spacing = total_height + line_spacing * (len(lines) - 1)

    x = (image.width - max_width) // 2
    y = (image.height - total_height_with_spacing) // 2

    # 绘制阴影（立体效果）
    shadow_offset = 3  # 阴影偏移像素
    y_shadow = y
    for i, line in enumerate(lines):
        draw.text((x + shadow_offset, y_shadow + shadow_offset), line, fill=shadow_color, font=font)
        y_shadow += line_heights[i] + line_spacing

    # 重置 y 坐标
    y_stroke = y
    # 绘制描边（加粗效果）
    stroke_width = 1  # 描边宽度
    for i, line in enumerate(lines):
        for dx, dy in [(-stroke_width, 0), (stroke_width, 0), (0, -stroke_width), (0, stroke_width)]:
            draw.text((x + dx, y_stroke + dy), line, fill=stroke_color, font=font)
        y_stroke += line_heights[i] + line_spacing

    # 重置 y 坐标
    y_text = y
    # 绘制主文字
    for i, line in enumerate(lines):
        draw.text((x, y_text), line, fill=text_color, font=font)
        y_text += line_heights[i] + line_spacing

    return image

# API 请求 payload
payload = {
    "text_prompts": [
        {
            "text": (
                "A clean, minimalist poster-style banner with a bold, flat design featuring scattered 2D coins, pickaxes, and abstract line elements in a balanced, grid-based composition. "
                "Use a solid color background with vibrant, contrasting pure colors (e.g., light black, light silver, or light green). Emphasize clean negative space and geometric simplicity, ultra-realistic, high contrast, no cryptocurrency logos or mining rigs."
                "There should be no text on the poster."
            ),
            "weight": 1
        },
        {
            "text": "blurry, low quality, distorted, text artifacts, extra limbs, cryptocurrency logos, mining rigs, gradients, complex textures, overly detailed",
            "weight": -1
        }
    ],
    "steps": 50,
    "width": API_SIZE[0],
    "height": API_SIZE[1],
    "cfg_scale": 9.0,
    "style_preset": "digital-art",
    "seed": 0
}

# API 请求头
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# 发送 API 请求
try:
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        image_data = base64.b64decode(result["artifacts"][0]["base64"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 计算生成图像的边缘颜色
        edge_color = get_edge_average_color(image)
        print(f"生成图像边缘颜色: {edge_color}")

        # 后处理：生成 1536x700
        if image.size != SIZE_1:
            final_image = Image.new("RGB", SIZE_1, edge_color)
            final_image = final_image.filter(ImageFilter.GaussianBlur(radius=5))
            offset_y = (SIZE_1[1] - image.size[1]) // 2
            final_image.paste(image, (0, offset_y))
            image_1536x700 = final_image
        else:
            image_1536x700 = image

        # 生成四张图像
        for size_key, output_path in OUTPUT_PATHS.items():
            size = SIZE_1 if "1536x700" in size_key else SIZE_2
            text = TEXT_EN if "_en" in size_key else TEXT_CN

            # 缩放图像（如果需要）
            if size == SIZE_2:
                img = image_1536x700.resize(SIZE_2, Image.LANCZOS)
            else:
                img = image_1536x700.copy()

            # 添加艺术字（加粗和立体效果）
            img = add_art_text(img, text, font_size=60)

            # 保存图像
            img.save(output_path, "PNG")
            print(f"Banner 生成成功，保存为 '{output_path}'，尺寸为 {size}")

    else:
        print(f"错误: {response.status_code} - {response.text}")

except Exception as e:
    print(f"API 调用或后处理异常: {str(e)}")