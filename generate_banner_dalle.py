import requests
from PIL import Image, ImageFilter
import io
import base64
import numpy as np
from openai import OpenAI

# OpenAI API 配置
API_KEY = ""  # 替换为您的 OpenAI API 密钥
client = OpenAI(api_key=API_KEY)

# 本地图像路径
ORIGINAL_IMAGE_PATH = "logo.png"  # 原始图像路径
RESIZED_IMAGE_PATH = "resized_logo.png"  # 调整后的图像路径

# 尺寸配置
API_SIZE = (1024, 1024)  # DALL-E API 支持的尺寸
FINAL_SIZE = (1536, 700)  # 最终 Banner 尺寸
# 若需 1152x408，取消注释以下行：
# API_SIZE = (1024, 1024)
# FINAL_SIZE = (1152, 408)

# 获取图像边缘平均颜色
def get_edge_average_color(image):
    img_array = np.asarray(image.convert("RGB"))  # 使用 np.asarray 兼容 numpy 2.0
    height, width = img_array.shape[:2]
    top_edge = img_array[0:10, :, :].mean(axis=(0, 1))
    bottom_edge = img_array[-10:, :, :].mean(axis=(0, 1))
    left_edge = img_array[:, 0:10, :].mean(axis=(0, 1))
    right_edge = img_array[:, -10:, :].mean(axis=(0, 1))
    avg_color = np.mean([top_edge, bottom_edge, left_edge, right_edge], axis=0).astype(int)
    return tuple(avg_color)

# 预处理图像：调整尺寸并用模糊的边缘颜色填充
try:
    # 加载原始图像
    original_image = Image.open(ORIGINAL_IMAGE_PATH).convert("RGBA")
    original_width, original_height = original_image.size

    # 获取边缘平均颜色
    edge_color = get_edge_average_color(original_image)

    # 创建画布，用边缘平均颜色填充
    new_image = Image.new("RGB", API_SIZE, edge_color)
    
    # 应用高斯模糊
    new_image = new_image.filter(ImageFilter.GaussianBlur(radius=10))

    # 将原始图像居中粘贴
    offset_x = (API_SIZE[0] - original_width) // 2
    offset_y = (API_SIZE[1] - original_height) // 2
    new_image.paste(original_image, (offset_x, offset_y), original_image)

    # 保存预处理图像
    new_image.save(RESIZED_IMAGE_PATH, "PNG")
    print(f"图像预处理成功，保存为 '{RESIZED_IMAGE_PATH}'")

except FileNotFoundError:
    print(f"错误: 找不到文件 '{ORIGINAL_IMAGE_PATH}'，请检查路径是否正确。")
    exit(1)
except Exception as e:
    print(f"图像预处理异常: {str(e)}")
    exit(1)

# DALL-E API 调用
try:
    with open(RESIZED_IMAGE_PATH, "rb") as image_file:
        response = client.images.edit(
            image=image_file,
            prompt=(
                "A modern banner with bold white text 'ANTPOOL Mining Pool Launches SHIC Merged-mining' centered in large font. "
                "Include the SHIC cryptocurrency logo from the reference image prominently in the center, "
                "with futuristic mining rigs and glowing circuit patterns in the background."
            ),
            n=1,
            size="1024x1024"
        )

    # 获取生成的图像
    image_url = response.data[0].url
    image_response = requests.get(image_url)
    image = Image.open(io.BytesIO(image_response.content)).convert("RGB")

    # 后处理：调整到最终尺寸
    if image.size != FINAL_SIZE:
        if image.size[1] < FINAL_SIZE[1]:  # 高度不足，填充
            edge_color = get_edge_average_color(image)
            final_image = Image.new("RGB", FINAL_SIZE, edge_color)
            final_image = final_image.filter(ImageFilter.GaussianBlur(radius=10))
            offset_y = (FINAL_SIZE[1] - image.size[1]) // 2
            final_image.paste(image, (0, offset_y))
            image = final_image
        else:  # 高度过大，裁剪
            crop_y = (image.size[1] - FINAL_SIZE[1]) // 2
            image = image.crop((0, crop_y, FINAL_SIZE[0], crop_y + FINAL_SIZE[1]))

    # 保存最终图像
    image.save("generated_banner.png", "PNG")
    print(f"Banner 生成成功，保存为 'generated_banner.png'，尺寸为 {FINAL_SIZE}")

except Exception as e:
    print(f"API 调用或后处理异常: {str(e)}")