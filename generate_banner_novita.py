import os
from PIL import Image, ImageFilter
import numpy as np
from novita_client import NovitaClient, Img2ImgV3Embedding
from novita_client.utils import base64_to_image

# Novita.ai API 配置
API_KEY = ""
API_URI = "https://api.novita.ai/v3/async/img2img"
client = NovitaClient(API_KEY, API_URI)

# 本地图像路径
ORIGINAL_IMAGE_PATH = "logo.png"  # 原始图像路径
RESIZED_IMAGE_PATH = "resized_logo.png"  # 调整后的图像路径
OUTPUT_PATH = "generated_banner.png"  # 输出 Banner 路径

# 尺寸配置
FINAL_SIZE = (1536, 700)  # 最终 Banner 尺寸
# 若需 1152x408，修改：
# FINAL_SIZE = (1152, 408)

# 获取边缘平均颜色
def get_edge_average_color(image):
    img_array = np.asarray(image.convert("RGB"))  # 兼容 numpy 2.0
    height, width = img_array.shape[:2]
    top_edge = img_array[0:10, :, :].mean(axis=(0, 1))
    bottom_edge = img_array[-10:, :, :].mean(axis=(0, 1))
    left_edge = img_array[:, 0:10, :].mean(axis=(0, 1))
    right_edge = img_array[:, -10:, :].mean(axis=(0, 1))
    avg_color = np.mean([top_edge, bottom_edge, left_edge, right_edge], axis=0).astype(int)
    return tuple(avg_color)

# 预处理图像：调整尺寸并用模糊的边缘颜色填充
try:
    original_image = Image.open(ORIGINAL_IMAGE_PATH).convert("RGBA")
    original_width, original_height = original_image.size
    edge_color = get_edge_average_color(original_image)
    new_image = Image.new("RGB", FINAL_SIZE, edge_color)
    new_image = new_image.filter(ImageFilter.GaussianBlur(radius=10))
    offset_x = (FINAL_SIZE[0] - original_width) // 2
    offset_y = (FINAL_SIZE[1] - original_height) // 2
    new_image.paste(original_image, (offset_x, offset_y), original_image)
    new_image.save(RESIZED_IMAGE_PATH, "PNG")
    print(f"图像预处理成功，保存为 '{RESIZED_IMAGE_PATH}'")

except FileNotFoundError:
    print(f"错误: 找不到文件 '{ORIGINAL_IMAGE_PATH}'，请检查路径是否正确。")
    exit(1)
except Exception as e:
    print(f"图像预处理异常: {str(e)}")
    exit(1)

# Novita.ai img2img_v3 调用
try:
    res = client.img2img_v3(
        model_name="MeinaHentai_V5.safetensors",  # 可替换为其他模型，例如 "realisticVisionV51_v51VAE_13732.safetensors"
        image_num=1,  # 指定生成 1 张图像
        steps=50,  # 增加步数以提高质量
        width=FINAL_SIZE[0],  # 直接使用目标宽度
        height=FINAL_SIZE[1],  # 直接使用目标高度
        input_image=RESIZED_IMAGE_PATH,  # 本地图像路径
        prompt=(
            "A modern banner with bold white text 'ANTPOOL Mining Pool Launches SHIC Merged-mining' centered in large sans-serif font. "
            "Include the SHIC cryptocurrency logo from the reference image prominently in the center, "
            "with detailed futuristic mining rigs and neon-blue glowing circuit patterns in the background."
        ),
        strength=0.5,  # 平衡参考图像和提示词
        guidance_scale=9.0,  # 提高提示词遵循程度
        embeddings=[Img2ImgV3Embedding(model_name=_) for _ in [
            "bad-image-v2-39000",
            "verybadimagenegative_v1.3_21434",
            "BadDream_53202",
            "badhandv4_16755",
            "easynegative_8955.safetensors"
        ]],  # 负面嵌入，优化生成质量
        seed=-1,  # 随机种子
        sampler_name="DPM++ 2M Karras",  # 采样器
        clip_skip=2  # 优化生成风格
    )

    # 保存生成的图像
    base64_to_image(res.images_encoded[0]).save(OUTPUT_PATH)
    print(f"Banner 生成成功，保存为 '{OUTPUT_PATH}'，尺寸为 {FINAL_SIZE}")

except Exception as e:
    print(f"API 调用异常: {str(e)}")