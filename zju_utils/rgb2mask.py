import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置 =================
input_dir = '/workspace/paddle_paddle/PaddleSeg/Matting/data/MaVeCoDD/val/alpha'      # 你的 JPG 标签文件夹
output_dir = 'test_output/alpha'     # 输出的 PNG 文件夹
threshold_value = 30                       # 【关键】容忍度：亮度大于30才算前景，过滤掉JPG黑底噪点
# =======================================

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"开始转换: JPG彩色 -> PNG二值掩码 (阈值: {threshold_value})")

files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

for filename in tqdm(files):
    img_path = os.path.join(input_dir, filename)
    
    # 1. 读取原始 JPG 图片
    img = cv2.imread(img_path) 
    if img is None: continue

    # 2. 转换为灰度图 (将三通道 RGB 合并为单通道亮度)
    # 这一步会把彩色变成 0-255 的灰度值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 【关键步骤】阈值处理 (Binarization with Threshold)
    # 逻辑：如果像素值 > 30，设为 1 (前景)；否则设为 0 (背景)
    # 使用 30 而不是 0，是为了切除 JPG 压缩带来的黑色背景噪点
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. 强制转换类型 (PaddleSeg 需要 uint8)
    mask = mask.astype(np.uint8)

    # 5. 保存为 PNG (必须改后缀！)
    # 替换后缀名为 .png
    new_filename = os.path.splitext(filename)[0] + ".png"
    save_path = os.path.join(output_dir, new_filename)
    
    # 使用无损 PNG 保存，确保 0 和 1 不会再变回 0.8 或 1.2
    cv2.imwrite(save_path, mask)

print("转换完成！")
print(f"请检查输出目录: {output_dir}")
print("注意：新图片在普通看图软件里全是黑的（因为前景是1），这是正常的！")