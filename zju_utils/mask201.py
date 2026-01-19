import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 输入你的原始 Label 文件夹路径
input_dir = '/workspace/paddle_paddle/PaddleSeg/Matting/data/MaVeCoDD/train/alpha' 

# 输出处理后的 Label 文件夹路径（建议新建一个，别覆盖原图）
output_dir = 'test_output' 

# 阈值：大于这个值的像素会被设为 1 (目标)，小于等于的设为 0 (背景)
threshold = 0 
# ===========================================

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"正在处理图片，将 0-255 映射为 [0, 1]...")

file_list = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for filename in tqdm(file_list):
    img_path = os.path.join(input_dir, filename)
    
    # 1. 以灰度模式读取
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"无法读取: {filename}")
        continue
    
    # 2. 二值化处理 (关键步骤)
    # 创建一个空的 mask，形状和原图一样
    new_mask = np.zeros_like(img, dtype=np.uint8)
    
    # 将原图中大于阈值的像素设为 1
    new_mask[img > threshold] = 9
    
    # 小于等于阈值的保持为 0 (初始化时已经是0了，所以不用动)

    # 3. 保存
    # 注意：保存后的图片在电脑上看是“全黑”的，因为像素值只有0和1
    # 这是正常的！千万不要怀疑它！模型能看见！
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, new_mask)

print("处理完成！")
print(f"新数据集保存在: {output_dir}")
print("请务必修改 train_list.txt 中的路径指向这个新文件夹！")