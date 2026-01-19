import os
import cv2
import numpy as np
from tqdm import tqdm

# 替换为你的 Label 路径
label_dir = '/workspace/paddle_paddle/PaddleSeg/data/MaVeCoDD/train/alpha' 

unique_values = set()

print("正在扫描数据集像素值...")
for filename in tqdm(os.listdir(label_dir)):
    if filename.endswith(('.png', '.jpg')):
        path = os.path.join(label_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # 获取这张图里的所有唯一像素值
        u = np.unique(img)
        unique_values.update(u)

print(f"你的数据集里包含的像素值有: {sorted(list(unique_values))}")