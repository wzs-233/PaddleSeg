import numpy as np
import os
import cv2


input_path = "/workspace/paddle_paddle/PaddleSeg/data/MaVeCoDD/train/alpha"
output_path = "test_output/train_color"
if not os.path.exists(output_path):
    os.makedirs(output_path)


files = os.listdir(input_path)
# mask: H x W, 值为 0 / 1 或 0 / 255
for image in files:
    mask = cv2.imread(os.path.join(input_path, image),cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(bool)

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color_mask[mask] = [255, 0, 0]   # 前景红色
    color_mask[~mask] = [0, 255, 0]    # 背景黑色
    cv2.imwrite(os.path.join(output_path, image), color_mask)