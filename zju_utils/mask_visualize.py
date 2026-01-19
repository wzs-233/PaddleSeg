import matplotlib.pyplot as plt
import cv2
import numpy as np

# 替换为你处理后的图片路径
img_path = '/workspace/paddle_paddle/PaddleSeg/zju_utils/berlin_000003_000019_gtFine_color.png'

# 1. 读取图片 (注意不要加 cv2.IMREAD_GRAYSCALE，因为已经是单通道了，或者加上也没事)
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# 2. 打印看看它的唯一值，确认只有 0 和 1
print(f"包含的像素值: {np.unique(img)}")

# 3. 可视化
plt.figure(figsize=(10, 5))

# 左边：显示原始数据（Matplotlib 会自动上色，紫色是0，黄色是1）
plt.subplot(1, 2, 1)
plt.title("Auto Color (Purple=0, Yellow=1)")
plt.imshow(img) 
plt.colorbar()

# 右边：模拟黑白图 (手动乘以255以便人眼观察)
plt.subplot(1, 2, 2)
plt.title("Black & White View")
plt.imshow(img * 255, cmap='gray') 

plt.show()