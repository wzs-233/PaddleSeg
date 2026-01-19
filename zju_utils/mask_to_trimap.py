import cv2
import numpy as np
import os

def mask_to_trimap(mask, k=10, iterations=1):
    """
    将二值掩膜转换为 Trimap
    :param mask: 输入的二值掩膜 (单通道)
    :param k: 腐蚀膨胀的卷积核大小，建议 >= 3，且随着图片分辨率增大而增大
    :param iterations: 腐蚀膨胀的迭代次数
    :return: Trimap图像
    """
    # [诊断 1] 检查输入是否为 0/1 格式 (常见于分割数据集)
    if np.max(mask) == 1:
        print("  -> 提示: 检测到 Mask 值为 [0, 1]，自动修正为 [0, 255]")
        mask = mask * 255

    # [诊断 2] 检查 k 值是否过小
    if k < 3:
        print(f"  -> 警告: k={k} 太小了！这将导致没有灰色边缘。建议至少设为 3 或更大。")

    # 确保 mask 是严格的二值图像
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((k, k), np.uint8)

    # 1. 腐蚀：收缩
    sure_fg = cv2.erode(mask, kernel, iterations=iterations)

    # 2. 膨胀：扩张
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    # 3. 计算未知区域
    unknown = cv2.subtract(dilated, sure_fg)

    # [诊断 3] 检查是否生成了未知区域
    unknown_pixel_count = np.sum(unknown == 255)
    if unknown_pixel_count == 0:
        print("  -> 错误: 未生成任何灰色区域(Unknown)。可能是 k 值太小，或者 mask 是全黑/全白的。")

    # 4. 构建 Trimap
    trimap = np.zeros_like(mask)
    trimap[unknown == 255] = 128  # 灰色：未知区域
    trimap[sure_fg == 255] = 255  # 白色：确定前景

    return trimap

def generate_dummy_mask(filename="dummy_mask.png"):
    """生成一个测试用的圆形掩膜"""
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(img, (256, 256), 150, 255, -1)
    cv2.imwrite(filename, img)
    return filename

if __name__ == '__main__':
    mask_path = "/workspace/paddle_paddle/PaddleSeg/Matting/output/results/new/val/fg/colorImage1013_1.png" 
    if not os.path.exists(mask_path):
        generate_dummy_mask(mask_path)

    print(f"正在处理: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"错误: 无法读取文件 {mask_path}")
    else:
        # --- 关键修改 ---
        # 尝试将 k 设为 20，让灰色边缘更明显
        # 如果你的图片分辨率很高（如 1080p 或 4K），k 需要设得更大（比如 30-50）
        current_k = 20
        print(f"使用核大小 k={current_k} 进行处理...")
        
        trimap = mask_to_trimap(mask, k=current_k, iterations=1)

        output_filename = "output_trimap.png"
        cv2.imwrite(output_filename, trimap)
        
        unique_values = np.unique(trimap)
        print(f"处理完成！已保存为: {output_filename}")
        print(f"Trimap 中包含的像素值: {unique_values}")
        
        if 128 in unique_values:
            print("成功: 包含了灰色区域 (128)。")
        else:
            print("失败: 仍然没有灰色区域，请尝试增大 k 值。")