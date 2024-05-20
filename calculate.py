import PIL.Image as Image
import numpy as np


def tir():
    # 设置热力图的路径
    test_tir_path = "./dataset/test/tir/"
    tir_img_paths = [f"{test_tir_path}{i}R.jpg" for i in range(1, 1001)]

    mean = 0.0
    std = 0.0

    for tir_img_path in tir_img_paths:
        # print(tir_img_path)

        img = Image.open(tir_img_path).convert("L")

        img_array = np.array(img) / 255.0
        # print("Image array:", img_array)
        # print("Image shape:", img_array.shape)
        mean += np.mean(img_array)
        std += np.std(img_array)
        # print("Mean:", mean)
        # print("Standard Deviation:", std)

    print("Mean:", mean / len(tir_img_paths))
    print("Standard Deviation:", std / len(tir_img_paths))


def rgb():
    test_tir_path = "./dataset/test/rgb/"
    tir_img_path = f"{test_tir_path}1.jpg"

    # 打印图像路径
    print(tir_img_path)

    # 打开图像并转换为灰度图
    img = Image.open(tir_img_path).convert("RGB")

    # 将图像转换为NumPy数组
    img_array = np.array(img)
    print("Image array:", img_array)


if __name__ == "__main__":
    tir()
    # rgb()
