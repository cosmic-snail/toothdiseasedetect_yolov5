import numpy as np
import cv2
import os

# 调整最大值
MAX_VALUE = 100


def update(input_img_path, output_img_path, lightness, saturation):
    """
    用于修改图片的亮度和饱和度
    :param input_img_path: 图片路径
    :param output_img_path: 输出图片路径
    :param lightness: 亮度
    :param saturation: 饱和度
    """

    image = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(input_img_path)
    image = image.astype(np.float32) / 255.0

    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1

    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)
    ok = cv2.imwrite(output_img_path, lsImg)
    if not ok:
        raise OSError(output_img_path)


def lightness_saturation(dataset_dir, output_dir, lightness=0, saturation=100):
    os.makedirs(output_dir, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    for name in os.listdir(dataset_dir):
        in_path = os.path.join(dataset_dir, name)
        if os.path.isdir(in_path):
            continue
        if os.path.splitext(name)[1].lower() not in exts:
            continue
        out_path = os.path.join(output_dir, name)
        update(in_path, out_path, lightness, saturation)


def lightness_variants(input_img_path, output_dir, lightness_values, saturation=100):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(input_img_path)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = '.jpg'

    out_paths = []
    for l in lightness_values:
        out_path = os.path.join(output_dir, f"{stem}_L{l}_S{saturation}{ext}")
        update(input_img_path, out_path, l, saturation)
        out_paths.append(out_path)
    return out_paths