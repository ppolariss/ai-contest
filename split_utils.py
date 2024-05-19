import os
import shutil

gt_dir = "./dataset/train/labels/"
tir_dir = "./dataset/train/tir/"
img_dir = "./dataset/train/rgb/"


def get_output_dir():
    return "./split_test"


def get_tir_paths(rgb_paths) -> list[str]:
    tir_paths = []
    for rgb_path in rgb_paths:
        base_path = os.path.splitext(os.path.basename(rgb_path))[0]
        tir_path = os.path.join(tir_dir, base_path + "R.jpg")
        tir_paths.append(tir_path)
    return tir_paths


def get_tir_path(rgb_path) -> str:
    base_path = os.path.splitext(os.path.basename(rgb_path))[0]
    tir_path = os.path.join(tir_dir, base_path + "R.jpg")
    return tir_path


def get_xml_path(rgb_path) -> str:
    base_path = os.path.splitext(os.path.basename(rgb_path))[0]
    xml_path = os.path.join(gt_dir, base_path + "R.xml")
    return xml_path


def get_xml_paths(rgb_paths) -> list[str]:
    xml_paths = []
    for rgb_path in rgb_paths:
        base_path = os.path.splitext(os.path.basename(rgb_path))[0]
        xml_path = os.path.join(gt_dir, base_path + "R.xml")
        xml_paths.append(xml_path)
    return xml_paths


def rm_dir(output_dir):
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)

        # If the entry is a directory, recursively clear it
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)


def get_rgb_paths() -> list[str]:
    img_paths = [
        os.path.join(img_dir, filename)
        for filename in os.listdir(img_dir)
        if filename.endswith(".jpg")
    ]
    return img_paths
