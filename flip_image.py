import os
import xml.etree.ElementTree as ET
from PIL import Image
import split_utils


def flip_image(image_path, flip_horizontal=True, flip_vertical=True):
    """
    Flip the image horizontally and/or vertically.

    :param image_path: Path to the image file.
    :param flip_horizontal: Boolean indicating whether to flip horizontally.
    :param flip_vertical: Boolean indicating whether to flip vertically.
    :return: Flipped image.
    """
    image = Image.open(image_path)
    if flip_horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def update_xml_for_flipped_image(
    xml_path,
    image_size,
    output_xml_path,
    flip_horizontal=True,
    flip_vertical=True,
):
    """
    Update the XML file for the flipped image.

    :param xml_path: Path to the XML file.
    :param image_size: Tuple of the original image size (width, height).
    :param flip_horizontal: Boolean indicating whether the image was flipped horizontally.
    :param flip_vertical: Boolean indicating whether the image was flipped vertically.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width, height = image_size
    # print(width, height)

    for obj in root.findall("object"):
        point = obj.find("point")
        x = int(point.find("x").text)
        y = int(point.find("y").text)

        if flip_horizontal:
            x = width - x
        if flip_vertical:
            y = height - y

        point.find("x").text = str(x)
        point.find("y").text = str(y)

    tree.write(output_xml_path)


def process_image_and_xml(
    image_path,
    tir_path,
    xml_path,
    output_image_path,
    output_tir_image_path,
    output_xml_path,
    flip_horizontal=True,
    flip_vertical=True,
):
    """
    Process the image and corresponding XML file for flipping.

    :param image_path: Path to the image file.
    :param xml_path: Path to the XML file.
    :param output_image_path: Path to save the flipped image.
    :param output_xml_path: Path to save the updated XML file.
    :param flip_horizontal: Boolean indicating whether to flip horizontally.
    :param flip_vertical: Boolean indicating whether to flip vertically.
    """
    # Flip the image
    image = flip_image(image_path, flip_horizontal, flip_vertical)
    image.save(output_image_path)
    tir_image = flip_image(tir_path, flip_horizontal, flip_vertical)
    tir_image.save(output_tir_image_path)

    # Get the image size
    width, height = image.size

    # Update the XML file
    update_xml_for_flipped_image(
        xml_path,
        (width, height),
        output_xml_path,
        flip_horizontal,
        flip_vertical,
    )


if __name__ == "__main__":
    output_dir = "expansion_dataset"
    os.makedirs(output_dir, exist_ok=True)
    split_utils.rm_dir(output_dir)

    image_paths = split_utils.get_rgb_paths()
    for idx in range(len(image_paths)):
        rgb_path = image_paths[idx]
        tir_path = split_utils.get_tir_path(rgb_path)
        xml_path = split_utils.get_xml_path(rgb_path)
        base_path = os.path.splitext(os.path.basename(rgb_path))[0]
        output_image_path = output_dir + "/" + base_path + "flipped.jpg"
        output_xml_path = output_dir + "/" + base_path + "flipped.xml"
        output_tir_image_path = output_dir + "/" + base_path + "flippedR.jpg"
        process_image_and_xml(
            rgb_path,
            tir_path,
            xml_path,
            output_image_path,
            output_tir_image_path,
            output_xml_path,
            flip_horizontal=idx % 3 != 2,
            flip_vertical=idx % 3 != 1,
        )
        break
