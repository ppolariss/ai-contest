# def downsample_and_combine(images):
#     """
#     Downsample four images and combine them such that each 2x2 block in the result
#     image has pixels from each of the four images.

#     :param images: List of four PIL Image objects.
#     :return: Combined downsampled PIL Image.
#     """
#     # Ensure there are exactly four images
#     if len(images) != 4:
#         raise ValueError("Exactly four images are required.")

#     # Convert images to numpy arrays
#     np_images = [np.array(img) for img in images]

#     # Ensure all images have the same dimensions
#     height, width, channels = np_images[0].shape
#     # print(np_images[0].size)
#     for np_img in np_images:
#         if np_img.shape != (height, width, channels):
#             raise ValueError("All images must have the same dimensions.")

#     # Create an empty array for the combined image
#     combined_image = np.zeros((height, width, channels), dtype=np.uint8)
#     print(combined_image.shape)

#     # Fill the combined image
#     for i in range(height):
#         for j in range(width):
#             combined_image[i, j] = np_images[(i % 2) * 2 + (j % 2)][i, j]
#     print(combined_image.shape)

#     return Image.fromarray(combined_image)


from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import os
import split_utils


output_tir_image_path = "downsample_combined/combined_tir_image.jpg"
output_image_path = "downsample_combined/combined_image.jpg"
output_xml_path = "downsample_combined/combined_image.xml"

os.makedirs("downsample_combined", exist_ok=True)

image_paths = [
    "dataset/train/rgb/1.jpg",
    "dataset/train/rgb/2.jpg",
    "dataset/train/rgb/3.jpg",
    "dataset/train/rgb/4.jpg",
]


xml_paths = split_utils.get_xml_paths(image_paths)


def downsample_image(image):
    """
    Downsample the image by a factor of 2.
    Only keep the top-left pixel of each 2x2 block.
    """
    np_image = np.array(image)
    downsampled_image = np_image[::2, ::2, :]
    return Image.fromarray(downsampled_image)


def combine_images(images):
    """
    Combine four images into one larger image.
    Each image should be downsampled to 1/4 of the original image size.
    """
    if len(images) != 4:
        raise ValueError("Exactly four images are required.")

    # Get the size of the downsampled images
    width, height = images[0].size
    combined_image = Image.new("RGB", (width * 2, height * 2))

    # Paste images into the combined image
    combined_image.paste(images[0], (0, 0))
    combined_image.paste(images[1], (width, 0))
    combined_image.paste(images[2], (0, height))
    combined_image.paste(images[3], (width, height))

    return combined_image


def update_xml(xml_paths):
    """
    Update the XML annotation files according to the new combined image.
    """
    if len(xml_paths) != 4:
        raise ValueError("Exactly four XML paths are required.")

    new_annotations = []
    width, height = Image.open(image_paths[0]).size
    width //= 2
    height //= 2

    for idx, xml_path in enumerate(xml_paths):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            point = obj.find("point")
            x = int(point.find("x").text)
            y = int(point.find("y").text)

            # Downsample the coordinates
            new_x = x // 2
            new_y = y // 2

            # Adjust coordinates based on the image's position in the combined image
            if idx == 1:
                new_x += width
            elif idx == 2:
                new_y += height
            elif idx == 3:
                new_x += width
                new_y += height

            new_annotations.append((obj.find("name").text, new_x, new_y))

    # Create a new XML tree for the combined image
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width * 2)
    ET.SubElement(size, "height").text = str(height * 2)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for name, x, y in new_annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        point = ET.SubElement(obj, "point")
        ET.SubElement(point, "x").text = str(x)
        ET.SubElement(point, "y").text = str(y)

    tree = ET.ElementTree(root)
    tree.write(output_xml_path)


def process_images_and_annotations():
    """
    Downsample images, combine them, and update the XML annotations.
    """
    if len(image_paths) != 4 or len(xml_paths) != 4:
        raise ValueError("Exactly four images and four XML paths are required.")

    downsampled_images = [
        downsample_image(Image.open(image_path)) for image_path in image_paths
    ]
    combined_image = combine_images(downsampled_images)
    combined_image.save(output_image_path)

    tir_paths = split_utils.get_tir_paths(image_paths)
    downsampled_tir_images = [
        downsample_image(Image.open(tir_path)) for tir_path in tir_paths
    ]
    combined_tir_image = combine_images(downsampled_tir_images)
    combined_tir_image.save(output_tir_image_path)

    print(image_paths, xml_paths, tir_paths)
    update_xml(xml_paths)


if __name__ == "__main__":
    process_images_and_annotations()

# # print(result_image.size)
# # Save the result
# result_image.save("combined_downsampled_image.jpg")
# result_image.show()
