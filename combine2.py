from PIL import Image
import numpy as np
import os


def downsample_and_combine(images):
    """
    Downsample four images and combine them such that each 2x2 block in the result
    image has pixels from each of the four images.

    :param images: List of four PIL Image objects.
    :return: Combined downsampled PIL Image.
    """
    # Ensure there are exactly four images
    if len(images) != 4:
        raise ValueError("Exactly four images are required.")

    # Convert images to numpy arrays
    np_images = [np.array(img) for img in images]

    # Ensure all images have the same dimensions
    height, width, channels = np_images[0].shape
    # print(np_images[0].size)
    for np_img in np_images:
        if np_img.shape != (height, width, channels):
            raise ValueError("All images must have the same dimensions.")

    # Create an empty array for the combined image
    combined_image = np.zeros((height, width, channels), dtype=np.uint8)
    print(combined_image.shape)

    # Fill the combined image
    for i in range(height):
        for j in range(width):
            combined_image[i, j] = np_images[(i % 2) * 2 + (j % 2)][i, j]
    print(combined_image.shape)

    return Image.fromarray(combined_image)


# Load images
image_paths = [
    "dataset/train/rgb/100.jpg",
    "dataset/train/rgb/200.jpg",
    "dataset/train/rgb/300.jpg",
    "dataset/train/rgb/400.jpg",
]

images = [Image.open(image_path) for image_path in image_paths]

# Downsample and combine images
result_image = downsample_and_combine(images)

# print(result_image.size)
# Save the result
result_image.save("combined_downsampled_image.jpg")
result_image.show()
