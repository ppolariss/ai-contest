import expansion.flip_image as flip_image
import expansion.split_utils as split_utils
import expansion.downsample_combine as downsample_combine
import expansion.split as split
import expansion.combine as combine
import os
import shutil
import label

if __name__ == "__main__":
    flip, downsample, combined = True, True, True

    output_dir = "expansion_dataset"
    output_rgb_dir = f"{output_dir}/rgb"
    output_tir_dir = f"{output_dir}/tir"
    output_labels_dir = f"{output_dir}/labels"
    # split_utils.rm_dir(output_dir)
    shutil.rmtree(output_dir, ignore_errors=False)
    shutil.copytree("dataset/train", output_dir)
    print("Copying done")

    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_tir_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    if flip:
        flip_image.flip(output_rgb_dir, output_tir_dir, output_labels_dir)
        print("Flipping done")

    if downsample:
        downsample_combine.downsample_combine(
            output_rgb_dir, output_tir_dir, output_labels_dir
        )
        print("Downsampling done")

    if combined:
        temp_dir = "split_temp"
        temp_rgb_dir = f"{temp_dir}/rgb"
        temp_tir_dir = f"{temp_dir}/tir"
        temp_labels_dir = f"{temp_dir}/labels"
        split_utils.rm_dir(temp_dir)

        os.makedirs(temp_rgb_dir, exist_ok=True)
        os.makedirs(temp_tir_dir, exist_ok=True)
        os.makedirs(temp_labels_dir, exist_ok=True)
        split.split(temp_rgb_dir, temp_tir_dir, temp_labels_dir).split()
        print("Splitting done")
        combine.combine(
            temp_rgb_dir,
            temp_tir_dir,
            temp_labels_dir,
            output_rgb_dir,
            output_tir_dir,
            output_labels_dir,
        ).combine()
        print("Combining done")

    label.label(output_labels_dir, output_labels_dir.replace("labels", "hdf5s"))
