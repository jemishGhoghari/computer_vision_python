import cv2
import os
import numpy as np
import multiprocessing

# Define the root directory path
root_dir = "D:\\Computer_Vision\\Kitti\\train"

def calcMeanAndPixels(file_name):
    """
    Calculate the sum of pixel intensities and total number of pixels for a single image.
    """
    file_path = os.path.join(root_dir, file_name)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    if image is not None:
        # Normalize image to range [0, 1]
        gray = image / 255.0

        # Return sum of pixel values and number of pixels
        return np.sum(gray), gray.size
    return 0.0, 0  # Return zero if the image is None


def calcSTD(file_name, mean_value):
    """
    Calculate the squared error and pixel count for a single image.
    """
    squared_error = 0.0
    total_pixels = 0

    file_path = os.path.join(root_dir, file_name)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    if image is not None:
        # Normalize image to range [0, 1]
        gray = image / 255.0

        # Compute squared error using NumPy's vectorized operations
        squared_error = np.sum((gray - mean_value) ** 2)
        total_pixels = gray.size

    return squared_error, total_pixels


if __name__ == "__main__":
    # Get list of all .png files in the directory
    image_files = [f for f in os.listdir(root_dir) if f.endswith(".png")]

    # Step 1: Calculate Mean
    with multiprocessing.Pool() as pool:
        mean_results = pool.map(calcMeanAndPixels, image_files)

    # Aggregate total intensity sum and pixel count to compute mean
    total_intensity_sum = sum(res[0] for res in mean_results)
    total_pixel_count = sum(res[1] for res in mean_results)
    mean_value = total_intensity_sum / total_pixel_count if total_pixel_count > 0 else 0.0

    # Step 2: Calculate Standard Deviation
    with multiprocessing.Pool() as pool:
        std_results = pool.starmap(calcSTD, [(f, mean_value) for f in image_files])

    # Aggregate squared errors and pixel counts
    total_squared_error = sum(res[0] for res in std_results)
    total_pixels = sum(res[1] for res in std_results)

    # Compute Standard Deviation
    std = (total_squared_error / total_pixels) ** 0.5 if total_pixels > 0 else 0.0

    print("Mean Value:", round(mean_value, 6))
    print("Standard Deviation:", round(std, 6))