import pandas as pd
import numpy as np
import os

# Imports are safe to leave here
from student_resource.src.utils import download_images

# 🛡️ This is the guard that fixes the problem
if __name__ == '__main__':
    print("Loading datasets to get image links...")
    # Load the datasets
    try:
        train_df = pd.read_csv('student_resource/dataset/train.csv')
        test_df = pd.read_csv('student_resource/dataset/test.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure you are running this script from the 'D:\\project\\ML' directory.")
        exit()

    # Define where to save the images
    IMAGE_DOWNLOAD_PATH = "student_resource/dataset/images/"
    print(f"Images will be saved to: {IMAGE_DOWNLOAD_PATH}")

    # Combine all unique image links from both train and test sets
    train_links = train_df['image_link'].dropna().unique()
    test_links = test_df['image_link'].dropna().unique()
    all_image_links = np.unique(np.concatenate([train_links, test_links]))

    print(f"Found {len(all_image_links)} unique images to download.")
    print("Starting download process... This may take several hours.")

    # Run the downloader function
    download_images(all_image_links, IMAGE_DOWNLOAD_PATH)

    print("\nImage download process complete!")