import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
IMAGE_FOLDER = 'student_resource/dataset/images'
FEATURE_OUTPUT_FILE = 'image_features.pkl'

# --- 1. Load DataFrames ---
print("Loading data...")
try:
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    test_df = pd.read_csv('student_resource/dataset/test.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    exit()

# --- 2. Setup Pre-trained Model (EfficientNet) ---
# We use a pre-trained model to extract features (embeddings)
print("Setting up pre-trained EfficientNet model...")

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B0 and set it to evaluation mode
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Remove the final classification layer (the 'head') to get the features
model.classifier = nn.Identity()
model.to(device)
model.eval()

# Define the required transformations for the model
preprocess = transforms.Compose([
    transforms.Resize(256),            # Resize to 256
    transforms.CenterCrop(224),        # Crop to 224 (required size for the model)
    transforms.ToTensor(),             # Convert to Tensor
    # Standard normalization for image models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. Feature Extraction Function ---
def extract_image_features(df):
    """
    Extracts features for all images referenced in a DataFrame and returns a dictionary.
    """
    # Combine all links and use a dictionary to store unique embeddings
    all_links = df['image_link'].dropna().unique()
    image_embeddings = {}

    print(f"\nExtracting features for {len(all_links)} unique images...")

    with torch.no_grad(): # Disable gradient calculations for speed and memory saving
        for link in tqdm(all_links):
            # Get the unique filename from the link
            filename = Path(link).name
            image_path = os.path.join(IMAGE_FOLDER, filename)

            if filename in image_embeddings:
                continue # Already processed this image

            if not os.path.exists(image_path):
                # Skip if the image failed to download (like the 404s you saw)
                continue

            try:
                # 1. Load and preprocess the image
                image = Image.open(image_path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0).to(device) # Add batch dimension

                # 2. Get the embedding
                embedding = model(image_tensor).cpu().numpy().flatten()
                image_embeddings[filename] = embedding
            
            except Exception as e:
                # Catch any issues with corrupted files or loading
                # print(f"Warning: Could not process {filename}. Error: {e}")
                pass 
                
    return image_embeddings

# --- 4. Run Extraction and Save Results ---

# Combine all unique links from both train and test to process everything at once
combined_df = pd.concat([train_df[['image_link']], test_df[['image_link']]]).drop_duplicates()
master_embeddings = extract_image_features(combined_df)

# Convert the dictionary to a DataFrame for easier merging
# The index will be the filename, and columns will be the embedding vector (e.g., dim_0 to dim_1280)
embeddings_df = pd.DataFrame.from_dict(master_embeddings, orient='index')

# Clean up index to just be the filename
embeddings_df.index.name = 'filename'
embeddings_df = embeddings_df.reset_index()

# Save the features to a file so you don't have to run this long process again
embeddings_df.to_pickle(FEATURE_OUTPUT_FILE)
print(f"\n✅ Image features successfully saved to {FEATURE_OUTPUT_FILE}")

# --- 5. Merge Features Back into Training and Test Data ---

# Create a 'filename' column in the original dataframes for merging
train_df['filename'] = train_df['image_link'].apply(lambda x: Path(x).name if pd.notna(x) else None)
test_df['filename'] = test_df['image_link'].apply(lambda x: Path(x).name if pd.notna(x) else None)

# Merge the new features
train_df = pd.merge(train_df, embeddings_df, on='filename', how='left')
test_df = pd.merge(test_df, embeddings_df, on='filename', how='left')

# Drop the helper column
train_df = train_df.drop(columns=['filename'])
test_df = test_df.drop(columns=['filename'])

print(f"\nFinal Train DF shape with new image features: {train_df.shape}")
print(f"Final Test DF shape with new image features: {test_df.shape}")
print("You are now ready to train the multi-modal model!")