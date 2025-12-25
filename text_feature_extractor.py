# --- Step 1: Install necessary libraries ---
# This ensures sentence-transformers and its dependencies (like PyTorch) are installed.

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import re
import pickle

# --- Configuration ---
TEXT_OUTPUT_FILE = 'semantic_text_features.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2' # A fast and high-quality model

# --- Step 2: Verify and Select GPU Device ---
# This is the most important check. It tells PyTorch to use the GPU.
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
else:
    device = 'cpu'
    print(f"WARNING: CUDA not available. Using device: {device}. This will be very slow.")

# --- Step 3: Load and Clean Data ---
print("Loading data...")
try:
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    test_df = pd.read_csv('student_resource/dataset/test.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}. Check directory structure.")
    exit()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

all_content = pd.concat([train_df['catalog_content'], test_df['catalog_content']])
all_content_cleaned = all_content.apply(clean_text).drop_duplicates().dropna()
print(f"Total unique, cleaned content items: {len(all_content_cleaned)}")

# --- Step 4: Load Model and Generate Embeddings on GPU ---
print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
# The `device` argument tells the model to load itself onto the GPU memory.
model = SentenceTransformer(MODEL_NAME, device=device)

BATCH_SIZE = 256 # A larger batch size is often more efficient on a GPU
all_embeddings = []

print("Generating embeddings...")
# `model.encode` will now perform all its calculations on the GPU.
all_embeddings = model.encode(
    all_content_cleaned.tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True # Directly output to NumPy array
)

print(f"\n✅ Embedding generation complete. Shape: {all_embeddings.shape}")

# --- Step 5: Save Features ---
# Create a dictionary to hold the features and the mapping
output_data = {
    'index_to_text': all_content_cleaned.to_dict(),
    'embeddings': all_embeddings
}

with open(TEXT_OUTPUT_FILE, 'wb') as f:
    pickle.dump(output_data, f)

print(f"Text features saved to {TEXT_OUTPUT_FILE}")
print("\n--- Process Complete ---")