
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import lightgbm as lgb
import warnings
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import joblib
from tqdm.notebook import tqdm
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

warnings.filterwarnings("ignore", category=UserWarning)

# --- Define paths ---
PROJECT_FOLDER = '.' # Assumes the script is in the root of your project folder
CSV_TRAIN_PATH = os.path.join(PROJECT_FOLDER, 'student_resource', 'dataset', 'train.csv')
CSV_TEST_PATH = os.path.join(PROJECT_FOLDER, 'student_resource', 'dataset', 'test.csv')
IMAGE_FEATURES_TRAIN_PATH = os.path.join(PROJECT_FOLDER, 'image_features.pkl')
DUPLICATES_FOLDER_PATH = os.path.join(PROJECT_FOLDER, 'student_resource', 'dataset', 'Visual_Duplicates')

# --- SMAPE Metric ---
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

# ----------------------------------------------------------------------------------------------------
## --- Step 2: Data Loading and Preparation ---
print("--- Loading & Preparing Data ---")
try:
    df_train_raw = pd.read_csv(CSV_TRAIN_PATH, encoding='latin-1')
    df_test_raw = pd.read_csv(CSV_TEST_PATH, encoding='latin-1')
    # This line is now correctly indented
    X_image_features_raw = joblib.load(IMAGE_FEATURES_TRAIN_PATH)
    print("✅ All data sources loaded.")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}"); exit()

min_rows = min(len(df_train_raw), len(X_image_features_raw))
if len(df_train_raw) > min_rows: df_train_raw = df_train_raw.iloc[:min_rows]
if len(X_image_features_raw) > min_rows: X_image_features_raw = X_image_features_raw[:min_rows]
print(f"✅ Data sources synchronized to {min_rows} rows.")

def engineer_features(df):
    df_copy = df.copy()
    ipq_patterns = re.compile(r'(\d{1,4}[\.\d]*)\s*(?:pack|count|pk|ct|pcs|pieces|bottles|cans|bags|oz|fl oz|grams|g|kg|lbs|ml|liters|l)\b', re.IGNORECASE)
    def find_ipq(text):
        if not isinstance(text, str): return 1
        match = ipq_patterns.search(text)
        if match:
            try: return float(match.group(1).rstrip('.'))
            except (ValueError, IndexError): return 1
        return 1
    df_copy['ipq'] = df_copy['catalog_content'].apply(find_ipq)
    def extract_value(text):
        if not isinstance(text, str): return np.nan
        match = re.search(r'value:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        return float(match.group(1)) if match else np.nan
    df_copy['Value'] = df_copy['catalog_content'].apply(extract_value)
    df_copy['original_content'] = df_copy['catalog_content'].fillna('')
    def extract_brand(text):
        if not isinstance(text, str): return "unknown"
        match = re.search(r'brand:\s*([a-zA-Z0-9\s]+)', text, re.IGNORECASE)
        return match.group(1).strip().lower() if match else "unknown"
    df_copy['brand'] = df_copy['catalog_content'].apply(extract_brand)
    def extract_unit(text):
        if not isinstance(text, str): return "unknown"
        match = re.search(r'unit:\s*([a-zA-Z\s]+)', text, re.IGNORECASE)
        return match.group(1).strip().lower() if match else "unknown"
    df_copy['Unit'] = df_copy['catalog_content'].apply(extract_unit)
    df_copy['is_organic'] = df_copy['original_content'].str.contains('organic', case=False).astype(int)
    df_copy['is_kit'] = df_copy['original_content'].str.contains('kit|bundle|pack', case=False).astype(int)
    return df_copy
df_train = engineer_features(df_train_raw); df_test = engineer_features(df_test_raw)
print("✅ Feature engineering complete.")

for col in ['ipq', 'Value']:
    cap_val = df_train[col].quantile(0.999)
    df_train[col] = df_train[col].clip(upper=cap_val); df_test[col] = df_test[col].clip(upper=cap_val)
print("✅ Numerical feature capping complete.")

nan_price_indices = df_train.index[df_train['price'].isnull()]
df_train.dropna(subset=['price'], inplace=True)
X_image_features = np.delete(X_image_features_raw, nan_price_indices, axis=0)
df_train.reset_index(drop=True, inplace=True)

df_train['log_price'] = np.log1p(df_train['price'])
bottom_quantile = df_train['log_price'].quantile(0.05); top_quantile = df_train['log_price'].quantile(0.95)
valid_indices_mask = (df_train['log_price'] >= bottom_quantile) & (df_train['log_price'] <= top_quantile)
df_train = df_train[valid_indices_mask].reset_index(drop=True)
X_image_features = X_image_features[valid_indices_mask.values]
assert len(df_train) == len(X_image_features), "❌ Mismatch after cleaning!"
print(f"✅ Final training data shape: {df_train.shape}")

id_to_group_map = {}
group_counter = 0
for folder_path, _, filenames in os.walk(DUPLICATES_FOLDER_PATH):
    if any(f.endswith('.jpg') for f in filenames):
        for filename in filenames:
            sample_id = int(os.path.splitext(filename)[0]); id_to_group_map[sample_id] = group_counter
        group_counter += 1
groups = df_train['sample_id'].map(id_to_group_map)
if groups.isnull().any():
    next_unique_group_id = group_counter; nan_indices = groups[groups.isnull()].index
    groups.loc[nan_indices] = range(next_unique_group_id, next_unique_group_id + len(nan_indices))
groups = groups.astype(int)
print(f"✅ Groups created.")

embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
print("Generating text embeddings for training data...")
X_text_train = embedder.encode(df_train['original_content'].tolist(), show_progress_bar=True)
df_brands_train = pd.get_dummies(df_train['brand'], prefix='brand', dummy_na=True)
df_units_train = pd.get_dummies(df_train['Unit'], prefix='unit', dummy_na=True)
other_feature_cols = ['ipq', 'Value', 'is_organic', 'is_kit']
X_other_features_df_train = df_train[other_feature_cols].join(df_brands_train).join(df_units_train)
y_train = df_train['log_price']
print("✅ All training features prepared.")

# ----------------------------------------------------------------------------------------------------
## --- Step 3: Train the Text-Only Expert Model ---
print("\n--- ✍️ Training Text-Only Expert ---")
gkf = GroupKFold(n_splits=5)
metrics_text = {'smape': [], 'r2': []}
numerical_cols = ['ipq', 'Value']
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_text_train, y_train, groups=groups)):
    X_text_fold_train, X_text_fold_val = X_text_train[train_idx], X_text_train[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    X_other_fold_train = X_other_features_df_train.iloc[train_idx].copy()
    X_other_fold_val = X_other_features_df_train.iloc[val_idx].copy()
    value_mean = X_other_fold_train['Value'].mean()
    X_other_fold_train['Value'] = X_other_fold_train['Value'].fillna(value_mean)
    X_other_fold_val['Value'] = X_other_fold_val['Value'].fillna(value_mean)
    scaler = StandardScaler()
    X_other_fold_train[numerical_cols] = scaler.fit_transform(X_other_fold_train[numerical_cols])
    X_other_fold_val[numerical_cols] = scaler.transform(X_other_fold_val[numerical_cols])
    X_fold_train = np.hstack([X_text_fold_train, X_other_fold_train.astype(float).values])
    X_fold_val = np.hstack([X_text_fold_val, X_other_fold_val.astype(float).values])
    model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', n_estimators=2000, learning_rate=0.02, num_leaves=40, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, device='gpu')
    model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds_log = model.predict(X_fold_val); preds_original = np.expm1(preds_log); y_val_original = np.expm1(y_fold_val)
    metrics_text['r2'].append(r2_score(y_val_original, preds_original)); metrics_text['smape'].append(smape(y_val_original, preds_original))
print(f"✅ Text Model CV complete. Average R²: {np.mean(metrics_text['r2']):.3f}, Average SMAPE: {np.mean(metrics_text['smape']):.2f}%")

print("   Retraining final text model on all data...")
final_scaler_text = StandardScaler()
X_other_train_text_final = X_other_features_df_train.copy()
final_text_value_mean = X_other_train_text_final['Value'].mean()
X_other_train_text_final['Value'] = X_other_train_text_final['Value'].fillna(final_text_value_mean)
X_other_train_text_final[numerical_cols] = final_scaler_text.fit_transform(X_other_train_text_final[numerical_cols])
X_train_final_text = np.hstack([X_text_train, X_other_train_text_final.astype(float).values])
final_text_model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', n_estimators=2000, learning_rate=0.02, num_leaves=40, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, device='gpu')
final_text_model.fit(X_train_final_text, y_train)
joblib.dump(final_text_model, 'text_model.pkl')
print(f"   ✅ Text model saved to: {os.path.abspath('text_model.pkl')}")

# ----------------------------------------------------------------------------------------------------
## --- Step 4: Train the Image-Based Expert Model ---
print("\n--- 🎨 Training Image-Based Expert ---")
metrics_image = {'smape': [], 'r2': []} 
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_image_features, y_train, groups=groups)):
    X_image_fold_train, X_image_fold_val = X_image_features[train_idx], X_image_features[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    X_other_fold_train = X_other_features_df_train.iloc[train_idx].copy()
    X_other_fold_val = X_other_features_df_train.iloc[val_idx].copy()
    value_mean = X_other_fold_train['Value'].mean()
    X_other_fold_train['Value'] = X_other_fold_train['Value'].fillna(value_mean)
    X_other_fold_val['Value'] = X_other_fold_val['Value'].fillna(value_mean)
    scaler = StandardScaler()
    X_other_fold_train[numerical_cols] = scaler.fit_transform(X_other_fold_train[numerical_cols])
    X_other_fold_val[numerical_cols] = scaler.transform(X_other_fold_val[numerical_cols])
    X_fold_train = np.hstack([X_image_fold_train, X_other_fold_train.astype(float).values])
    X_fold_val = np.hstack([X_image_fold_val, X_other_fold_val.astype(float).values])
    model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', n_estimators=2000, learning_rate=0.02, num_leaves=40, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, device='gpu')
    model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds_log = model.predict(X_fold_val); preds_original = np.expm1(preds_log); y_val_original = np.expm1(y_fold_val)
    metrics_image['r2'].append(r2_score(y_val_original, preds_original)); metrics_image['smape'].append(smape(y_val_original, preds_original))

print(f"✅ Image Model CV complete. Average R²: {np.mean(metrics_image['r2']):.3f}, Average SMAPE: {np.mean(metrics_image['smape']):.2f}%")
print("   Retraining final image model on all data...")
final_scaler_image = StandardScaler()
X_other_train_image_final = X_other_features_df_train.copy()
final_image_value_mean = X_other_train_image_final['Value'].mean()
X_other_train_image_final['Value'] = X_other_train_image_final['Value'].fillna(final_image_value_mean)
X_other_train_image_final[numerical_cols] = final_scaler_image.fit_transform(X_other_train_image_final[numerical_cols])
X_train_final_image = np.hstack([X_image_features, X_other_train_image_final.astype(float).values])
final_image_model = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', n_estimators=2000, learning_rate=0.02, num_leaves=40, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, device='gpu')
final_image_model.fit(X_train_final_image, y_train)
joblib.dump(final_image_model, 'image_model.pkl')
print(f"   ✅ Image model saved to: {os.path.abspath('image_model.pkl')}")

# ----------------------------------------------------------------------------------------------------
## --- Step 5: Process Test Set Images On-the-Fly ---
print("\n--- 🖼️ Processing Test Set Images from URLs ---")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output; x = GlobalAveragePooling2D()(x)
image_feature_extractor = Model(inputs=base_model.input, outputs=x)
def download_and_preprocess_image(url):
    try:
        response = requests.get(url, timeout=10); response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB').resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        return preprocess_input(img_array)
    except Exception: return None
def get_test_image_features(df):
    all_features = []
    for url in tqdm(df['image_link'], desc="Extracting Test Image Features"):
        preprocessed_img = download_and_preprocess_image(url)
        if preprocessed_img is not None:
            features = image_feature_extractor.predict(preprocessed_img, verbose=0)
            all_features.append(features[0])
        else: all_features.append(np.zeros(2048))
    return np.array(all_features)
X_image_features_test = get_test_image_features(df_test)
print(f"✅ Test image features created with shape: {X_image_features_test.shape}")

# ----------------------------------------------------------------------------------------------------
## --- Step 6: Generate Final Submissions ---
print("\n--- 📄 Generating Final Submission Files ---")
print("   Generating text embeddings for test set...")
X_text_test = embedder.encode(df_test['original_content'].tolist(), show_progress_bar=True)
df_brands_test = pd.get_dummies(df_test['brand'], prefix='brand', dummy_na=True).reindex(columns=df_brands_train.columns, fill_value=0)
df_units_test = pd.get_dummies(df_test['Unit'], prefix='unit', dummy_na=True).reindex(columns=df_units_train.columns, fill_value=0)

# --- A. Prepare Test Data for TEXT MODEL ---
X_other_test_text = df_test[other_feature_cols].join(df_brands_test).join(df_units_test).copy()
X_other_test_text['Value'] = X_other_test_text['Value'].fillna(final_text_value_mean)
X_other_test_text[numerical_cols] = final_scaler_text.transform(X_other_test_text[numerical_cols])
X_test_final_text = np.hstack([X_text_test, X_other_test_text.astype(float).values])

# --- B. Prepare Test Data for IMAGE MODEL ---
X_other_test_image = df_test[other_feature_cols].join(df_brands_test).join(df_units_test).copy()
X_other_test_image['Value'] = X_other_test_image['Value'].fillna(final_image_value_mean)
X_other_test_image[numerical_cols] = final_scaler_image.transform(X_other_test_image[numerical_cols])
X_test_final_image = np.hstack([X_image_features_test, X_other_test_image.astype(float).values])

# --- C. Predict and save each submission ---
print("\n   Predicting and saving submissions...")
# 1. Text-Only Submission
text_preds_log = final_text_model.predict(X_test_final_text)
text_preds = np.expm1(text_preds_log)
submission_text = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': text_preds})
submission_text.to_csv('submission_text_only.csv', index=False)
print(f"   ✅ Text-only submission saved to: {os.path.abspath('submission_text_only.csv')}")

# 2. Image-Only Submission
image_preds_log = final_image_model.predict(X_test_final_image)
image_preds = np.expm1(image_preds_log)
submission_image = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': image_preds})
submission_image.to_csv('submission_image_only.csv', index=False)
print(f"   ✅ Image-only submission saved to: {os.path.abspath('submission_image_only.csv')}")

# 3. Ensemble (Simple Averaging) Submission
ensemble_preds = (text_preds + image_preds) / 2
submission_ensemble = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': ensemble_preds})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print(f"   ✅ Ensemble submission saved to: {os.path.abspath('submission_ensemble.csv')}")

print("\n--- All submissions created successfully! ---")