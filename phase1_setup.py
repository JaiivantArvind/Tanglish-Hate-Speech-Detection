"""
Phase 1: Environment Setup, Data Download, and Initial Processing
==================================================================
Description:
    Installs required packages, checks GPU availability, clones the DravidianCodeMix
    dataset repository, extracts the Tamil subfolder files, loads train/dev/test CSVs,
    and maps string labels to integer IDs.

Dependencies:
    - Internet connection for git clone and pip installations.

Outputs:
    - phase1_data.joblib: Contains train_df, dev_df, test_df, and label2id dictionary.
"""

import os
import sys
import zipfile
import subprocess
import pandas as pd
import joblib

# ==============================================================================
# Step 1: Package Installation Helper
# ==============================================================================
def install_packages():
    required_packages = [
        "transformers", "datasets", "torch", "scikit-learn",
        "pandas", "numpy", "matplotlib", "seaborn", "accelerate", "tqdm", "joblib"
    ]
    print("Checking and installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + required_packages)
    print("✅ All required packages installed successfully.")

# ==============================================================================
# Step 2: GPU Verification
# ==============================================================================
def check_gpu():
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Available: {device_name}")
    else:
        print("⚠️ WARNING: GPU not available. Running on CPU.")

# ==============================================================================
# Step 3: Dataset Cloning & Extraction
# ==============================================================================
def download_dataset():
    repo_dir = "DravidianCodeMix-Dataset"
    if not os.path.exists(repo_dir):
        print("Cloning DravidianCodeMix dataset repository...")
        subprocess.check_call(["git", "clone", "https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset.git"])
        print("✅ Repository cloned successfully.")

    zip_path = os.path.join(repo_dir, "DravidianCodeMix-2020.zip")
    extract_dir = "DravidianCodeMix-Extracted"

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Extracted dataset to '{extract_dir}'")
    else:
        raise FileNotFoundError(f"Zip file not found at {zip_path}")

    # List files in Tamil dataset folder
    dataset_folder = os.path.join(extract_dir, "DravidianCodeMix")
    print("\n--- Tamil Dataset Files Found ---")
    if os.path.exists(dataset_folder):
        for f in os.listdir(dataset_folder):
            if "tamil" in f.lower():
                print(f"  - {f}")
    return extract_dir

# ==============================================================================
# Step 4 & 5: Load Data & Map Labels
# ==============================================================================
def load_and_clean_data(extract_dir):
    train_path = os.path.join(extract_dir, "DravidianCodeMix", "tamil_offensive_full_train.csv")
    dev_path   = os.path.join(extract_dir, "DravidianCodeMix", "tamil_offensive_full_dev.csv")
    test_path  = os.path.join(extract_dir, "DravidianCodeMix", "tamil_offensive_full_test.csv")

    def load_split(file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')
        df = df.rename(columns={0: 'text', 1: 'category'})
        return df[['text', 'category']]

    train_df = load_split(train_path)
    dev_df   = load_split(dev_path)
    test_df  = load_split(test_path)

    print(f"\nRaw Dataset Shapes -> Train: {train_df.shape}, Dev: {dev_df.shape}, Test: {test_df.shape}")
    print("\n--- First 5 rows of train_df ---")
    print(train_df.head())

    print("\n--- Label Distribution in Raw train_df ---")
    print(train_df['category'].value_counts(dropna=False))

    # Exact label mapping dictionary
    label2id = {
        "Not_offensive": 0,
        "Offensive_Untargetede": 1,
        "Offensive_Targeted_Insult_Individual": 2,
        "Offensive_Targeted_Insult_Group": 3,
        "Offensive_Targeted_Insult_Other": 4,
    }

    # Map labels and drop invalid/unmapped rows
    for name, df in [("train_df", train_df), ("dev_df", dev_df), ("test_df", test_df)]:
        df['label'] = df['category'].map(label2id)
        initial_len = len(df)
        df.dropna(subset=['label', 'text'], inplace=True)
        df['label'] = df['label'].astype(int)
        print(f"✅ {name}: {len(df)} samples remaining after dropping {initial_len - len(df)} unmapped/missing rows.")
        print(f"   Unique integer labels: {sorted(df['label'].unique())}")

    # Save to disk for Phase 2
    output_file = "phase1_data.joblib"
    joblib.dump({
        "train_df": train_df,
        "dev_df": dev_df,
        "test_df": test_df,
        "label2id": label2id
    }, output_file)
    print(f"\n✅ Phase 1 completed successfully! Saved data to '{output_file}'.")

if __name__ == "__main__":
    install_packages()
    check_gpu()
    extract_dir = download_dataset()
    load_and_clean_data(extract_dir)
