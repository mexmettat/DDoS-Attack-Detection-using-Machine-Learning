import pandas as pd
import numpy as np
import os
import glob
import joblib
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score, 
    precision_score, accuracy_score, roc_auc_score, average_precision_score
)

# Configuration
PROCESSED_DATA_DIR = r"data\processed"
MODEL_SAVE_DIR = r"models"
METRICS_DIR = r"metrics"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Dataset groups
TRAIN_FILES_PATTERN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TEST_FILES_PATTERN = ["DNS", "LDAP", "MSSQL", "NTP", "NetBIOS", "Portmap", "SNMP", "Syn", "TFTP", "UDP", "UDPLag"]

def load_grouped_data(patterns, sample_size=None):
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(PROCESSED_DATA_DIR, f"*{pattern}*_cleaned.csv")))
    
    if not all_files:
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        print(f"Loading: {os.path.basename(f)}")
        df = pd.read_csv(f)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def calculate_detailed_metrics(model, X_test, y_test, model_prefix, stage_name):
    print(f"\n--- {model_prefix} ({stage_name}) Evaluation ---")
    
   # Inference Time
    start_time = time.time()
    y_probs = model.predict_proba(X_test)[:, 1] 
    y_pred = (y_probs >= 0.50).astype(int)
    
    end_time = time.time()
    inference_time = (end_time - start_time) / len(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    metrics = {
        "model": model_prefix,
        "stage": stage_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "inference_time_per_sample": float(inference_time)
    }

    # Save JSON metrics
    json_path = os.path.join(METRICS_DIR, f"{model_prefix.lower()}_{stage_name.lower()}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix Plot
    plt.figure(figsize=(6,4))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Conf Matrix: {model_prefix} ({stage_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(METRICS_DIR, f"{model_prefix.lower()}_{stage_name.lower()}_conf.png"))
    plt.close()

    print(f"Metrics saved to: {json_path}")
    return metrics

def main():
    
    print("Step 1: Loading Training Data (Mixed 2017 & 2019)...")
    
    e
    train_data_2017 = load_grouped_data(TRAIN_FILES_PATTERN, sample_size=40000) 
    
    
    data_2019 = load_grouped_data(TEST_FILES_PATTERN, sample_size=15000)
    
    
    column_mapping = {
        'Fwd Packets Length Total': 'Total Length of Fwd Packets',
        'Bwd Packets Length Total': 'Total Length of Bwd Packets',
        'Packet Length Min': 'Min Packet Length',
        'Packet Length Max': 'Max Packet Length',
        'Avg Packet Size': 'Average Packet Size',
        'Init Fwd Win Bytes': 'Init_Win_bytes_forward',
        'Init Bwd Win Bytes': 'Init_Win_bytes_backward',
        'Fwd Act Data Packets': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward'
    }
    data_2019.rename(columns=column_mapping, inplace=True)
    data_2019 = data_2019.reindex(columns=train_data_2017.columns, fill_value=0)
    
    
    train_data_2019, final_test_2019 = train_test_split(data_2019, test_size=10000, random_state=42, stratify=data_2019['Label'])
    
    
    train_data = pd.concat([train_data_2017, train_data_2019], ignore_index=True)
    
    
    X = train_data.drop(columns=['Label'])
    y = train_data['Label'].values
    train_columns = X.columns
    joblib.dump(train_columns, os.path.join(MODEL_SAVE_DIR, "train_columns.pkl"))

    # --------------------------------------------------------

    # 2. Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Scaling robust better for these models
    print("Step 2: Scaling features (Fitting only on X_train)...")
    scaler = RobustScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler for inference
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))

    # 4. Train Random Forest
    print("\nStep 3: Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(rf_model, os.path.join(MODEL_SAVE_DIR, "random_forest_model.pkl"))
    calculate_detailed_metrics(rf_model, X_val_scaled, y_val, "RF", "Val")

    # 5. Train XGBoost
    print("\nStep 4: Training XGBoost...")
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    joblib.dump(xgb_model, os.path.join(MODEL_SAVE_DIR, "xgboost_model.pkl"))
    calculate_detailed_metrics(xgb_model, X_val_scaled, y_val, "XGB", "Val")

   # 6. Final Test with CICDDoS2019
    print("\nStep 5: Final Test with unseen CICDDoS2019 data...")
    
    X_final_test = final_test_2019.drop(columns=['Label'])
    y_final_test = final_test_2019['Label'].values
    
    X_final_test_scaled = scaler.transform(X_final_test)
    
    calculate_detailed_metrics(rf_model, X_final_test_scaled, y_final_test, "RF", "Test2019")
    calculate_detailed_metrics(xgb_model, X_final_test_scaled, y_final_test, "XGB", "Test2019")
    
    print("\n[SUCCESS] Detailed metrics, models, and scaler saved.")

if __name__ == "__main__":
    main()
