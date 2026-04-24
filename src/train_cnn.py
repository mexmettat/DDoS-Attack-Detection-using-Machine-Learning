from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import os
import glob
import time
import json
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score, 
    precision_score, accuracy_score, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROCESSED_DATA_DIR = r"data\processed"
MODEL_SAVE_DIR = r"models"
METRICS_DIR = r"metrics"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

TRAIN_FILES_PATTERN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TEST_FILES_PATTERN = ["DNS", "LDAP", "MSSQL", "NTP", "NetBIOS", "Portmap", "SNMP", "Syn", "TFTP", "UDP", "UDPLag"]

def load_grouped_data(patterns, sample_size=None):
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(PROCESSED_DATA_DIR, f"*{pattern}*_cleaned.csv")))
    
    dfs = []
    for f in all_files:
        print(f"Loading: {os.path.basename(f)}")
        df = pd.read_csv(f)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def calculate_detailed_metrics_cnn(model, X_test, y_test, stage_name):
    print(f"\n--- CNN ({stage_name}) Evaluation ---")
    
    start_time = time.time()
    y_probs = model.predict(X_test).flatten()
    end_time = time.time()
    inference_time = (end_time - start_time) / len(X_test)
    
    # THRESHOLD = 0.15 if stage_name == "Test2019" else 0.50 
    # y_pred = (y_probs >= THRESHOLD).astype(int)

    # Fonksiyonun içindeki o kısmı doğrudan şöyle sadeleştirebilirsin:
    y_pred = (y_probs >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    metrics = {
        "model": "CNN",
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
    json_path = os.path.join(METRICS_DIR, f"cnn_{stage_name.lower()}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix Plot
    plt.figure(figsize=(6,4))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Conf Matrix: CNN ({stage_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(METRICS_DIR, f"cnn_{stage_name.lower()}_conf.png"))
    plt.close()

    return metrics

def main():
   # --- STEP 1: KARMA EĞİTİM (MIXED DATASET) HAZIRLIĞI ---
    print("Step 1: Loading Training Data (Mixed 2017 & 2019)...")
    
    # 1.A: 2017 Verisini Yükle
    train_data_2017 = load_grouped_data(TRAIN_FILES_PATTERN, sample_size=40000) 
    
    # 1.B: 2019 Verisini Yükle (Eğitim ve Test için toplam 15.000 satır çekiyoruz)
    data_2019 = load_grouped_data(TEST_FILES_PATTERN, sample_size=15000)
    
    # 2019 Kolon İsimlerini 2017'ye Çevir
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
    
    # 1.C: 2019 Verisini Eğitim (%33) ve Final Testi (%67) olarak böl (Veri sızıntısını önlemek için!)
    # 5.000 satır eğitime gidecek, 10.000 satır yepyeni test için ayrılacak.
    train_data_2019, final_test_2019 = train_test_split(data_2019, test_size=10000, random_state=42, stratify=data_2019['Label'])
    
    # 1.D: Ana Eğitim Setini Oluştur (2017 + 2019'un bir kısmı)
    train_data = pd.concat([train_data_2017, train_data_2019], ignore_index=True)
    
    # Modelin kolonları tanıması için kaydet
    X = train_data.drop(columns=['Label'])
    y = train_data['Label'].values
    train_columns = X.columns
    joblib.dump(train_columns, os.path.join(MODEL_SAVE_DIR, "train_columns.pkl"))
    # --------------------------------------------------------

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Scaling (Hoca's advice)
    print("Step 2: Scaling features for CNN (Fit ONLY on X_train)...")
    scaler = MinMaxScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler_cnn.pkl"))

    # 4. Reshape for CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)

    # 5. Build and Train
    print("\nStep 3: Building and Training 1D-CNN (with Early Stopping)...")
    input_shape = (X_train_reshaped.shape[1], 1)
    model = build_cnn(input_shape)

    counts = np.bincount(y_train)
    class_weights = {0: (1.0/counts[0])*(len(y_train)/2.0), 1: (1.0/counts[1])*(len(y_train)/2.0)}

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train_reshaped, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val_reshaped, y_val),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Save
    model.save(os.path.join(MODEL_SAVE_DIR, "cnn_model.h5"))
    with open(os.path.join(METRICS_DIR, "cnn_history.json"), "w") as f:
        json.dump(history.history, f)

    # Plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.title('CNN Loss'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.title('CNN Accuracy'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(METRICS_DIR, "cnn_training_plots.png")); plt.close()

    # Evaluation
    calculate_detailed_metrics_cnn(model, X_val_reshaped, y_val, "Val")

    # 6. Final Test with CICDDoS2019 (Alignment & Scaling)
    print("\nStep 4: Final Test with CICDDoS2019 (using pre-split data to avoid leakage)...")
    test_data = final_test_2019
    
    if not test_data.empty:
        X_final_test = test_data.drop(columns=['Label'])
        y_final_test = test_data['Label'].values
        
        # SCALING (Already aligned in Step 1)
        X_final_test_scaled = scaler.transform(X_final_test)
        
        # RESHAPE
        X_final_test_reshaped = X_final_test_scaled.reshape(X_final_test_scaled.shape[0], X_final_test_scaled.shape[1], 1)
        
        calculate_detailed_metrics_cnn(model, X_final_test_reshaped, y_final_test, "Test2019")

    print("\n[SUCCESS] CNN restored and fixed.")

if __name__ == "__main__":
    main()
