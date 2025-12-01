import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
from torch import nn
from torch.autograd import Variable

# ---------------------------
# 0. Configuración de rutas
# ---------------------------
#BASE_PATH = "/content/drive/MyDrive/CMAPSS_GRU/"
BASE_PATH = r"C:\Users\David\Documents\Master-Big-Data-Data-Sciencee-e-Inteligencia-Artificial\TFM\AeroGPT\data\CMAPSS"
RAW_PATH = os.path.join(BASE_PATH, "raw_5")
MODEL_PATH = os.path.join(BASE_PATH, "models_5")
FIG_PATH = os.path.join(BASE_PATH, "figures_5")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

FD_LIST = ["FD001", "FD002", "FD003", "FD004"]
WINDOW_SIZE = 30

# ---------------------------
# 1. Funciones
# ---------------------------
def load_fd_dataset(fd):
    col_names = ['unit_nr','time_cycles','setting_1','setting_2','setting_3'] + [f's_{i}' for i in range(1,22)]
    train_file = os.path.join(RAW_PATH, f"train_{fd}.txt")
    test_file  = os.path.join(RAW_PATH, f"test_{fd}.txt")
    rul_file   = os.path.join(RAW_PATH, f"RUL_{fd}.txt")

    train_df = pd.read_csv(train_file, sep='\s+', header=None, names=col_names)
    test_df  = pd.read_csv(test_file,  sep='\s+', header=None, names=col_names)
    rul_df   = pd.read_csv(rul_file,  sep='\s+', header=None, names=['RUL'])

    return train_df, test_df, rul_df

def add_rul(train_df):
    max_cycles = train_df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr','max_cycle']
    df = train_df.merge(max_cycles, on='unit_nr', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    return df.drop(columns=['max_cycle'])

def scale_features(train_df, test_df, feature_cols):
    # Llenar NaN con media del train
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    test_df[feature_cols]  = test_df[feature_cols].fillna(train_df[feature_cols].mean())
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler

def create_sequences(df, feature_cols, target_col, window_size=30):
    X, y = [], []
    for unit in df['unit_nr'].unique():
        unit_df = df[df['unit_nr']==unit].sort_values(by='time_cycles')
        features = unit_df[feature_cols].values
        targets = unit_df[target_col].values
        if len(unit_df) >= window_size:
            for i in range(len(unit_df) - window_size + 1):
                seq = features[i:i+window_size]
                if np.isnan(seq).any():
                    continue
                X.append(seq)
                y.append(targets[i+window_size-1])
    return np.array(X), np.array(y)

def create_last_window_test(test_df, rul_df, feature_cols, window_size=30):
    X_test, y_test = [], []
    for i, unit in enumerate(test_df['unit_nr'].unique()):
        unit_df = test_df[test_df['unit_nr']==unit].sort_values(by='time_cycles')
        seq = unit_df[feature_cols].values
        # Ajuste si la secuencia es menor que window_size
        if len(seq) < window_size:
            padding = np.zeros((window_size - len(seq), seq.shape[1]))
            seq = np.vstack([padding, seq])
        else:
            seq = seq[-window_size:]
        X_test.append(seq)
        y_test.append(rul_df.iloc[i,0])
    return np.array(X_test), np.array(y_test)

# ---------------------------
# 2. Modelo GRU Mejorado
# ---------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim1, num_layers=2, dropout=0.3, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, num_layers=2, dropout=0.3, batch_first=True)

        self.linear = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        out,_ = self.gru1(x)
        out,_ = self.gru2(out)
        out = out[:,-1,:]
        out = self.linear(out)
        return out

# ---------------------------
# 3. Entrenamiento
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_COLS = ['setting_1','setting_2','setting_3'] + [f's_{i}' for i in range(1,22)]

for fd in FD_LIST:
    print(f"\nProcesando {fd}...")
    train_df, test_df, rul_df = load_fd_dataset(fd)
    train_df = add_rul(train_df)

    # Escalar
    train_df, test_df, scaler = scale_features(train_df, test_df, FEATURE_COLS)

    # Secuencias
    X, y = create_sequences(train_df, FEATURE_COLS, 'RUL', WINDOW_SIZE)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X)*0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1).to(device)

    print(f"{fd} -> Train: {X_train.shape}, Val: {X_val.shape}")

    # Modelo
    model = GRUModel(len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = np.inf
    patience = 30
    wait = 0
    n_epochs = 500
    batch_size = 32

    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        # Training
        model.train()
        perm = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0
        for i in range(0, X_train_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_train_tensor[idx]
            batch_y = y_train_tensor[idx]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        epoch_loss /= len(X_train_tensor)
        train_losses.append(epoch_loss)

        # Validación
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
            val_losses.append(val_loss)
            val_mae = mean_absolute_error(y_val_tensor.cpu(), val_pred.cpu())
            val_rmse = np.sqrt(val_loss)

        # Scheduler
        scheduler.step(val_loss)

        # Print cada 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} -> Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'best_model_{fd}.pth'))
        else:
            wait += 1
            if wait >= patience:
                print("⏹ Early stopping")
                break

    # Guardar scaler
    joblib.dump(scaler, os.path.join(MODEL_PATH, f'scaler_{fd}.pkl'))

    # ---------------------------
    # 4. Test: última ventana
    # ---------------------------
    X_test, y_test = create_last_window_test(test_df, rul_df, FEATURE_COLS, WINDOW_SIZE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1).to(device)

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'best_model_{fd}.pth')))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)

    mse = criterion(y_pred, y_test_tensor).item()
    mae = mean_absolute_error(y_test_tensor.cpu(), y_pred.cpu())
    rmse = np.sqrt(mse)
    print(f"{fd} -> Test MSE={mse:.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

    # ---------------------------
    # 5. Gráficas
    # ---------------------------
    # Train/Val loss
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{fd} - Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(FIG_PATH, f'TrainValLoss_{fd}.png'))
    plt.show()

    # RUL predicho vs real
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='RUL_real')
    plt.plot(y_pred.cpu().numpy(), label='RUL_predicho')
    plt.title(f"{fd} - RUL: Real vs Predicho")
    plt.xlabel("Unidad")
    plt.ylabel("RUL")
    plt.legend()
    plt.savefig(os.path.join(FIG_PATH, f'RUL_pred_{fd}.png'))
    plt.show()