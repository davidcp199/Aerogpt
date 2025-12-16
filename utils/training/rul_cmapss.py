import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
from torch import nn

# ---------------------------
# 0. Configuración general
# ---------------------------
BASE_PATH = r"C:\Users\David\Documents\Master-Big-Data-Data-Sciencee-e-Inteligencia-Artificial\TFM\AeroGPT\data\CMAPSS"
RAW_PATH   = os.path.join(BASE_PATH, "raw")
MODEL_PATH = os.path.join(BASE_PATH, "models_new")
FIG_PATH   = os.path.join(BASE_PATH, "figures_new")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

FD_LIST = ["FD001", "FD002", "FD003", "FD004"]

WINDOW_SIZE = 30
RUL_CAP = 125
BATCH_SIZE = 16
N_EPOCHS = 500
PATIENCE = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Selección de features
# ---------------------------
SETTING_COLS = ['setting_1','setting_2','setting_3']
SENSOR_COLS = [
    's_2','s_3','s_4','s_7','s_8','s_9',
    's_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21'
]

# ---------------------------
# 2. Carga de datos
# ---------------------------
def load_fd_dataset(fd):
    col_names = ['unit_nr','time_cycles'] + SETTING_COLS + [f's_{i}' for i in range(1,22)]
    train_df = pd.read_csv(os.path.join(RAW_PATH, f"train_{fd}.txt"),
                           sep='\s+', header=None, names=col_names)
    test_df  = pd.read_csv(os.path.join(RAW_PATH, f"test_{fd}.txt"),
                           sep='\s+', header=None, names=col_names)
    rul_df   = pd.read_csv(os.path.join(RAW_PATH, f"RUL_{fd}.txt"),
                           sep='\s+', header=None, names=['RUL'])
    return train_df, test_df, rul_df

# ---------------------------
# 3. Feature engineering
# ---------------------------
def add_rul(df):
    max_cycles = df.groupby('unit_nr')['time_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_cycles']
    df['RUL'] = df['RUL'].clip(upper=RUL_CAP)
    df['RUL_norm'] = df['RUL'] / RUL_CAP
    return df

def add_time_features(df):
    max_cycles = df.groupby('unit_nr')['time_cycles'].transform('max')
    df['time_norm'] = df['time_cycles'] / max_cycles
    return df

def add_derivatives(df, sensor_cols):
    for s in sensor_cols:
        df[f'd_{s}'] = df.groupby('unit_nr')[s].diff().fillna(0)
    return df

# ---------------------------
# 4. Escalado
# ---------------------------
def scale_features(train_df, test_df, feature_cols):
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler

# ---------------------------
# 5. Split estricto por unidad
# ---------------------------
def split_by_unit(df, val_ratio=0.2):
    units = df['unit_nr'].unique()
    np.random.shuffle(units)
    split = int(len(units)*(1-val_ratio))
    return units[:split], units[split:]

# ---------------------------
# 6. Creación de secuencias
# ---------------------------
def create_sequences(df, feature_cols, target_col):
    X, y = [], []
    for unit in df['unit_nr'].unique():
        u = df[df.unit_nr == unit].sort_values('time_cycles')
        for i in range(len(u) - WINDOW_SIZE + 1):
            X.append(u[feature_cols].iloc[i:i+WINDOW_SIZE].values)
            y.append(u[target_col].iloc[i+WINDOW_SIZE-1])
    return np.array(X), np.array(y)

def create_last_window_test(test_df, rul_df, feature_cols):
    X_test, y_test = [], []
    for i, unit in enumerate(test_df['unit_nr'].unique()):
        u = test_df[test_df.unit_nr == unit].sort_values('time_cycles')
        seq = u[feature_cols].values
        if len(seq) < WINDOW_SIZE:
            pad = np.repeat(seq[:1], WINDOW_SIZE - len(seq), axis=0)
            seq = np.vstack([pad, seq])
        else:
            seq = seq[-WINDOW_SIZE:]
        X_test.append(seq)
        y_test.append(min(rul_df.iloc[i,0], RUL_CAP) / RUL_CAP)
    return np.array(X_test), np.array(y_test)

# ---------------------------
# 7. Modelo GRU final
# ---------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out1,_ = self.gru1(x)
        out2,_ = self.gru2(out1)
        out = out1[:,-1,:] + out2[:,-1,:]
        return self.head(out)

# ---------------------------
# 8. Pérdida Huber ponderada
# ---------------------------
class WeightedHuberLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, pred, target):
        weights = torch.exp(-target)
        return torch.mean(weights * self.huber(pred, target))

# ---------------------------
# 9. Curriculum learning
# ---------------------------
def curriculum_mask(y, epoch):
    y = y.view(-1)  # [batch]
    if epoch < 50:
        return y < 0.4
    return torch.ones_like(y, dtype=torch.bool)


# ===========================
# 10. Entrenamiento por FD
# ===========================
for fd in FD_LIST:
    print(f"\nProcesando {fd}")

    train_df, test_df, rul_df = load_fd_dataset(fd)
    train_df = add_rul(train_df)
    train_df = add_time_features(train_df)
    train_df = add_derivatives(train_df, SENSOR_COLS)

    test_df  = add_time_features(test_df)
    test_df  = add_derivatives(test_df, SENSOR_COLS)

    FEATURE_COLS = SETTING_COLS + SENSOR_COLS + \
                   ['time_norm'] + [f'd_{s}' for s in SENSOR_COLS]

    train_df, test_df, scaler = scale_features(train_df, test_df, FEATURE_COLS)

    train_units, val_units = split_by_unit(train_df)

    X_train, y_train = create_sequences(
        train_df[train_df.unit_nr.isin(train_units)],
        FEATURE_COLS, 'RUL_norm'
    )

    X_val, y_val = create_sequences(
        train_df[train_df.unit_nr.isin(val_units)],
        FEATURE_COLS, 'RUL_norm'
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
    X_val   = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val   = torch.tensor(y_val, dtype=torch.float32).view(-1,1).to(device)

    model = GRUModel(X_train.shape[2]).to(device)
    criterion = WeightedHuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    best_val = np.inf
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0

        for i in range(0, X_train.size(0), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]

            batch_x = X_train[idx]
            batch_y = y_train[idx]

            mask = curriculum_mask(batch_y, epoch)
            if mask.sum() == 0:
                continue

            optimizer.zero_grad()
            preds = model(batch_x[mask])
            loss = criterion(preds, batch_y[mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()


        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"best_model_{fd}.pth"))
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    joblib.dump(scaler, os.path.join(MODEL_PATH, f"scaler_{fd}.pkl"))

def nasa_score(y_true, y_pred):
    score = 0.0
    for yt, yp in zip(y_true, y_pred):
        e = yp - yt
        if e < 0:
            score += np.exp(-e / 13) - 1
        else:
            score += np.exp(e / 10) - 1
    return score

# ---------------------------
# 11. Test final (oficial CMAPSS)
# ---------------------------
X_test, y_test = create_last_window_test(test_df, rul_df, FEATURE_COLS)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"best_model_{fd}.pth")))
model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy().flatten()

# Desnormalización
y_pred_rul = y_pred * RUL_CAP
y_true_rul = y_test * RUL_CAP

# Métricas estándar
mae = mean_absolute_error(y_true_rul, y_pred_rul)
rmse = np.sqrt(mean_squared_error(y_true_rul, y_pred_rul))

# Métrica NASA
score = nasa_score(y_true_rul, y_pred_rul)

print(f"{fd} -> MAE={mae:.2f}, RMSE={rmse:.2f}, NASA Score={score:.1f}")




