import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from utils.config_loader import load_all_configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_, paths_cfg, _ = load_all_configs(ROOT)
MODEL_DIR = paths_cfg["paths"]["cmapss_models"]


# 1. Modelo. GRU igual a entrenamiento
class GRUModel(nn.Module):
    def __init__(self, input_dim=24, hidden_dim1=256, hidden_dim2=128, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim1, num_layers=2, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, num_layers=2, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = out[:, -1, :]
        return self.linear(out)


# 2. FEATURES
FEATURE_COLS = ['setting_1','setting_2','setting_3'] + [f's_{i}' for i in range(1,22)]
WINDOW_SIZE = 30


# 3. CARGA DE MODELO, SCALER y CALIBRACIÓN

def load_model(base_path, fd_code="FD001"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(MODEL_DIR, f"best_model_{fd_code}.pth")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{fd_code}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")

    model = GRUModel(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler, device

def load_calibrator(fd_code="FD001"):
    """
    Carga calibrador específico por FD si existe.
    Si no, intenta cargar calibrador global.
    Si no hay ninguno, devuelve None.
    """
    calib_fd_path = os.path.join(MODEL_DIR, f"calib_{fd_code}.pkl")
    calib_global_path = os.path.join(MODEL_DIR, "calib_global.pkl")

    if os.path.exists(calib_fd_path):
        return joblib.load(calib_fd_path), "fd"
    elif os.path.exists(calib_global_path):
        return joblib.load(calib_global_path), "global"
    else:
        return None, None


# 4. EXPANSIÓN DE HISTORIAL HASTA 30 FILAS

def generate_synthetic_history(row, length=30):
    """
    Genera datos sintéticos suaves basados en la última fila disponible.
    """
    base = row.values.astype(float)
    seq = []

    for i in range(length):
        noise = np.random.normal(0, 0.015, size=len(base))

        # Suavizado progresivo
        factor = 1 - (i / (length * 50))
        synthetic = base * factor + noise

        seq.append(synthetic)

    return pd.DataFrame(seq, columns=row.index)


def expand_history_to_30(df):
    """
    Expande el dataframe del usuario hasta 30 filas.
    - Si tiene 1 fila → genera 30 totalmente sintéticas.
    - Si tiene entre 2 y 29 → añade filas sintéticas hasta llegar a 30.
    - Si tiene ≥ 30 → usa las últimas 30.
    """
    n = len(df)

    # Caso 1: solo 1 fila
    if n == 1:
        return generate_synthetic_history(df.iloc[0], length=WINDOW_SIZE)

    # Caso 2: entre 2 y 29
    if n < WINDOW_SIZE:
        needed = WINDOW_SIZE - n
        last_row = df.iloc[-1]
        synthetic = generate_synthetic_history(last_row, length=needed)
        return pd.concat([df, synthetic], ignore_index=True)

    # Caso 3: 30 o más devolver todas las filas
    return df


# 5. PREPROCESAMIENTO DEL INPUT

def preprocess_user_data(df, scaler):
    df = df.copy()

    # aseguramos que tiene todas las columnas
    missing_cols = set(FEATURE_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    return df



# 6. VENTANA PARA EL MODELO

def make_window(df, window_size=30):
    seq = df[FEATURE_COLS].values

    if len(seq) < window_size:
        pad = np.zeros((window_size - len(seq), seq.shape[1]))
        seq = np.vstack([pad, seq])
    else:
        seq = seq[-window_size:]

    return np.expand_dims(seq, axis=0)



# 7. PREDICCIÓN FINAL DE RUL 

def predict_RUL(user_df, base_path, fd="FD001"):
    model, scaler, device = load_model(base_path, fd)

    # Expandir la historia
    user_df = expand_history_to_30(user_df)

    df_clean = preprocess_user_data(user_df, scaler)

    X = make_window(df_clean, WINDOW_SIZE)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Predicción base
    with torch.no_grad():
        y_pred_raw = model(X_tensor).cpu().numpy().flatten()[0]

    y_pred_raw = float(y_pred_raw)

    # --- CALIBRACIÓN ---
    calibrator, calib_type = load_calibrator(fd)

    if calibrator is not None:
        y_pred_cal = calibrator.predict(
            np.array([[y_pred_raw]])
        )[0]
    else:
        y_pred_cal = y_pred_raw

    # Seguridad: RUL no negativo
    y_pred_cal = max(0.0, float(y_pred_cal))

    return {
        "predicted_RUL": y_pred_cal,
        "predicted_RUL_raw": y_pred_raw,
        "calibration": calib_type
    }

