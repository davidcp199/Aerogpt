import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib


# ============================================================
# 1. MODELO GRU IDENTICO AL ENTRENADO
# ============================================================
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


# ============================================================
# 2. FEATURE COLUMNS
# ============================================================
FEATURE_COLS = ['setting_1','setting_2','setting_3'] + [f's_{i}' for i in range(1,22)]
WINDOW_SIZE = 30


# ============================================================
# 3. CARGA DE MODELO + SCALER
# ============================================================
def load_model(base_path, fd_code="FD001"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(base_path, "models", f"best_model_{fd_code}.pth")
    scaler_path = os.path.join(base_path, "models", f"scaler_{fd_code}.pkl")

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


# ============================================================
# 4. SI EL USUARIO SOLO DA 1 CICLO → GENERAR HISTORIA SINTÉTICA
# ============================================================
def generate_synthetic_history(row, length=30):
    """
    Genera una historia temporal suave y consistente con el ciclo actual.
    Útil si el usuario solo entrega 1 fila.
    """
    base = row.values.flatten()
    seq = []

    for i in range(length):
        noise = np.random.normal(0, 0.01, size=len(base))
        decay = (length - i) / length
        synthetic = base * (0.98 + 0.02 * decay) + noise
        seq.append(synthetic)

    return pd.DataFrame(seq, columns=row.index)


# ============================================================
# 5. PREPROCESAMIENTO DEL INPUT
# ============================================================
def preprocess_user_data(df, scaler):
    df = df.copy()

    # aseguramos que tiene todas las columnas
    missing_cols = set(FEATURE_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    return df


# ============================================================
# 6. VENTANA PARA EL MODELO
# ============================================================
def make_window(df, window_size=30):
    seq = df[FEATURE_COLS].values

    if len(seq) < window_size:
        pad = np.zeros((window_size - len(seq), seq.shape[1]))
        seq = np.vstack([pad, seq])
    else:
        seq = seq[-window_size:]

    return np.expand_dims(seq, axis=0)


# ============================================================
# 7. PREDICCIÓN FINAL DE RUL
# ============================================================
def predict_RUL(user_df, base_path, fd="FD001"):
    model, scaler, device = load_model(base_path, fd)

    # Si solo hay 1 ciclo → generar historia sintética
    if len(user_df) == 1:
        user_df = generate_synthetic_history(user_df.iloc[0], length=WINDOW_SIZE)

    df_clean = preprocess_user_data(user_df, scaler)

    X = make_window(df_clean, WINDOW_SIZE)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()[0]

    y_pred = max(0, float(y_pred))

    return {
        "predicted_RUL": y_pred
    }


# ============================================================
# 8. EJEMPLO DE USO LOCAL
# ============================================================
if __name__ == "__main__":
    data = {
        "setting_1": [0.5],
        "setting_2": [0.1],
        "setting_3": [0.3],
        **{f"s_{i}": [1000.0] for i in range(1,22)}
    }
    df_user = pd.DataFrame(data)

    base_path = r"C:\Users\David\Documents\Master-Big-Data-Data-Sciencee-e-Inteligencia-Artificial\TFM\AeroGPT\data\CMAPSS"

    pred = predict_RUL(df_user, base_path, "FD004")
    print(pred)
