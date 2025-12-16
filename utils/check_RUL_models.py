import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error,
                             explained_variance_score)
from scipy.stats import pearsonr

# Rutas (ajusta si es necesario)
BASE_MODELS = r"C:\Users\David\Documents\AeroGPT\models\cmapss"
RESULTS_DIR = r"C:\Users\David\Documents\AeroGPT\results\CMAPSS\Models_Validation"
RAW_PATH = r"C:\Users\David\Documents\AeroGPT\data\raw\CMAPSS"

os.makedirs(RESULTS_DIR, exist_ok=True)

FD_LIST = ["FD001","FD002","FD003","FD004"]
WINDOW_SIZE = 30
FEATURE_COLS = ['setting_1','setting_2','setting_3'] + [f's_{i}' for i in range(1,22)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reutiliza tus funciones / modelo ---
# Copia aquí las definiciones de load_fd_dataset, create_last_window_test y GRUModel
# (Pegar las implementaciones que ya tienes)
# Por brevedad asumo que están definidas en el mismo scope:
# load_fd_dataset(fd), create_last_window_test(test_df, rul_df, feature_cols, window_size), GRUModel(input_dim)

def load_fd_dataset(fd):
    col_names = ['unit_nr','time_cycles','setting_1','setting_2','setting_3'] + [f's_{i}' for i in range(1,22)]
    train_file = os.path.join(RAW_PATH, f"train_{fd}.txt")
    test_file  = os.path.join(RAW_PATH, f"test_{fd}.txt")
    rul_file   = os.path.join(RAW_PATH, f"RUL_{fd}.txt")

    train_df = pd.read_csv(train_file, sep='\s+', header=None, names=col_names)
    test_df  = pd.read_csv(test_file,  sep='\s+', header=None, names=col_names)
    rul_df   = pd.read_csv(rul_file,  sep='\s+', header=None, names=['RUL'])

    return train_df, test_df, rul_df

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

# ---------------------------
# Comprobación de presencia de archivos
# ---------------------------
missing = []
for fd in FD_LIST:
    mp = os.path.join(BASE_MODELS, f'best_model_{fd}.pth')
    sp = os.path.join(BASE_MODELS, f'scaler_{fd}.pkl')
    if not os.path.exists(mp):
        missing.append(mp)
    if not os.path.exists(sp):
        missing.append(sp)
if missing:
    raise FileNotFoundError("Faltan archivos: \n" + "\n".join(missing))

# Utilidades
def safe_array(x):
    return np.array(x).ravel()

def evaluate_fd(fd):
    # Cargar datos
    _, test_df, rul_df = load_fd_dataset(fd)

    # Cargar scaler y escalar test
    scaler = joblib.load(os.path.join(BASE_MODELS, f'scaler_{fd}.pkl'))
    # Comprobación NaNs y aplicar transform
    if test_df[FEATURE_COLS].isna().any().any():
        test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(test_df[FEATURE_COLS].mean())
    try:
        test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
    except Exception as e:
        raise RuntimeError(f"Error aplicando scaler en {fd}: {e}")

    # Crear X_test, y_test
    X_test, y_test = create_last_window_test(test_df, rul_df, FEATURE_COLS, WINDOW_SIZE)
    if np.isnan(X_test).any():
        raise ValueError(f"NaNs en X_test para {fd} después del escalado.")
    if X_test.shape[0] == 0:
        raise ValueError(f"No hay ventanas para {fd} (X_test vacío).")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Cargar modelo
    model = GRUModel(len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(os.path.join(BASE_MODELS, f'best_model_{fd}.pth'), map_location=device))
    model.eval()

    # Predecir
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().ravel()
    y_true = safe_array(y_test)

    # Métricas básicas
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)

    # Errores y estadísticas
    errors = y_pred - y_true
    abs_err = np.abs(errors)
    rel_err_pct = np.abs(errors) / (np.abs(y_true) + 1e-8)
    bias = errors.mean()
    err_mean = errors.mean()
    err_std = errors.std()
    pct_over_10 = (rel_err_pct > 0.10).mean()
    pct_over_20 = (rel_err_pct > 0.20).mean()

    # Pearson corr (si var no nula)
    try:
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)
        else:
            pearson_r, pearson_p = np.nan, np.nan
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan

    # Guardar predicciones por unidad
    preds_df = pd.DataFrame({
        'unit': np.arange(1, len(y_true)+1),
        'RUL_true': y_true,
        'RUL_pred': y_pred,
        'error': errors,
        'abs_error': abs_err,
        'rel_error_pct': rel_err_pct
    })
    preds_file = os.path.join(RESULTS_DIR, f'preds_{fd}.xlsx')
    preds_df.to_excel(preds_file, index=False)

    # Resumen de métricas
    summary = {
        'FD': fd,
        'n_units': len(y_true),
        'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
        'mape': mape, 'medae': medae, 'explained_variance': ev,
        'bias_mean_error': bias, 'error_std': err_std,
        'pct_rel_err>10%': pct_over_10, 'pct_rel_err>20%': pct_over_20,
        'pearson_r': pearson_r, 'pearson_p': pearson_p
    }
    return summary

# Ejecutar evaluación para cada FD
results = []
for fd in FD_LIST:
    try:
        print("Evaluando", fd)
        res = evaluate_fd(fd)
        results.append(res)
    except Exception as e:
        print(f"Error evaluando {fd}: {e}")

metrics_df = pd.DataFrame(results)
metrics_csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
metrics_xlsx = os.path.join(RESULTS_DIR, "metrics_summary.xlsx")
metrics_df.to_csv(metrics_csv, index=False)
metrics_df.to_excel(metrics_xlsx, index=False)

print("Guardado en:", RESULTS_DIR)