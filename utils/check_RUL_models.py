# diagnose_recalibrate_lines.py
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn

# ---------------------------
# RUTAS
# ---------------------------
BASE_PATH = r"C:\Users\U68976\Documents\Mis documentos\GYM\TFM\AAA"
RAW_PATH = r"C:\Users\David\Documents\AeroGPT\data\raw\CMAPSS"
MODEL_PATH = r"C:\Users\David\Documents\AeroGPT\models\cmapss"
FIG_PATH = r"C:\Users\David\Documents\AeroGPT\results\CMAPSS\GRU"
os.makedirs(FIG_PATH, exist_ok=True)

WINDOW_SIZE = 30
FD_LIST = ["FD001","FD002","FD003","FD004"]
FEATURE_COLS = ['setting_1','setting_2','setting_3'] + [f's_{i}' for i in range(1,22)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# CARGAR MODELOS Y CONFS
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

def scale_using_saved(train_df, test_df, feature_cols, scaler):
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    test_df[feature_cols]  = test_df[feature_cols].fillna(train_df[feature_cols].mean())
    try:
        train_df[feature_cols] = scaler.transform(train_df[feature_cols])
        test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    except Exception as e:
        warnings.warn(f"Scaler unloadable: {e}. Refit scaler from train_df (risky).")
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        train_df[feature_cols] = sc.fit_transform(train_df[feature_cols])
        test_df[feature_cols]  = sc.transform(test_df[feature_cols])
    return train_df, test_df

def create_last_window_test(test_df, rul_df, feature_cols, window_size=30):
    X_test, y_test = [], []
    for i, unit in enumerate(sorted(test_df['unit_nr'].unique())):
        unit_df = test_df[test_df['unit_nr']==unit].sort_values(by='time_cycles')
        seq = unit_df[feature_cols].values
        if len(seq) < window_size:
            padding = np.zeros((window_size - len(seq), seq.shape[1]))
            seq = np.vstack([padding, seq])
        else:
            seq = seq[-window_size:]
        X_test.append(seq)
        y_test.append(rul_df.iloc[i,0])
    return np.array(X_test), np.array(y_test)

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

def nasa_score(y_true, y_pred):
    e = y_pred - y_true
    score = np.where(e < 0, np.exp(-e/13.0)-1.0, np.exp(e/10.0)-1.0)
    return np.sum(score)

# ---------------------------
# PLOT RUL
# ---------------------------
def plot_rul_lines(fd, y_true, y_pred, figpath, suffix=""):
    os.makedirs(figpath, exist_ok=True)
    idx = np.arange(len(y_true))
    plt.figure(figsize=(10,5))
    plt.plot(idx, y_true, '-o', markersize=4, label='RUL real', color='tab:blue')
    plt.plot(idx, y_pred, '-o', markersize=4, label='RUL predicho', color='tab:orange')
    plt.xlabel('Unidad (índice)')
    plt.ylabel('RUL')
    plt.title(f'{fd} - RUL real vs predicho{suffix}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'{fd}_rul_lines{suffix}.png'), bbox_inches='tight')
    plt.close()

# ---------------------------
# MAIN LOOP: generar diagnósticos
# ---------------------------
results = []
all_true = []
all_pred = []

for fd in FD_LIST:
    print("Processing", fd)
    train_df, test_df, rul_df = load_fd_dataset(fd)
    train_df = add_rul(train_df)

    scaler_path = os.path.join(MODEL_PATH, f'scaler_{fd}.pkl')
    model_path  = os.path.join(MODEL_PATH, f'best_model_{fd}.pth')
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print(f"Missing scaler/model for {fd}, skipping.")
        continue

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        warnings.warn(f"joblib.load scaler failed: {e}. Refit will be used.")
        scaler = None

    if scaler is not None:
        train_df_s, test_df_s = scale_using_saved(train_df.copy(), test_df.copy(), FEATURE_COLS, scaler)
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        train_df_s = train_df.copy()
        test_df_s = test_df.copy()
        train_df_s[FEATURE_COLS] = sc.fit_transform(train_df_s[FEATURE_COLS].fillna(train_df_s[FEATURE_COLS].mean()))
        test_df_s[FEATURE_COLS]  = sc.transform(test_df_s[FEATURE_COLS].fillna(train_df_s[FEATURE_COLS].mean()))

    X_test, y_true = create_last_window_test(test_df_s, rul_df, FEATURE_COLS, WINDOW_SIZE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = GRUModel(len(FEATURE_COLS)).to(device)
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    except Exception as e:
        warnings.warn(f"Could not load state_dict: {e}. Attempting torch.load direct.")
        model = torch.load(model_path, map_location=device)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy().flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ns = nasa_score(y_true, y_pred)
    print(f"{fd} BEFORE calib -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, NASA: {ns:.3f}")

    results.append({'fd':fd, 'n_samples':len(y_true), 'MAE_before':mae, 'RMSE_before':rmse, 'NASA_before':ns})
    all_true.append(y_true); all_pred.append(y_pred)

    # gráficas antes de calibrar
    plot_rul_lines(fd, y_true, y_pred, FIG_PATH, suffix="_before")

    # recalibración lineal simple
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1,1), y_true)
    y_pred_cal = lr.predict(y_pred.reshape(-1,1))

    mae_c = mean_absolute_error(y_true, y_pred_cal)
    rmse_c = np.sqrt(np.mean((y_true - y_pred_cal)**2))
    ns_c = nasa_score(y_true, y_pred_cal)
    print(f"{fd} AFTER  calib -> MAE: {mae_c:.3f}, RMSE: {rmse_c:.3f}, NASA: {ns_c:.3f}")

    results[-1].update({'MAE_after':mae_c, 'RMSE_after':rmse_c, 'NASA_after':ns_c, 'calib_coef':lr.coef_.item(), 'calib_intercept':lr.intercept_.item()})

    # gráficas después de calibrar
    plot_rul_lines(fd, y_true, y_pred_cal, FIG_PATH, suffix="_after")

    # guardar predicciones a CSV
    df_out = pd.DataFrame({'RUL_true':y_true, 'RUL_pred':y_pred, 'RUL_pred_cal':y_pred_cal})
    df_out.to_csv(os.path.join(FIG_PATH, f'{fd}_predictions.csv'), index=False)

# resumen global
if len(all_true) > 0:
    all_true_arr = np.concatenate(all_true)
    all_pred_arr = np.concatenate(all_pred)
    mae_g = mean_absolute_error(all_true_arr, all_pred_arr)
    rmse_g = np.sqrt(np.mean((all_true_arr - all_pred_arr)**2))
    ns_g = nasa_score(all_true_arr, all_pred_arr)
    print("GLOBAL BEFORE -> MAE:", mae_g, "RMSE:", rmse_g, "NASA:", ns_g)

    from sklearn.linear_model import LinearRegression
    lr_g = LinearRegression().fit(all_pred_arr.reshape(-1,1), all_true_arr)
    all_pred_cal = lr_g.predict(all_pred_arr.reshape(-1,1))
    mae_g_c = mean_absolute_error(all_true_arr, all_pred_cal)
    rmse_g_c = np.sqrt(np.mean((all_true_arr - all_pred_cal)**2))
    ns_g_c = nasa_score(all_true_arr, all_pred_cal)
    print("GLOBAL AFTER  -> MAE:", mae_g_c, "RMSE:", rmse_g_c, "NASA:", ns_g_c)

    pd.DataFrame(results).to_csv(os.path.join(FIG_PATH, "diagnose_summary_lines.csv"), index=False)
    pd.DataFrame({'RUL_true':all_true_arr, 'RUL_pred':all_pred_arr, 'RUL_pred_cal':all_pred_cal}).to_csv(os.path.join(FIG_PATH, "global_predictions_lines.csv"), index=False)

print("Done. Figures and CSVs in:", FIG_PATH)