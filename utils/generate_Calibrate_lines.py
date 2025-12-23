import os
import joblib
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

CSV_DIR = r"C:\Users\David\Documents\AeroGPT\results\CMAPSS\GRU"
MODEL_DIR = r"C:\Users\David\Documents\AeroGPT\models\cmapss"

def nasa_score(y_true, y_pred):
    e = y_pred - y_true
    score = np.where(e < 0, np.exp(-e/13.0)-1.0, np.exp(e/10.0)-1.0)
    return np.sum(score)

def calibrate_from_csvs(model_dir=CSV_DIR):
    # Buscar CSVs *_predictions.csv
    pattern = os.path.join(model_dir, "*_predictions.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No prediction CSVs encontrados en {model_dir}")

    summary_rows = []
    all_true_list = []
    all_pred_list = []

    for f in files:
        basename = os.path.basename(f)
        fd = basename.split("_")[0]
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Warning: no se pudo leer {f}: {e}")
            continue

        if not set(['RUL_true','RUL_pred']).issubset(df.columns):
            print(f"Warning: {f} no contiene columnas RUL_true y RUL_pred, se omite.")
            continue

        y_true = df['RUL_true'].values.astype(float)
        y_pred = df['RUL_pred'].values.astype(float)

        if len(y_true) == 0:
            print(f"Warning: {f} sin muestras, se omite.")
            continue

        all_true_list.append(y_true)
        all_pred_list.append(y_pred)

        lr = LinearRegression()
        lr.fit(y_pred.reshape(-1,1), y_true)
        y_cal = lr.predict(y_pred.reshape(-1,1))

        mae_b = mean_absolute_error(y_true, y_pred)
        mae_a = mean_absolute_error(y_true, y_cal)
        ns_b = nasa_score(y_true, y_pred)
        ns_a = nasa_score(y_true, y_cal)

        calib_path = os.path.join(model_dir, f"calib_{fd}.pkl")
        try:
            joblib.dump(lr, calib_path)
        except Exception as e:
            print(f"Warning: no se pudo guardar calibrador para {fd} en {calib_path}: {e}")

        summary_rows.append({
            'fd': fd,
            'n_samples': int(len(y_true)),
            'MAE_before': float(mae_b),
            'MAE_after': float(mae_a),
            'NASA_before': float(ns_b),
            'NASA_after': float(ns_a),
            'calib_coef': float(np.array(lr.coef_).ravel()[0]),
            'calib_intercept': float(lr.intercept_),
            'calib_path': calib_path
        })

    # Calibrador global
    if all_true_list and all_pred_list:
        all_true = np.concatenate(all_true_list)
        all_pred = np.concatenate(all_pred_list)

        if len(all_true) > 0:
            lr_g = LinearRegression().fit(all_pred.reshape(-1,1), all_true)
            all_cal = lr_g.predict(all_pred.reshape(-1,1))

            mae_b_g = mean_absolute_error(all_true, all_pred)
            mae_a_g = mean_absolute_error(all_true, all_cal)
            ns_b_g = nasa_score(all_true, all_pred)
            ns_a_g = nasa_score(all_true, all_cal)

            calib_g_path = os.path.join(model_dir, "calib_global.pkl")
            try:
                joblib.dump(lr_g, calib_g_path)
            except Exception as e:
                print(f"Warning: no se pudo guardar calibrador global en {calib_g_path}: {e}")
                calib_g_path = None

            summary_rows.append({
                'fd': 'GLOBAL',
                'n_samples': int(len(all_true)),
                'MAE_before': float(mae_b_g),
                'MAE_after': float(mae_a_g),
                'NASA_before': float(ns_b_g),
                'NASA_after': float(ns_a_g),
                'calib_coef': float(np.array(lr_g.coef_).ravel()[0]),
                'calib_intercept': float(lr_g.intercept_),
                'calib_path': calib_g_path
            })

    if not summary_rows:
        raise RuntimeError("No se generaron calibradores (ningún CSV válido encontrado).")

    summary_df = pd.DataFrame(summary_rows).sort_values(by='fd')
    summary_csv = os.path.join(model_dir, "calibration_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("Calibration complete. Summary saved to", summary_csv)
    return summary_df

if __name__ == "__main__":
    df_summary = calibrate_from_csvs()
    print(df_summary)