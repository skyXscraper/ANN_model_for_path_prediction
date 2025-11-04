"""
LAGEOS-1 Orbit Prediction Enhancement using Artificial Neural Networks
Final Integrated Script – Data Processing + ANN Training (2023)
"""

import numpy as np
import pandas as pd
import glob, os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ============================================================================
# PART 1 – CPF PARSING AND PREPROCESSING
# ============================================================================

def mjd_to_datetime(mjd, seconds_of_day):
    mjd_epoch = datetime(1858, 11, 17)
    return mjd_epoch + timedelta(days=mjd, seconds=seconds_of_day)

def parse_cpf_file(filepath):
    """Parse single CPF .hts file"""
    records = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("10"):
                parts = line.split()
                try:
                    mjd = int(parts[2])
                    sod = float(parts[3])
                    x, y, z = float(parts[-3])/1000, float(parts[-2])/1000, float(parts[-1])/1000
                    records.append({
                        "epoch_datetime": mjd_to_datetime(mjd, sod),
                        "X": x, "Y": y, "Z": z
                    })
                except Exception:
                    continue
    return pd.DataFrame(records)

def merge_all_cpf_files(folder):
    print("="*60)
    print("Reading CPF files ...")
    print("="*60)
    all_files = glob.glob(os.path.join(folder, "*.hts"))
    if not all_files:
        print("⚠ No CPF files found.")
        return pd.DataFrame(columns=["epoch_datetime", "X", "Y", "Z"])
    dfs = []
    for f in all_files:
        df = parse_cpf_file(f)
        if len(df):
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["epoch_datetime", "X", "Y", "Z"])
    cpf = pd.concat(dfs, ignore_index=True).drop_duplicates("epoch_datetime")
    cpf["epoch_datetime"] = pd.to_datetime(cpf["epoch_datetime"], errors="coerce")
    cpf = cpf.dropna(subset=["epoch_datetime"]).sort_values("epoch_datetime").reset_index(drop=True)
    print(f"✓ Total CPF records: {len(cpf)}")
    print(f"  Date range: {cpf['epoch_datetime'].min()} → {cpf['epoch_datetime'].max()}")
    return cpf

# ============================================================================
# PART 2 – KEPLERIAN PROPAGATION + ERROR CALCULATION
# ============================================================================

class KeplerianPropagator:
    MU = 398600.4418

    @staticmethod
    def propagate(a,e,i,omega,OMEGA,M0,dt):
        n = np.sqrt(KeplerianPropagator.MU / a**3)
        M = (M0 + n * dt) % (2*np.pi)
        return a,e,i,omega,OMEGA,M

    @staticmethod
    def elements_to_state(a,e,i,omega,OMEGA,M):
        E = KeplerianPropagator.solve_kepler(M,e)
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
        r = a*(1 - e*np.cos(E))
        x_p = r*np.cos(nu); y_p = r*np.sin(nu)
        p = a*(1-e**2)
        vx_p = -np.sqrt(KeplerianPropagator.MU/p)*np.sin(nu)
        vy_p = np.sqrt(KeplerianPropagator.MU/p)*(e+np.cos(nu))
        ci,si,co,so,cO,sO = np.cos(i),np.sin(i),np.cos(omega),np.sin(omega),np.cos(OMEGA),np.sin(OMEGA)
        R = np.array([
            [cO*co - sO*so*ci, -cO*so - sO*co*ci, sO*si],
            [sO*co + cO*so*ci, -sO*so + cO*co*ci, -cO*si],
            [so*si, co*si, ci]
        ])
        r_vec = R @ np.array([x_p, y_p, 0])
        v_vec = R @ np.array([vx_p, vy_p, 0])
        return np.concatenate([r_vec, v_vec])

    @staticmethod
    def solve_kepler(M, e, tol=1e-8):
        E = M if e < 0.8 else np.pi
        for _ in range(100):
            f = E - e*np.sin(E) - M
            E -= f / (1 - e*np.cos(E))
            if abs(f) < tol: break
        return E

def compute_error(pred, truth):
    return truth - pred

# ============================================================================
# PART 3 – DATASET CREATION
# ============================================================================

def create_ml_dataset(tle_df, cpf_df, max_days=7, step_hr=12, tol_sec=300):
    print("="*60)
    print("Creating ML dataset ...")
    print("="*60)
    cpf_df["epoch_datetime"] = pd.to_datetime(cpf_df["epoch_datetime"])
    dataset = []
    max_pred = max_days * 86400
    step = step_hr * 3600

    for idx, row in tle_df.iterrows():
        if idx % 50 == 0:
            print(f"Processing TLE {idx}/{len(tle_df)}...")
        t_cur = row["epoch_datetime"]
        cur_state = KeplerianPropagator.elements_to_state(
            row["semi_major_axis_km"], row["eccentricity"], row["inclination_rad"],
            np.deg2rad(row["arg_perigee_deg"]), np.deg2rad(row["raan_rad"]),
            np.deg2rad(row["mean_anomaly_deg"])
        )

        for dt in range(step, max_pred+step, step):
            t_future = t_cur + timedelta(seconds=dt)
            diffs = np.abs((cpf_df["epoch_datetime"] - pd.to_datetime(t_future)).dt.total_seconds())
            if diffs.empty: continue
            min_idx = diffs.idxmin()
            if diffs[min_idx] < tol_sec:
                true = cpf_df.loc[min_idx, ["X","Y","Z"]].values
                pred_a, pred_e, pred_i, pred_w, pred_O, pred_M = KeplerianPropagator.propagate(
                    row["semi_major_axis_km"], row["eccentricity"], row["inclination_rad"],
                    np.deg2rad(row["arg_perigee_deg"]), np.deg2rad(row["raan_rad"]),
                    np.deg2rad(row["mean_anomaly_deg"]), dt)
                pred_state = KeplerianPropagator.elements_to_state(pred_a,pred_e,pred_i,pred_w,pred_O,pred_M)
                err = compute_error(pred_state[:3], true)
                dataset.append({
                    "epoch_current": t_cur, "epoch_future": t_future, "delta_t_days": dt/86400,
                    "a": row["semi_major_axis_km"], "e": row["eccentricity"], "i": row["inclination_rad"],
                    "err_x": err[0], "err_y": err[1], "err_z": err[2]
                })

    df = pd.DataFrame(dataset)
    print(f"✓ Created {len(df)} samples.")
    return df

# ============================================================================
# PART 4 – ANN MODEL CLASS
# ============================================================================

class OrbitANN:
    def __init__(self, layers=1, neurons=20):
        self.layers = layers; self.neurons = neurons
        self.scaler_X, self.scaler_y = StandardScaler(), StandardScaler()
        self.model = None

    def build(self, input_dim):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        for _ in range(self.layers):
            model.add(layers.Dense(self.neurons, activation="sigmoid"))
        model.add(layers.Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=200):
        Xs_train = self.scaler_X.fit_transform(X_train)
        ys_train = self.scaler_y.fit_transform(y_train.reshape(-1,1))
        Xs_val = self.scaler_X.transform(X_val)
        ys_val = self.scaler_y.transform(y_val.reshape(-1,1))
        self.model = self.build(X_train.shape[1])
        cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        self.model.fit(Xs_train, ys_train, validation_data=(Xs_val, ys_val),
                       epochs=epochs, batch_size=64, verbose=0, callbacks=[cb])

    def predict(self, X):
        Xs = self.scaler_X.transform(X)
        yp = self.model.predict(Xs, verbose=0)
        return self.scaler_y.inverse_transform(yp).flatten()

# ============================================================================
# PART 5 – MAIN PIPELINE
# ============================================================================

def main():
    cpf_folder = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_files"
    tle_file = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\data_1.csv"
    out_dir = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\outputs_final"
    os.makedirs(out_dir, exist_ok=True)

    print("\n🚀 LAGEOS-1 ANN Orbit Prediction – Full Pipeline (2023)\n")

    cpf = merge_all_cpf_files(cpf_folder)
    cpf = cpf[(cpf["epoch_datetime"] >= "2023-01-01") & (cpf["epoch_datetime"] <= "2023-12-31")]

    tle = pd.read_csv(tle_file)
    tle["epoch_datetime"] = pd.to_datetime(tle["epoch_datetime"])
    tle_2023 = tle[(tle["epoch_datetime"] >= "2023-01-01") & (tle["epoch_datetime"] <= "2023-12-31")]
    if len(tle_2023) == 0:
        tle_2023 = tle.copy()

    df = create_ml_dataset(tle_2023, cpf)
    if df.empty:
        print("❌ No overlapping data – stopping.")
        return

    df.to_csv(os.path.join(out_dir, "ml_dataset_full.csv"), index=False)
    print(f"✓ Saved dataset ({len(df)} samples)")

    # ------------------------------------------------------------------------
    # TRAIN ANN MODEL
    # ------------------------------------------------------------------------
    X = df[["a","e","i","delta_t_days"]].values
    results = {}
    for comp in ["err_x","err_y","err_z"]:
        print(f"\n=== Training model for {comp} ===")
        y = df[comp].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = OrbitANN(layers=1, neurons=20)
        model.train(X_train, y_train, X_val, y_val, epochs=300)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results[comp] = (mae, rmse)
        print(f"{comp}: MAE={mae:.3f} km, RMSE={rmse:.3f} km")

        # Plot results
        plt.figure()
        plt.scatter(y_val, y_pred, alpha=0.6)
        plt.xlabel("True Error (km)")
        plt.ylabel("Predicted Error (km)")
        plt.title(f"ANN Prediction – {comp}")
        plt.grid()
        plt.savefig(os.path.join(out_dir, f"ANN_{comp}.png"), dpi=150)
        plt.close()

    print("\n=== Overall Results ===")
    for comp, (mae, rmse) in results.items():
        print(f"{comp}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    print("\n✅ ANN Training complete. Plots saved to:", out_dir)

if __name__ == "__main__":
    main()
