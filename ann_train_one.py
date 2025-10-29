"""
LAGEOS-1 Orbit Prediction Enhancement using Artificial Neural Networks
Complete Implementation with CPF File Parsing
Based on: Peng & Bai (2018) - ANN-Based ML Approach to Improve Orbit Prediction Accuracy
"""

import numpy as np
import pandas as pd
import glob
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ============================================================================
# PART 1: CPF FILE PARSING AND PREPROCESSING
# ============================================================================

def parse_cpf_file(filepath):
    """
    Parse CPF (Consolidated Prediction Format) file
    
    CPF Format:
    H1, H2, H5, H9: Header lines
    10: Position record (MJD, seconds, X, Y, Z in meters)
    20: Velocity record (VX, VY, VZ in m/s)
    """
    positions = []
    velocities = []
    current_position = None
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            
            if line.startswith('10'):
                # Position record: 10 0 MJD SECONDS X Y Z
                mjd = int(parts[2])
                seconds = float(parts[3])
                leap_indicator = int(parts[4])
                x = float(parts[5]) / 1000.0  # Convert meters to km
                y = float(parts[6]) / 1000.0
                z = float(parts[7]) / 1000.0
                
                # Convert MJD to datetime
                epoch = mjd_to_datetime(mjd, seconds)
                
                current_position = {
                    'epoch_datetime': epoch,
                    'X': x,
                    'Y': y,
                    'Z': z
                }
                
            elif line.startswith('20') and current_position is not None:
                # Velocity record: 20 0 VX VY VZ
                vx = float(parts[2]) / 1000.0  # Convert m/s to km/s
                vy = float(parts[3]) / 1000.0
                vz = float(parts[4]) / 1000.0
                
                current_position['VX'] = vx
                current_position['VY'] = vy
                current_position['VZ'] = vz
                
                positions.append(current_position)
                current_position = None
    
    return pd.DataFrame(positions)

def mjd_to_datetime(mjd, seconds_of_day):
    """Convert Modified Julian Date to datetime"""
    # MJD epoch is November 17, 1858
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)
    delta = timedelta(days=mjd, seconds=seconds_of_day)
    return mjd_epoch + delta

def merge_all_cpf_files(folder_path):
    """Merge all CPF files in folder into single dataframe"""
    print("=" * 60)
    print("PARSING CPF FILES")
    print("=" * 60)
    
    all_files = glob.glob(folder_path + "/*.hts")
    print(f"Found {len(all_files)} CPF files")
    
    dfs = []
    for file in all_files:
        try:
            df = parse_cpf_file(file)
            print(f"✓ Parsed {file.split('/')[-1]}: {len(df)} records")
            dfs.append(df)
        except Exception as e:
            print(f"✗ Error parsing {file}: {e}")
    
    # Merge and remove duplicates
    cpf_truth = pd.concat(dfs, ignore_index=True)
    cpf_truth = cpf_truth.drop_duplicates(subset=['epoch_datetime'])
    cpf_truth = cpf_truth.sort_values('epoch_datetime').reset_index(drop=True)
    
    print(f"\nTotal CPF records: {len(cpf_truth)}")
    print(f"Date range: {cpf_truth['epoch_datetime'].min()} to {cpf_truth['epoch_datetime'].max()}")
    
    return cpf_truth


# ============================================================================
# PART 2: TLE TO CARTESIAN STATE CONVERSION
# ============================================================================

class TLEToCartesian:
    """Convert TLE orbital elements to Cartesian ECI state"""
    
    MU_EARTH = 398600.4418  # km^3/s^2
    
    @staticmethod
    def elements_to_state(a, e, i, omega, OMEGA, M):
        """
        Convert Keplerian elements to Cartesian state (ECI frame)
        
        Args:
            a: semi-major axis (km)
            e: eccentricity
            i: inclination (rad)
            omega: argument of perigee (rad)
            OMEGA: right ascension of ascending node (rad)
            M: mean anomaly (rad)
        
        Returns:
            state: [X, Y, Z, VX, VY, VZ] in ECI frame
        """
        # Solve Kepler's equation for eccentric anomaly
        E = TLEToCartesian.mean_to_eccentric_anomaly(M, e)
        
        # Compute true anomaly
        nu = TLEToCartesian.eccentric_to_true_anomaly(E, e)
        
        # Compute position and velocity in perifocal frame
        r_pqw, v_pqw = TLEToCartesian.perifocal_state(a, e, nu)
        
        # Transform to ECI frame
        r_eci, v_eci = TLEToCartesian.perifocal_to_eci(r_pqw, v_pqw, i, omega, OMEGA)
        
        return np.concatenate([r_eci, v_eci])
    
    @staticmethod
    def mean_to_eccentric_anomaly(M, e, tol=1e-8, max_iter=50):
        """Solve Kepler's equation using Newton-Raphson"""
        E = M if e < 0.8 else np.pi
        
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            fp = 1 - e * np.cos(E)
            E_new = E - f / fp
            
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        
        return E
    
    @staticmethod
    def eccentric_to_true_anomaly(E, e):
        """Convert eccentric anomaly to true anomaly"""
        return 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )
    
    @staticmethod
    def perifocal_state(a, e, nu):
        """Compute position and velocity in perifocal (PQW) frame"""
        mu = TLEToCartesian.MU_EARTH
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(nu))
        
        # Position
        r_pqw = r * np.array([np.cos(nu), np.sin(nu), 0])
        
        # Velocity
        v_pqw = np.sqrt(mu / p) * np.array([
            -np.sin(nu),
            e + np.cos(nu),
            0
        ])
        
        return r_pqw, v_pqw
    
    @staticmethod
    def perifocal_to_eci(r_pqw, v_pqw, i, omega, OMEGA):
        """Transform from perifocal to ECI frame"""
        # Rotation matrix: R = R3(-OMEGA) * R1(-i) * R3(-omega)
        cos_omega, sin_omega = np.cos(omega), np.sin(omega)
        cos_OMEGA, sin_OMEGA = np.cos(OMEGA), np.sin(OMEGA)
        cos_i, sin_i = np.cos(i), np.sin(i)
        
        R = np.array([
            [cos_OMEGA * cos_omega - sin_OMEGA * sin_omega * cos_i,
             -cos_OMEGA * sin_omega - sin_OMEGA * cos_omega * cos_i,
             sin_OMEGA * sin_i],
            [sin_OMEGA * cos_omega + cos_OMEGA * sin_omega * cos_i,
             -sin_OMEGA * sin_omega + cos_OMEGA * cos_omega * cos_i,
             -cos_OMEGA * sin_i],
            [sin_omega * sin_i,
             cos_omega * sin_i,
             cos_i]
        ])
        
        r_eci = R @ r_pqw
        v_eci = R @ v_pqw
        
        return r_eci, v_eci


# ============================================================================
# PART 3: COORDINATE FRAME TRANSFORMATIONS
# ============================================================================

def eci_to_rsw(state_eci):
    """
    Transform state from ECI to RSW (Radial-Along track-Cross track) frame
    
    RSW frame definition:
    - R (Radial): Points from Earth center to satellite
    - S (Along-track): Perpendicular to R in orbital plane, velocity direction
    - W (Cross-track): Normal to orbital plane (angular momentum direction)
    """
    r_vec = state_eci[:3]
    v_vec = state_eci[3:]
    
    # Unit vectors
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)
    s_hat = np.cross(h_hat, r_hat)
    
    # Rotation matrix
    R_eci_to_rsw = np.array([r_hat, s_hat, h_hat])
    
    # Transform
    r_rsw = R_eci_to_rsw @ r_vec
    v_rsw = R_eci_to_rsw @ v_vec
    
    return np.concatenate([r_rsw, v_rsw])

def compute_prediction_error_rsw(predicted_eci, true_eci):
    """Compute prediction error in RSW frame"""
    error_eci = true_eci - predicted_eci
    
    # Use predicted state as reference for RSW frame
    r_vec = predicted_eci[:3]
    v_vec = predicted_eci[3:]
    
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)
    s_hat = np.cross(h_hat, r_hat)
    
    R_eci_to_rsw = np.array([r_hat, s_hat, h_hat])
    
    # Transform error
    error_pos_rsw = R_eci_to_rsw @ error_eci[:3]
    error_vel_rsw = R_eci_to_rsw @ error_eci[3:]
    
    return np.concatenate([error_pos_rsw, error_vel_rsw])


# ============================================================================
# PART 4: DATASET CREATION
# ============================================================================

def create_ml_dataset(tle_df, cpf_df, max_prediction_days=7, time_tolerance_sec=300):
    """
    Create machine learning dataset for orbit prediction
    
    Learning Variables (Λ):
    1. Prediction duration (Δt)
    2. Current orbital elements (a, e, i, ω, Ω, M)
    3. Current state in ECI (X, Y, Z, VX, VY, VZ)
    4. B* drag term
    5. Trigonometric elements (sin/cos of i, Ω)
    6. Predicted state at future epoch
    
    Target Variables (e):
    7. True prediction error in RSW: (ex, ey, ez, evx, evy, evz)
    """
    print("\n" + "=" * 60)
    print("CREATING MACHINE LEARNING DATASET")
    print("=" * 60)
    
    dataset = []
    max_pred_seconds = max_prediction_days * 86400
    
    # Process each TLE as current epoch
    for i in range(len(tle_df)):
        if i % 100 == 0:
            print(f"Processing TLE {i}/{len(tle_df)}...")
        
        current = tle_df.iloc[i]
        t_current = current['epoch_datetime']
        
        # Compute current state from TLE
        current_state = TLEToCartesian.elements_to_state(
            current['semi_major_axis_km'],
            current['eccentricity'],
            current['inclination_rad'],
            current['arg_perigee_deg'] * np.pi / 180,
            current['raan_rad'],
            current['mean_anomaly_deg'] * np.pi / 180
        )
        
        # Try to predict to future TLEs
        for j in range(i + 1, len(tle_df)):
            future = tle_df.iloc[j]
            t_future = future['epoch_datetime']
            
            delta_t_sec = (t_future - t_current).total_seconds()
            delta_t_days = delta_t_sec / 86400
            
            if delta_t_sec > max_pred_seconds:
                break
            
            # Predicted state (simple Keplerian propagation)
            predicted_state = TLEToCartesian.elements_to_state(
                future['semi_major_axis_km'],
                future['eccentricity'],
                future['inclination_rad'],
                future['arg_perigee_deg'] * np.pi / 180,
                future['raan_rad'],
                future['mean_anomaly_deg'] * np.pi / 180
            )
            
            # Find closest CPF truth measurement
            time_diff = np.abs((cpf_df['epoch_datetime'] - t_future).dt.total_seconds())
            closest_idx = time_diff.argmin()
            
            if time_diff.iloc[closest_idx] < time_tolerance_sec:
                cpf_match = cpf_df.iloc[closest_idx]
                true_state = np.array([
                    cpf_match['X'], cpf_match['Y'], cpf_match['Z'],
                    cpf_match['VX'], cpf_match['VY'], cpf_match['VZ']
                ])
                
                # Compute prediction error in RSW frame
                error_rsw = compute_prediction_error_rsw(predicted_state, true_state)
                
                # Compile data point
                data_point = {
                    # Time info
                    'epoch_current': t_current,
                    'epoch_future': t_future,
                    'delta_t_days': delta_t_days,
                    'delta_t_hours': delta_t_days * 24,
                    
                    # Current orbital elements
                    'a': current['semi_major_axis_km'],
                    'e': current['eccentricity'],
                    'i_rad': current['inclination_rad'],
                    'i_deg': current['inclination_deg'],
                    'omega_rad': current['arg_perigee_deg'] * np.pi / 180,
                    'OMEGA_rad': current['raan_rad'],
                    'OMEGA_deg': current['raan_deg'],
                    'M_rad': current['mean_anomaly_deg'] * np.pi / 180,
                    'M_deg': current['mean_anomaly_deg'],
                    
                    # Current state (ECI)
                    'X_curr': current_state[0],
                    'Y_curr': current_state[1],
                    'Z_curr': current_state[2],
                    'VX_curr': current_state[3],
                    'VY_curr': current_state[4],
                    'VZ_curr': current_state[5],
                    
                    # Drag and motion parameters
                    'bstar': current['bstar'],
                    'mean_motion': current['mean_motion_rad_s'],
                    'period_min': current['orbital_period_min'],
                    
                    # Trigonometric features
                    'sin_i': current['sin_inclination'],
                    'cos_i': current['cos_inclination'],
                    'sin_OMEGA': current['sin_raan'],
                    'cos_OMEGA': current['cos_raan'],
                    
                    # Predicted future state (ECI)
                    'X_pred': predicted_state[0],
                    'Y_pred': predicted_state[1],
                    'Z_pred': predicted_state[2],
                    'VX_pred': predicted_state[3],
                    'VY_pred': predicted_state[4],
                    'VZ_pred': predicted_state[5],
                    
                    # Target: Prediction errors in RSW frame
                    'e_x': error_rsw[0],       # Radial error (km)
                    'e_y': error_rsw[1],       # Along-track error (km)
                    'e_z': error_rsw[2],       # Cross-track error (km)
                    'e_vx': error_rsw[3],      # Radial velocity error (km/s)
                    'e_vy': error_rsw[4],      # Along-track velocity error (km/s)
                    'e_vz': error_rsw[5],      # Cross-track velocity error (km/s)
                }
                
                dataset.append(data_point)
    
    df = pd.DataFrame(dataset)
    print(f"\nDataset created: {len(df)} samples")
    print(f"Prediction duration range: {df['delta_t_days'].min():.2f} to {df['delta_t_days'].max():.2f} days")
    
    return df


# ============================================================================
# PART 5: ANN MODEL
# ============================================================================

class OrbitPredictionANN:
    """
    Artificial Neural Network for orbit prediction error correction
    Based on Peng & Bai (2018)
    """
    
    def __init__(self, n_hidden_layers=1, neurons_per_layer=20, random_seed=42):
        self.n_hidden_layers = n_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.random_seed = random_seed
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
    
    def build_model(self, input_dim, component_name):
        """Build ANN with log-sigmoid hidden layers and linear output"""
        model = keras.Sequential(name=f'ANN_{component_name}')
        
        # Input
        model.add(layers.InputLayer(input_shape=(input_dim,)))
        
        # Hidden layers with sigmoid activation (log-sigmoid)
        for i in range(self.n_hidden_layers):
            model.add(layers.Dense(
                self.neurons_per_layer,
                activation='sigmoid',
                kernel_initializer=keras.initializers.GlorotUniform(seed=self.random_seed),
                name=f'hidden_{i+1}'
            ))
            model.add(layers.Dropout(0.1))  # Light regularization
        
        # Output layer (linear)
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_component(self, X_train, y_train, X_val, y_val, component_name, epochs=500):
        """Train one ANN for one error component"""
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        self.scalers_X[component_name] = scaler_X
        self.scalers_y[component_name] = scaler_y
        
        # Build model
        model = self.build_model(X_train_scaled.shape[1], component_name)
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        self.models[component_name] = model
        
        return history
    
    def train_all_components(self, X_train, y_train, X_val, y_val):
        """Train ANNs for all 6 error components"""
        
        components = ['e_x', 'e_y', 'e_z', 'e_vx', 'e_vy', 'e_vz']
        component_names = {
            'e_x': 'Radial Position',
            'e_y': 'Along-Track Position',
            'e_z': 'Cross-Track Position',
            'e_vx': 'Radial Velocity',
            'e_vy': 'Along-Track Velocity',
            'e_vz': 'Cross-Track Velocity'
        }
        
        histories = {}
        
        print("\n" + "=" * 60)
        print("TRAINING ANN MODELS")
        print("=" * 60)
        
        for comp in components:
            print(f"\nTraining {component_names[comp]} ({comp})...")
            
            history = self.train_component(
                X_train,
                y_train[comp].values,
                X_val,
                y_val[comp].values,
                comp
            )
            
            histories[comp] = history
            
            # Report
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            print(f"  Final Train MSE: {final_train_loss:.6f}")
            print(f"  Final Val MSE: {final_val_loss:.6f}")
            print(f"  Epochs trained: {len(history.history['loss'])}")
        
        return histories
    
    def predict(self, X, component):
        """Predict error for one component"""
        X_scaled = self.scalers_X[component].transform(X)
        y_pred_scaled = self.models[component].predict(X_scaled, verbose=0).flatten()
        y_pred = self.scalers_y[component].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return y_pred
    
    def predict_all(self, X):
        """Predict all components"""
        predictions = {}
        for comp in ['e_x', 'e_y', 'e_z', 'e_vx', 'e_vy', 'e_vz']:
            predictions[comp] = self.predict(X, comp)
        return pd.DataFrame(predictions)
    
    def save_models(self, base_path):
        """Save trained models"""
        import pickle
        for comp, model in self.models.items():
            model.save(f"{base_path}_model_{comp}.h5")
            with open(f"{base_path}_scaler_X_{comp}.pkl", 'wb') as f:
                pickle.dump(self.scalers_X[comp], f)
            with open(f"{base_path}_scaler_y_{comp}.pkl", 'wb') as f:
                pickle.dump(self.scalers_y[comp], f)
        print(f"Models saved to {base_path}_*")


# ============================================================================
# PART 6: EVALUATION
# ============================================================================

def compute_PML(true_error, ml_predicted_error):
    """Performance metric: % of error remaining after ML correction"""
    residual = true_error - ml_predicted_error
    PML = 100 * np.sum(np.abs(residual)) / np.sum(np.abs(true_error))
    return PML

def evaluate_model(ann, X_test, y_test):
    """Evaluate trained ANN models"""
    
    print("\n" + "=" * 60)
    print("PERFORMANCE EVALUATION")
    print("=" * 60)
    
    components = {
        'e_x': 'Radial',
        'e_y': 'Along-Track',
        'e_z': 'Cross-Track',
        'e_vx': 'Radial Velocity',
        'e_vy': 'Along-Track Velocity',
        'e_vz': 'Cross-Track Velocity'
    }
    
    results = {}
    
    print(f"\n{'Component':<25} {'PML (%)':<12} {'Mean |Error| (km)':<20} {'Mean |Residual| (km)':<20}")
    print("-" * 80)
    
    for comp, name in components.items():
        true_error = y_test[comp].values
        ml_pred_error = ann.predict(X_test, comp)
        
        PML = compute_PML(true_error, ml_pred_error)
        mean_true = np.mean(np.abs(true_error))
        mean_residual = np.mean(np.abs(true_error - ml_pred_error))
        
        results[comp] = {
            'PML': PML,
            'mean_true_error': mean_true,
            'mean_residual_error': mean_residual,
            'improvement': 100 - PML
        }
        
        print(f"{name:<25} {PML:>10.2f}%  {mean_true:>18.4f}  {mean_residual:>18.4f}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Paths - UPDATE THESE
    cpf_folder = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_files"
    tle_file = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\data_1.csv"
    output_dir = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\outputs"
    
    # Step 1: Parse and merge CPF files
    print("\n" + "=" * 60)
    print("STEP 1: PARSE CPF FILES")
    print("=" * 60)
    cpf_truth = merge_all_cpf_files(cpf_folder)
    cpf_truth.to_csv(f"{output_dir}/cpf_truth_merged.csv", index=False)
    print(f"✓ Saved merged CPF data")
    
    # Step 2: Load TLE data
    print("\n" + "=" * 60)
    print("STEP 2: LOAD TLE DATA")
    print("=" * 60)
    tle_df = pd.read_csv(tle_file)
    tle_df['epoch_datetime'] = pd.to_datetime(tle_df['epoch_datetime'])
    print(f"Loaded {len(tle_df)} TLE records")
    print(f"Date range: {tle_df['epoch_datetime'].min()} to {tle_df['epoch_datetime'].max()}")
    
    # Step 3: Create ML dataset
    ml_dataset = create_ml_dataset(tle_df, cpf_truth, max_prediction_days=7)
    ml_dataset.to_csv(f"{output_dir}/ml_dataset.csv", index=False)
    print(f"✓ Saved ML dataset")
    
    # Step 4: Prepare train/val/test splits
    print("\n" + "=" * 60)
    print("STEP 4: PREPARE DATA SPLITS")
    print("=" * 60)
    
    # Define learning variables (features)
    feature_cols = [
        'delta_t_days', 'delta_t_hours',
        'a', 'e', 'i_rad', 'omega_rad', 'OMEGA_rad', 'M_rad',
        'X_curr', 'Y_curr', 'Z_curr', 'VX_curr', 'VY_curr', 'VZ_curr',
        'bstar', 'mean_motion', 'period_min',
        'sin_i', 'cos_i', 'sin_OMEGA', 'cos_OMEGA',
        'X_pred', 'Y_pred', 'Z_pred', 'VX_pred', 'VY_pred', 'VZ_pred'
    ]
    
    target_cols = ['e_x', 'e_y', 'e_z', 'e_vx', 'e_vy', 'e_vz']
    
    X = ml_dataset[feature_cols].values
    y = ml_dataset[target_cols]
    
    # Split: 70% train, 15% val, 15% test (temporal split for Type II generalization)
    split_idx1 = int(0.7 * len(X))
    split_idx2 = int(0.85 * len(X))
    
    X_train, X_val, X_test = X[:split_idx1], X[split_idx1:split_idx2], X[split_idx2:]
    y_train, y_val, y_test = y[:split_idx1], y[split_idx1:split_idx2], y[split_idx2:]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Step 5: Train ANN models
    print("\n" + "=" * 60)
    print("STEP 5: TRAIN ANN MODELS")
    print("=" * 60)
    
    # Try different network architectures
    configs = [
        {'layers': 1, 'neurons': 20, 'name': '1x20'},
        {'layers': 1, 'neurons': 25, 'name': '1x25'},
        {'layers': 2, 'neurons': 20, 'name': '2x20'},
    ]
    
    best_config = None
    best_results = None
    best_ann = None
    
    for config in configs:
        print(f"\n--- Testing configuration: {config['name']} ---")
        
        ann = OrbitPredictionANN(
            n_hidden_layers=config['layers'],
            neurons_per_layer=config['neurons'],
            random_seed=42
        )
        
        histories = ann.train_all_components(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_predictions = ann.predict_all(X_val)
        val_pml = {}
        for comp in target_cols:
            val_pml[comp] = compute_PML(y_val[comp].values, val_predictions[comp].values)
        
        avg_pml = np.mean(list(val_pml.values()))
        print(f"\nAverage Validation PML: {avg_pml:.2f}%")
        
        if best_config is None or avg_pml < best_results:
            best_config = config
            best_results = avg_pml
            best_ann = ann
            print("✓ New best configuration!")
    
    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION: {best_config['name']} with avg PML = {best_results:.2f}%")
    print(f"{'='*60}")
    
    # Step 6: Final evaluation on test set
    print("\n" + "=" * 60)
    print("STEP 6: FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_results = evaluate_model(best_ann, X_test, y_test)
    
    # Step 7: Visualize results
    print("\n" + "=" * 60)
    print("STEP 7: GENERATE VISUALIZATIONS")
    print("=" * 60)
    
    plot_results(best_ann, X_test, y_test, ml_dataset[split_idx2:], output_dir)
    
    # Step 8: Save models
    print("\n" + "=" * 60)
    print("STEP 8: SAVE TRAINED MODELS")
    print("=" * 60)
    
    best_ann.save_models(f"{output_dir}/lageos_ann")
    
    # Save results summary
    results_df = pd.DataFrame(test_results).T
    results_df.to_csv(f"{output_dir}/performance_metrics.csv")
    print(f"✓ Saved performance metrics")
    
    # Step 9: Generate report
    generate_report(test_results, best_config, output_dir)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    
    return best_ann, test_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_results(ann, X_test, y_test, test_metadata, output_dir):
    """Generate comprehensive result visualizations"""
    
    # Predict
    y_pred = ann.predict_all(X_test)
    
    components = {
        'e_x': ('Radial', 'km'),
        'e_y': ('Along-Track', 'km'),
        'e_z': ('Cross-Track', 'km'),
        'e_vx': ('Radial Velocity', 'km/s'),
        'e_vy': ('Along-Track Velocity', 'km/s'),
        'e_vz': ('Cross-Track Velocity', 'km/s')
    }
    
    # Plot 1: Error vs Prediction Duration
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    delta_t = test_metadata['delta_t_days'].values
    
    for idx, (comp, (name, unit)) in enumerate(components.items()):
        ax = axes[idx]
        
        true_error = y_test[comp].values
        ml_error = y_pred[comp].values
        residual = true_error - ml_error
        
        # Bin by prediction duration
        bins = np.linspace(0, 7, 15)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        true_binned = []
        residual_binned = []
        
        for i in range(len(bins) - 1):
            mask = (delta_t >= bins[i]) & (delta_t < bins[i+1])
            if np.sum(mask) > 0:
                true_binned.append(np.mean(np.abs(true_error[mask])))
                residual_binned.append(np.mean(np.abs(residual[mask])))
            else:
                true_binned.append(0)
                residual_binned.append(0)
        
        ax.plot(bin_centers, true_binned, 'o-', label='Original Error', color='red', linewidth=2)
        ax.plot(bin_centers, residual_binned, 's-', label='Residual Error', color='green', linewidth=2)
        ax.set_xlabel('Prediction Duration (days)', fontsize=10)
        ax.set_ylabel(f'Mean |Error| ({unit})', fontsize=10)
        ax.set_title(f'{name} Error ({comp})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_vs_duration.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: error_vs_duration.png")
    plt.close()
    
    # Plot 2: Error Distribution Histograms
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (comp, (name, unit)) in enumerate(components.items()):
        ax = axes[idx]
        
        true_error = y_test[comp].values
        residual = true_error - y_pred[comp].values
        
        ax.hist(true_error, bins=50, alpha=0.5, label='Original Error', color='red', edgecolor='black')
        ax.hist(residual, bins=50, alpha=0.5, label='Residual Error', color='green', edgecolor='black')
        ax.set_xlabel(f'Error ({unit})', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name} Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: error_distributions.png")
    plt.close()
    
    # Plot 3: Scatter plots - Predicted vs True
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (comp, (name, unit)) in enumerate(components.items()):
        ax = axes[idx]
        
        true_error = y_test[comp].values
        ml_error = y_pred[comp].values
        
        ax.scatter(true_error, ml_error, alpha=0.3, s=10, color='blue')
        
        # Perfect prediction line
        lims = [min(true_error.min(), ml_error.min()), max(true_error.max(), ml_error.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'True Error ({unit})', fontsize=10)
        ax.set_ylabel(f'ML Predicted Error ({unit})', fontsize=10)
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predicted_vs_true.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: predicted_vs_true.png")
    plt.close()
    
    # Plot 4: Performance Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    comp_names = [components[c][0] for c in components.keys()]
    pml_values = [compute_PML(y_test[c].values, y_pred[c].values) for c in components.keys()]
    
    colors = ['green' if p < 30 else 'orange' if p < 60 else 'red' for p in pml_values]
    bars = ax.bar(comp_names, pml_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, pml_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='No Improvement (100%)')
    ax.set_ylabel('PML (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metric (PML) by Component\n(Lower is Better)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_metrics.png")
    plt.close()
    
    # Plot 5: Training History (if available)
    # This would require saving histories in main()


def generate_report(results, config, output_dir):
    """Generate text report of results"""
    
    report_path = f"{output_dir}/analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LAGEOS-1 ORBIT PREDICTION ENHANCEMENT - ANALYSIS REPORT\n")
        f.write("Artificial Neural Network-Based Machine Learning Approach\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BEST MODEL CONFIGURATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Architecture: {config['name']}\n")
        f.write(f"Hidden Layers: {config['layers']}\n")
        f.write(f"Neurons per Layer: {config['neurons']}\n")
        f.write(f"Activation Function: Log-Sigmoid (hidden), Linear (output)\n")
        f.write(f"Optimizer: Adam with LR reduction\n")
        f.write(f"Early Stopping: Patience = 20\n\n")
        
        f.write("PERFORMANCE METRICS (PML %):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Component':<25} {'PML (%)':<12} {'Improvement (%)':<15}\n")
        f.write("-" * 70 + "\n")
        
        component_names = {
            'e_x': 'Radial Position',
            'e_y': 'Along-Track Position',
            'e_z': 'Cross-Track Position',
            'e_vx': 'Radial Velocity',
            'e_vy': 'Along-Track Velocity',
            'e_vz': 'Cross-Track Velocity'
        }
        
        for comp, name in component_names.items():
            pml = results[comp]['PML']
            improvement = results[comp]['improvement']
            f.write(f"{name:<25} {pml:>10.2f}%  {improvement:>13.2f}%\n")
        
        avg_pml = np.mean([results[c]['PML'] for c in component_names.keys()])
        avg_improvement = 100 - avg_pml
        
        f.write("-" * 70 + "\n")
        f.write(f"{'AVERAGE':<25} {avg_pml:>10.2f}%  {avg_improvement:>13.2f}%\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write("• PML < 30%: Excellent performance (significant error reduction)\n")
        f.write("• PML 30-60%: Moderate performance (partial error reduction)\n")
        f.write("• PML > 60%: Limited improvement (challenging component)\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 70 + "\n")
        
        # Analyze results
        excellent = [name for comp, name in component_names.items() if results[comp]['PML'] < 30]
        moderate = [name for comp, name in component_names.items() if 30 <= results[comp]['PML'] < 60]
        limited = [name for comp, name in component_names.items() if results[comp]['PML'] >= 60]
        
        if excellent:
            f.write(f"Excellent Performance: {', '.join(excellent)}\n")
        if moderate:
            f.write(f"Moderate Performance: {', '.join(moderate)}\n")
        if limited:
            f.write(f"Limited Improvement: {', '.join(limited)}\n")
            f.write("  → Cross-track errors are typically harder to predict due to:\n")
            f.write("     - Already well-modeled Earth oblateness (J2 perturbation)\n")
            f.write("     - Errors dominated by measurement noise\n")
            f.write("     - Less systematic error patterns\n\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Use ML-corrected predictions for radial and along-track components\n")
        f.write("2. Consider separate strategies for cross-track components\n")
        f.write("3. Investigate additional features (solar activity, atmospheric density)\n")
        f.write("4. Test deeper networks (3-4 hidden layers) for cross-track errors\n")
        f.write("5. Extend training data to multiple years for better generalization\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Saved: analysis_report.txt")


# ============================================================================
# UTILITY: APPLY TRAINED MODEL TO NEW DATA
# ============================================================================

def apply_ml_correction(ann, tle_current, tle_future, prediction_duration_days):
    """
    Apply trained ANN to correct orbit prediction
    
    Args:
        ann: Trained OrbitPredictionANN model
        tle_current: Current TLE record (pandas Series)
        tle_future: Future TLE record for prediction (pandas Series)
        prediction_duration_days: Prediction duration in days
    
    Returns:
        corrected_state: ML-corrected state in ECI frame
        original_state: Original predicted state
        ml_error_correction: Predicted error correction in RSW frame
    """
    
    # Compute states
    current_state = TLEToCartesian.elements_to_state(
        tle_current['semi_major_axis_km'],
        tle_current['eccentricity'],
        tle_current['inclination_rad'],
        tle_current['arg_perigee_deg'] * np.pi / 180,
        tle_current['raan_rad'],
        tle_current['mean_anomaly_deg'] * np.pi / 180
    )
    
    original_predicted_state = TLEToCartesian.elements_to_state(
        tle_future['semi_major_axis_km'],
        tle_future['eccentricity'],
        tle_future['inclination_rad'],
        tle_future['arg_perigee_deg'] * np.pi / 180,
        tle_future['raan_rad'],
        tle_future['mean_anomaly_deg'] * np.pi / 180
    )
    
    # Prepare features
    features = np.array([[
        prediction_duration_days,
        prediction_duration_days * 24,
        tle_current['semi_major_axis_km'],
        tle_current['eccentricity'],
        tle_current['inclination_rad'],
        tle_current['arg_perigee_deg'] * np.pi / 180,
        tle_current['raan_rad'],
        tle_current['mean_anomaly_deg'] * np.pi / 180,
        current_state[0], current_state[1], current_state[2],
        current_state[3], current_state[4], current_state[5],
        tle_current['bstar'],
        tle_current['mean_motion_rad_s'],
        tle_current['orbital_period_min'],
        tle_current['sin_inclination'],
        tle_current['cos_inclination'],
        tle_current['sin_raan'],
        tle_current['cos_raan'],
        original_predicted_state[0], original_predicted_state[1], original_predicted_state[2],
        original_predicted_state[3], original_predicted_state[4], original_predicted_state[5]
    ]])
    
    # Predict error correction in RSW frame
    ml_error_rsw = ann.predict_all(features).iloc[0].values
    
    # Convert RSW correction back to ECI
    r_vec = original_predicted_state[:3]
    v_vec = original_predicted_state[3:]
    
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)
    s_hat = np.cross(h_hat, r_hat)
    
    R_rsw_to_eci = np.array([r_hat, s_hat, h_hat]).T
    
    error_pos_eci = R_rsw_to_eci @ ml_error_rsw[:3]
    error_vel_eci = R_rsw_to_eci @ ml_error_rsw[3:]
    
    # Apply correction
    corrected_state = original_predicted_state - np.concatenate([error_pos_eci, error_vel_eci])
    
    return corrected_state, original_predicted_state, ml_error_rsw


# ============================================================================
# RUN MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    
    # Execute main pipeline
    trained_model, final_results = main()
    
    print("\n" + "🎉" * 30)
    print("SUCCESS! LAGEOS-1 orbit prediction model trained and evaluated.")
    print("🎉" * 30)