"""
LAGEOS-1 CPF File Parser - Standalone Script
Parse all CPF files and create merged truth data CSV
"""

import pandas as pd
import glob
from datetime import datetime, timedelta
import os

def mjd_to_datetime(mjd, seconds_of_day):
    """
    Convert Modified Julian Date to datetime
    MJD epoch is November 17, 1858
    """
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)
    delta = timedelta(days=int(mjd), seconds=seconds_of_day)
    return mjd_epoch + delta

def parse_cpf_file(filepath):
    """
    Parse CPF (Consolidated Prediction Format) file for LAGEOS-1
    
    CPF File Structure:
    - H1, H2, H5, H9: Header lines (metadata)
    - 10: Position record format: 10 DirectionFlag MJD SecondsOfDay LeapSecond X Y Z
    - 20: Velocity record format: 20 DirectionFlag VX VY VZ
    
    Units:
    - Position: meters (converted to km)
    - Velocity: m/s (converted to km/s)
    - Time: MJD (Modified Julian Date) + seconds of day
    """
    
    positions = []
    current_position = None
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                
                # Skip empty lines
                if not parts:
                    continue
                
                # Position record
                if parts[0] == '10':
                    """
                    Example: 10 0 53704  84600.00000  0  -1327941.189  6431786.532  -10436243.142
                    Fields:
                    [0] = '10' (record type)
                    [1] = direction flag (0 = predict, 1 = filtered, 2 = smoothed)
                    [2] = MJD (Modified Julian Date)
                    [3] = Seconds of day
                    [4] = Leap second indicator
                    [5] = X position (meters)
                    [6] = Y position (meters)
                    [7] = Z position (meters)
                    """
                    
                    direction_flag = int(parts[1])
                    mjd = int(parts[2])
                    seconds_of_day = float(parts[3])
                    leap_indicator = int(parts[4])
                    
                    # Position in meters, convert to km
                    x_m = float(parts[5])
                    y_m = float(parts[6])
                    z_m = float(parts[7])
                    
                    x_km = x_m / 1000.0
                    y_km = y_m / 1000.0
                    z_km = z_m / 1000.0
                    
                    # Convert MJD to datetime
                    epoch = mjd_to_datetime(mjd, seconds_of_day)
                    
                    # Store current position (wait for corresponding velocity)
                    current_position = {
                        'epoch_datetime': epoch,
                        'mjd': mjd,
                        'seconds_of_day': seconds_of_day,
                        'X': x_km,
                        'Y': y_km,
                        'Z': z_km,
                        'direction_flag': direction_flag
                    }
                
                # Velocity record (follows immediately after position record)
                elif parts[0] == '20' and current_position is not None:
                    """
                    Example: 20 0  3257.254913  4449.442509  2332.122643
                    Fields:
                    [0] = '20' (record type)
                    [1] = direction flag (should match position record)
                    [2] = VX velocity (m/s)
                    [3] = VY velocity (m/s)
                    [4] = VZ velocity (m/s)
                    """
                    
                    # Velocity in m/s, convert to km/s
                    vx_ms = float(parts[2])
                    vy_ms = float(parts[3])
                    vz_ms = float(parts[4])
                    
                    vx_kms = vx_ms / 1000.0
                    vy_kms = vy_ms / 1000.0
                    vz_kms = vz_ms / 1000.0
                    
                    # Add velocities to current position
                    current_position['VX'] = vx_kms
                    current_position['VY'] = vy_kms
                    current_position['VZ'] = vz_kms
                    
                    # Append complete record
                    positions.append(current_position)
                    current_position = None  # Reset for next position
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()
    
    return pd.DataFrame(positions)

def merge_all_cpf_files(folder_path, output_csv_path):
    """
    Merge all CPF files in folder into a single CSV file
    
    Args:
        folder_path: Path to folder containing CPF .txt files
        output_csv_path: Path where merged CSV will be saved
    
    Returns:
        DataFrame with merged CPF data
    """
    
    print("=" * 70)
    print("LAGEOS-1 CPF FILE PARSER")
    print("=" * 70)
    print(f"\nSearching for CPF files in: {folder_path}\n")
    
    # Find all .txt files in folder
    all_files = glob.glob(os.path.join(folder_path, "*.hts"))
    
    if len(all_files) == 0:
        print("⚠️  No .hts files found in the specified folder!")
        return pd.DataFrame()
    
    print(f"Found {len(all_files)} CPF file(s):\n")
    
    dfs = []
    total_records = 0
    
    # Parse each file
    for idx, file in enumerate(all_files, 1):
        filename = os.path.basename(file)
        
        try:
            df = parse_cpf_file(file)
            
            if len(df) > 0:
                dfs.append(df)
                total_records += len(df)
                print(f"  [{idx:2d}] ✓ {filename:<40} → {len(df):5d} records")
            else:
                print(f"  [{idx:2d}] ✗ {filename:<40} → No valid records")
        
        except Exception as e:
            print(f"  [{idx:2d}] ✗ {filename:<40} → Error: {e}")
    
    if len(dfs) == 0:
        print("\n⚠️  No data could be parsed from any files!")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    print(f"\n{'─' * 70}")
    print("Merging and cleaning data...")
    
    cpf_merged = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (same epoch)
    initial_count = len(cpf_merged)
    cpf_merged = cpf_merged.drop_duplicates(subset=['epoch_datetime'], keep='first')
    duplicates_removed = initial_count - len(cpf_merged)
    
    # Sort by time
    cpf_merged = cpf_merged.sort_values('epoch_datetime').reset_index(drop=True)
    
    # Save to CSV
    cpf_merged.to_csv(output_csv_path, index=False)
    
    # Summary statistics
    print(f"{'─' * 70}")
    print("\n📊 SUMMARY:")
    print(f"  • Total records parsed:     {total_records:,}")
    print(f"  • Duplicates removed:       {duplicates_removed:,}")
    print(f"  • Final unique records:     {len(cpf_merged):,}")
    print(f"\n⏰ TIME SPAN:")
    print(f"  • Start: {cpf_merged['epoch_datetime'].min()}")
    print(f"  • End:   {cpf_merged['epoch_datetime'].max()}")
    print(f"  • Duration: {(cpf_merged['epoch_datetime'].max() - cpf_merged['epoch_datetime'].min()).days} days")
    
    # Data quality check
    print(f"\n🔍 DATA QUALITY:")
    null_counts = cpf_merged.isnull().sum()
    if null_counts.sum() == 0:
        print("  ✓ No missing values detected")
    else:
        print("  ⚠️  Missing values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"     - {col}: {count}")
    
    # Position and velocity ranges
    print(f"\n📍 POSITION RANGE (km):")
    print(f"  • X: [{cpf_merged['X'].min():,.2f}, {cpf_merged['X'].max():,.2f}]")
    print(f"  • Y: [{cpf_merged['Y'].min():,.2f}, {cpf_merged['Y'].max():,.2f}]")
    print(f"  • Z: [{cpf_merged['Z'].min():,.2f}, {cpf_merged['Z'].max():,.2f}]")
    
    print(f"\n🚀 VELOCITY RANGE (km/s):")
    print(f"  • VX: [{cpf_merged['VX'].min():.4f}, {cpf_merged['VX'].max():.4f}]")
    print(f"  • VY: [{cpf_merged['VY'].min():.4f}, {cpf_merged['VY'].max():.4f}]")
    print(f"  • VZ: [{cpf_merged['VZ'].min():.4f}, {cpf_merged['VZ'].max():.4f}]")
    
    # Orbital parameters
    import numpy as np
    radii = np.sqrt(cpf_merged['X']**2 + cpf_merged['Y']**2 + cpf_merged['Z']**2)
    velocities = np.sqrt(cpf_merged['VX']**2 + cpf_merged['VY']**2 + cpf_merged['VZ']**2)
    
    print(f"\n🛰️  ORBITAL PARAMETERS:")
    print(f"  • Mean radius:    {radii.mean():,.2f} km")
    print(f"  • Radius range:   [{radii.min():,.2f}, {radii.max():,.2f}] km")
    print(f"  • Mean velocity:  {velocities.mean():.4f} km/s")
    print(f"  • Velocity range: [{velocities.min():.4f}, {velocities.max():.4f}] km/s")
    
    print(f"\n{'─' * 70}")
    print(f"✅ Merged CPF data saved to:")
    print(f"   {output_csv_path}")
    print(f"{'=' * 70}\n")
    
    return cpf_merged


def display_sample_data(df, n_samples=5):
    """Display sample records from the dataframe"""
    
    print("\n📋 SAMPLE DATA (first {} records):".format(n_samples))
    print("─" * 70)
    
    # Select columns to display
    display_cols = ['epoch_datetime', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
    
    print(df[display_cols].head(n_samples).to_string(index=False))
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    # Path to folder containing CPF files
    CPF_FOLDER = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_files"
    
    # Output path for merged CSV
    OUTPUT_CSV = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_truth_merged_final.csv"
    
    # ========================================================================
    # RUN PARSER
    # ========================================================================
    
    # Parse and merge all CPF files
    cpf_data = merge_all_cpf_files(CPF_FOLDER, OUTPUT_CSV)
    
    # Display sample data
    if len(cpf_data) > 0:
        display_sample_data(cpf_data, n_samples=10)
        
        # Additional validation
        print("✅ SUCCESS! CPF data is ready for ML model training.")
        print(f"\nNext steps:")
        print(f"  1. Use the merged CSV file: {OUTPUT_CSV}")
        print(f"  2. Run the main ANN training script with your TLE data")
        print(f"  3. The ML model will use this CPF data as 'ground truth'")
    else:
        print("❌ ERROR: No data was parsed. Please check your CPF files.")