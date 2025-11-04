import pandas as pd
import glob
from datetime import datetime, timedelta
import os

def mjd_to_datetime(mjd, seconds_of_day):
    """Convert Modified Julian Date to datetime"""
    mjd_epoch = datetime(1858, 11, 17)
    return mjd_epoch + timedelta(days=int(mjd), seconds=seconds_of_day)

def parse_hts_file(filepath):
    """Parse old CPF .hts file containing only '10' position records"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == '10':
                try:
                    mjd = int(parts[2])
                    seconds = float(parts[3])
                    x_m = float(parts[5])
                    y_m = float(parts[6])
                    z_m = float(parts[7])
                    epoch = mjd_to_datetime(mjd, seconds)
                    data.append([epoch, x_m / 1000.0, y_m / 1000.0, z_m / 1000.0])
                except Exception:
                    continue
    return pd.DataFrame(data, columns=['epoch_datetime', 'X_km', 'Y_km', 'Z_km'])

def merge_hts_folder(folder, output_csv):
    all_files = glob.glob(os.path.join(folder, "*.hts"))
    dfs = []
    for fpath in all_files:
        df = parse_hts_file(fpath)
        if len(df) > 0:
            dfs.append(df)
            print(f"Parsed {len(df)} records from {os.path.basename(fpath)}")
        else:
            print(f"No valid records in {os.path.basename(fpath)}")

    if not dfs:
        print("⚠️ No data found in any file.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=['epoch_datetime']).sort_values('epoch_datetime')
    merged.to_csv(output_csv, index=False)
    print(f"\n✅ Saved merged CPF data to: {output_csv}")
    print(f"Total records: {len(merged)}")
    print(f"Time span: {merged['epoch_datetime'].min()} → {merged['epoch_datetime'].max()}")
    return merged

if __name__ == "__main__":
    folder = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_files"
    out_csv = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_truth_positions.csv"
    merge_hts_folder(folder, out_csv)
