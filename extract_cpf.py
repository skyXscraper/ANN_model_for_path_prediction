import pandas as pd
import glob
from datetime import datetime

def parse_cpf_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('10'):
                parts = line.split()
                year = int(parts[1])
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                second = float(parts[6])
                x = float(parts[7])
                y = float(parts[8])
                z = float(parts[9])
                epoch = datetime(year, month, day, hour, minute, int(second))
                data.append([epoch, x, y, z])
    return pd.DataFrame(data, columns=['epoch_datetime', 'x', 'y', 'z'])

# 🛰 Merge all LAGEOS-1 CPF files
folder = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_files"
all_files = glob.glob(folder + "/*.hts")

dfs = []
for file in all_files:
    try:
        df = parse_cpf_file(file)
        dfs.append(df)
    except Exception as e:
        print("Error parsing", file, e)

cpf_truth = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['epoch_datetime'])
cpf_truth = cpf_truth.sort_values('epoch_datetime').reset_index(drop=True)

# Save
out_csv = r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_truth.csv"
cpf_truth.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(cpf_truth.head())
