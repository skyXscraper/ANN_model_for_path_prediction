import glob

files = glob.glob(r"C:\Users\Tanvi\Documents\Space_paper\lageos-1\cpf_files\*.hts")
print(f"Found {len(files)} CPF files")
if files:
    print("Example lines from first file:\n")
    with open(files[0]) as f:
        for _ in range(10):
            print(f.readline().rstrip())
