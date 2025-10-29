import re
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd

# ---------------------------
# Helper parsers & utilities
# ---------------------------

MU_EARTH = 398600.4418  # km^3 / s^2
EARTH_RADIUS_KM = 6378.1363

def tle_checksum_ok(line: str) -> bool:
    """Compute TLE checksum using the standard algorithm for columns 1-68 and compare to last digit."""
    # Last character expected to be checksum digit
    line68 = line.rstrip('\n')
    if len(line68) < 69:
        # pad/truncate to 69 chars to be safe
        line68 = line68.ljust(69)
    target = line68[68]
    tot = 0
    for ch in line68[:68]:
        if ch.isdigit():
            tot += int(ch)
        elif ch == '-':
            tot += 1
    calc = tot % 10
    try:
        return calc == int(target)
    except Exception:
        return False

def parse_bstar(raw: str) -> float:
    """Parse BSTAR/second-derivative style fields like '24098-4' or ' 00000+0' to float."""
    s = raw.strip()
    if not s or re.fullmatch(r'[+\- ]*0+([+\-]0)?', s):
        return 0.0
    # remove spaces
    s = s.replace(' ', '')
    # find mantissa and exponent
    m = re.match(r'([+\-]?\d+)([+\-]\d+)', s)
    if not m:
        # fallback: try to parse as float
        try:
            return float(s)
        except:
            return 0.0
    mant = int(m.group(1))
    exp = int(m.group(2))
    # The TLE encoded value = mantissa * 10^(exp-5)
    val = mant * (10 ** (exp - 5))
    return float(val)

def parse_epoch(epoch_year_two: int, epoch_day: float) -> datetime:
    """Convert TLE epoch (YY, day-of-year.fraction) to UTC datetime."""
    yy = int(epoch_year_two)
    if yy >= 57:
        year = 1900 + yy
    else:
        year = 2000 + yy
    day_whole = int(math.floor(epoch_day))
    day_fraction = epoch_day - day_whole
    # day_of_year counts from 1 => day_whole==1 means Jan 1
    dt = datetime(year, 1, 1) + timedelta(days=day_whole - 1, seconds=day_fraction * 86400.0)
    return dt

def semi_major_axis_from_mean_motion(mean_motion_rev_per_day: float) -> float:
    """Compute semi-major axis (km) from mean motion (rev/day)."""
    n_rad_s = mean_motion_rev_per_day * 2 * math.pi / 86400.0
    a = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)
    return a

def parse_tle_pair(line1: str, line2: str) -> dict:
    """Parse one TLE pair (two strings) into a dict of fields (raw + derived)."""
    # Basic sanity: ensure lines start with 1 and 2
    if not line1.startswith('1') or not line2.startswith('2'):
        raise ValueError("Lines must start with '1' and '2'")
    # Validate checksum (we'll keep record even if checksum fails but flag it)
    chk1 = tle_checksum_ok(line1)
    chk2 = tle_checksum_ok(line2)

    # Parse fields by fixed-column positions per the TLE spec
    # Line 1 (columns are 1-indexed in spec; Python slices are zero-indexed)
    satnum = line1[2:7].strip()
    classification = line1[7].strip()
    int_desig_year = line1[9:11].strip()
    int_desig_launch = line1[11:14].strip()
    int_desig_piece = line1[14:17].strip()
    epoch_yy = line1[18:20].strip()
    epoch_day = line1[20:32].strip()
    ndot = line1[33:43].strip()       # first derivative of mean motion
    ndot2 = line1[44:52].strip()      # second derivative (in exponent format sometimes)
    bstar = line1[53:61].strip()      # BSTAR
    ephemeris_type = line1[62].strip()
    element_set = line1[64:68].strip()
    checksum1 = line1[68].strip()

    # Line 2
    satnum2 = line2[2:7].strip()
    incl = line2[8:16].strip()
    raan = line2[17:25].strip()
    ecc = line2[26:33].strip()
    argp = line2[34:42].strip()
    mna = line2[43:51].strip()
    mm = line2[52:63].strip()
    revnum = line2[63:68].strip()
    checksum2 = line2[68].strip()

    # Convert/clean fields
    try:
        epoch_day_f = float(epoch_day)
    except:
        epoch_day_f = np.nan

    try:
        epoch_yy_i = int(epoch_yy)
    except:
        epoch_yy_i = np.nan

    # parse numeric fields (use safe conversions)
    def safe_float(s):
        try:
            return float(s)
        except:
            return np.nan

    first_derivative = safe_float(ndot.replace(' ', '').replace('+', ''))
    # second derivative is often in special format (e.g., '00000-0')
    second_derivative = 0.0
    if ndot2:
        # convert if looks like exponent style (e.g. 00000-0 -> 0.0)
        m2 = re.match(r'\s*([+\-]?\d+)([+\-]\d+)', ndot2)
        if m2:
            mant = int(m2.group(1))
            exp = int(m2.group(2))
            second_derivative = mant * (10 ** (exp - 5))
        else:
            second_derivative = safe_float(ndot2)

    bstar_val = parse_bstar(bstar)

    inclination = safe_float(incl)
    raan_val = safe_float(raan)
    ecc_val = np.nan
    if ecc:
        # eccentricity in line2 has implied decimal and 7 digits total e.g., 0044362 -> 0.0044362
        try:
            ecc_val = float(ecc) * 1e-7
        except:
            ecc_val = np.nan
    argp_val = safe_float(argp)
    mean_anomaly = safe_float(mna)
    mean_motion = safe_float(mm)

    # Derived:
    epoch_dt = None
    if not np.isnan(epoch_yy_i) and not np.isnan(epoch_day_f):
        try:
            epoch_dt = parse_epoch(epoch_yy_i, epoch_day_f)
        except Exception:
            epoch_dt = None

    semi_major_axis = np.nan
    orbital_period_min = np.nan
    mean_motion_rad_s = np.nan
    perigee_alt_km = np.nan
    apogee_alt_km = np.nan
    if not np.isnan(mean_motion):
        try:
            semi_major_axis = semi_major_axis_from_mean_motion(mean_motion)
            mean_motion_rad_s = mean_motion * 2 * math.pi / 86400.0
            orbital_period_min = 1440.0 / mean_motion
            if not np.isnan(ecc_val):
                perigee_radius = semi_major_axis * (1.0 - ecc_val)
                apogee_radius = semi_major_axis * (1.0 + ecc_val)
                perigee_alt_km = perigee_radius - EARTH_RADIUS_KM
                apogee_alt_km = apogee_radius - EARTH_RADIUS_KM
        except Exception:
            pass

    parsed = {
        "satellite_number": satnum,
        "satellite_number_line2": satnum2,
        "classification": classification,
        "int_designator_year": int_desig_year,
        "int_designator_launch": int_desig_launch,
        "int_designator_piece": int_desig_piece,
        "epoch_year_two": epoch_yy_i,
        "epoch_day": epoch_day_f,
        "epoch_datetime": epoch_dt,
        "first_derivative_mean_motion": first_derivative,
        "second_derivative_mean_motion": second_derivative,
        "bstar": bstar_val,
        "ephemeris_type": ephemeris_type,
        "element_set_number": element_set,
        "inclination_deg": inclination,
        "raan_deg": raan_val,
        "eccentricity": ecc_val,
        "arg_perigee_deg": argp_val,
        "mean_anomaly_deg": mean_anomaly,
        "mean_motion_rev_per_day": mean_motion,
        "rev_number_at_epoch": revnum,
        "checksum1_ok": chk1,
        "checksum2_ok": chk2,
        # derived
        "semi_major_axis_km": semi_major_axis,
        "mean_motion_rad_s": mean_motion_rad_s,
        "orbital_period_min": orbital_period_min,
        "perigee_alt_km": perigee_alt_km,
        "apogee_alt_km": apogee_alt_km
    }
    return parsed

# ---------------------------
# Main pipeline
# ---------------------------

def preprocess_tle_file(raw_tle_path: str, out_csv: str = "tles_cleaned.csv", drop_invalid=True):
    """
    raw_tle_path: path to a plain text file containing consecutive TLEs.
    Each TLE must be two lines (optionally with an object name line above).
    """
    # Read file and collect pairs
    with open(raw_tle_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip() != '']

    # Normalize lines: remove any pure name lines (not starting with '1 ' or '2 ')
    normalized = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('1 ') or ln.startswith('1'):
            # expect next line to be '2'
            if i+1 < len(lines) and (lines[i+1].startswith('2 ') or lines[i+1].startswith('2')):
                normalized.append((lines[i], lines[i+1]))
                i += 2
            else:
                # malformed pair -> skip this and advance
                i += 1
        else:
            # name or stray line -> skip and advance
            i += 1

    # Parse all pairs
    parsed_list = []
    for l1, l2 in normalized:
        try:
            parsed = parse_tle_pair(l1, l2)
            parsed['raw_line1'] = l1
            parsed['raw_line2'] = l2
            parsed_list.append(parsed)
        except Exception as e:
            # keep a small trace for debugging; skip bad ones
            parsed_list.append({
                "satellite_number": None,
                "error": str(e),
                "raw_line1": l1,
                "raw_line2": l2
            })

    df = pd.DataFrame(parsed_list)

    # -------------------
    # QUALITY ASSESSMENT
    # -------------------
    # Identify records missing critical fields
    critical_fields = ["satellite_number", "epoch_datetime", "mean_motion_rev_per_day", "inclination_deg"]
    missing_critical = df[critical_fields].isnull().any(axis=1).sum()
    total = len(df)
    print(f"Total raw TLE pairs parsed: {total}")
    print(f"Records with missing critical fields: {missing_critical}")

    # Remove malformed rows if requested
    if drop_invalid:
        before = len(df)
        # Drop rows with missing satellite_number or epoch or mean motion
        df = df.dropna(subset=["satellite_number", "epoch_datetime", "mean_motion_rev_per_day"]).reset_index(drop=True)
        after = len(df)
        print(f"Dropped {before-after} records with missing critical fields. Remaining: {after}")

    # Fix or drop physically impossible values
    # Eccentricity: must be in [0, 1) for bound orbits. Mark and drop or clip.
    invalid_ecc = df[(df['eccentricity'].notna()) & ((df['eccentricity'] < 0) | (df['eccentricity'] >= 1))]
    if len(invalid_ecc):
        print(f"Found {len(invalid_ecc)} eccentricity outliers. Setting to NaN.")
        df.loc[invalid_ecc.index, 'eccentricity'] = np.nan

    # Fill non-critical missing numbers sensibly
    # BSTAR: fill with 0.0 for missing
    if 'bstar' in df.columns:
        n_missing_bstar = df['bstar'].isna().sum()
        if n_missing_bstar:
            df['bstar'] = df['bstar'].fillna(0.0)
            print(f"Filled {n_missing_bstar} missing BSTAR values with 0.0")

    # Remove duplicates based on satellite_number + epoch_datetime
    if 'epoch_datetime' in df.columns:
        before_dup = len(df)
        df = df.drop_duplicates(subset=['satellite_number', 'epoch_datetime']).reset_index(drop=True)
        print(f"Removed {before_dup - len(df)} duplicate records based on (satellite_number, epoch_datetime)")

    # Final numeric conversions and checks
    numeric_cols = [
        'first_derivative_mean_motion', 'second_derivative_mean_motion', 'bstar',
        'inclination_deg', 'raan_deg', 'eccentricity', 'arg_perigee_deg',
        'mean_anomaly_deg', 'mean_motion_rev_per_day',
        'semi_major_axis_km', 'mean_motion_rad_s', 'orbital_period_min',
        'perigee_alt_km', 'apogee_alt_km'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Final drop of rows with NaN in derived essential fields
    df = df.dropna(subset=['semi_major_axis_km', 'mean_motion_rev_per_day']).reset_index(drop=True)

    # Add angle sin/cos features to avoid circular discontinuity
    df['inclination_rad'] = np.deg2rad(df['inclination_deg'])
    df['sin_inclination'] = np.sin(df['inclination_rad'])
    df['cos_inclination'] = np.cos(df['inclination_rad'])
    # RAAN wrap and sin/cos
    df['raan_rad'] = np.deg2rad(df['raan_deg'].fillna(0.0))
    df['sin_raan'] = np.sin(df['raan_rad'])
    df['cos_raan'] = np.cos(df['raan_rad'])

    # Save cleaned CSV
    df.to_csv(out_csv, index=False)
    print(f"Saved cleaned dataset to: {out_csv}")
    return df

# ---------------------------
# If run as script
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess raw TLE text file to cleaned CSV for ML."
    )
    # 1️⃣ Define a *variable name*, not the actual filename
    parser.add_argument("raw_tle_file", help="Path to raw TLE text file (e.g., lageos-1.txt)")
    parser.add_argument("--out", "-o", default="tles_cleaned.csv", help="Output CSV path")
    args = parser.parse_args()

    df_clean = preprocess_tle_file(args.raw_tle_file, args.out)
    print(f"Finished preprocessing. Rows in cleaned df: {len(df_clean)}")
