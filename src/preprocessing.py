import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import load_data

# ================================
# 1. Load raw + prune columns
# ================================

def load_raw_2015() -> pd.DataFrame:
    """
    Load the pre-pruned BRFSS 2015 dataset from 2015_raw.csv.
    This file is assumed to already contain only the selected columns.
    """
    df = load_data("2015_raw.csv", subfolder="raw")
    return df.copy()

# ================================
# 2. Mapping 7/9/77/99/555/888... -> NaN/0
# ================================

def apply_codebook_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply BRFSS codebook rules to replace coded missing values 
    (e.g., 7/9/77/99/555/777/999...) with NaN or 0 depending on variable type.
    This converts the encoded missing values into true missing values for analysis.
    """
    TINY = 5.397605346934028e-79

    df_clean_2 = df.copy()

    # === 1. Định nghĩa các nhóm cột theo pattern giống nhau ===

    # 1.1 Yes/No: 1–2 hợp lệ, 7/9 = missing
    yesno_cols = [
        "HLTHPLN1", "MEDCOST", "BLOODCHO", "TOLDHI2",
        "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3",
        "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3",
        "ADDEPEV2", "CHCKIDNY", "VETERAN3", "INTERNET",
        "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE",
        "DIFFWALK", "DIFFDRES", "DIFFALON", "EXERANY2",
        "LMTJOIN3", "ARTHDIS2", "FLUSHOT6",
        "PNEUVAC3", "HIVTST6", "BPMEDS",
        "SMOKE100", "DRNKANY5", "_FRTLT1", "_VEGLT1"
    ]

    # 1.2 Multi-level: 7/8/9 = missing
    multi_cat_cols = [
        "GENHLTH", "CHECKUP1", "PERSDOC2", "CHOLCHK" , "DIABETE3",
        "MARITAL", "EDUCA", "ARTHSOCL", "SMOKDAY2", "USENOW3",
        "SEATBELT", "TRNSGNDR", "BPHIGH4"
    ]

    # 1.3 Cột có 77/99 = missing
    code77_99_cols = [
        "INCOME2", "AVEDRNK2", "MAXDRNKS",
        "JOINPAIN", "IMFVPLAC", "_PRACE1", "_MRACE1"
    ]

    # 1.4 Calc/flag _xxx: 9 = missing
    flag_9_cols = [
        "_CHISPNC", "_RFHLTH", "_HCVU651", "_RFHYPE5",
        "_CHOLCHK", "_RFCHOL", "_LTASTH1",
        "_CASTHM1", "_ASTHMS1", "_HISPANC", "_RACE",
        "_RFBMI5", "_CHLDCNT", "_EDUCAG", "_INCOMG", "_SMOKER3",
        "_RFSMOK3", "_RFBING5", "_RFDRHV5", "_TOTINDA",
        "PAMISS1_", "_PACAT1", "_PAINDX1", "_PA150R2",
        "_PA300R2",
        "_RACEG21", "_RACEGR3",
        "_PA30021", "_PASTRNG", "_PAREC1", "_PASTAE1",
        "_LMTACT1", "_LMTWRK1", "_LMTSCL1", "_RFSEAT2",
        "_RFSEAT3", "_FLSHOT6", "_PNEUMO2", "_AIDTST3"
    ]

    # 1.5 Nhóm frequency: 555 = 0, 777/999 = NaN
    frequency_cols = [
        "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN",
        "FVORANG", "VEGETAB1", "EXEROFT1", "EXEROFT2", "STRENGTH"
    ]

    # 1.6 Nhóm EXERHMM/EXEROFT chỉ có 777/999 = NaN
    exercise_missing_cols = [
        "EXERHMM1", "EXERHMM2", "EXEROFT1", "EXEROFT2"
    ]

    # 1.7 Nhóm “days 0–30, 88 = 0, 77/99 = NaN”
    days_0_30_cols = ["PHYSHLTH", "MENTHLTH", "POORHLTH"]

    # 1.8 Nhóm loại hình hoạt động (77/99 = NaN)
    activity_type_cols = ["EXRACT11", "EXRACT21"]

    # === 2. Apply các pattern dùng vòng for ===

    pattern_replacements = [
        (yesno_cols,         {7: np.nan, 9: np.nan}),
        (multi_cat_cols,     {7: np.nan, 8: np.nan, 9: np.nan}),
        (code77_99_cols,     {77: np.nan, 99: np.nan}),
        (flag_9_cols,        {9: np.nan}),
        (frequency_cols,     {555: 0, 777: np.nan, 999: np.nan}),
        (exercise_missing_cols, {777: np.nan, 999: np.nan}),
        (days_0_30_cols,     {88: 0, 77: np.nan, 99: np.nan}),
        (activity_type_cols, {77: np.nan, 99: np.nan}),
    ]

    for cols, mapping in pattern_replacements:
        existing = [c for c in cols if c in df_clean_2.columns]
        if existing:
            df_clean_2[existing] = df_clean_2[existing].replace(mapping)

    # === 3. Các cột/logic đặc biệt (riêng từng biến) ===

    # 3.1 Số siêu nhỏ ~0 trong calc-var -> 0
    df_clean_2 = df_clean_2.replace({TINY: 0})

    # 3.2 WTKG3: weight (kg x 100), 99999 = missing
    if "WTKG3" in df_clean_2.columns:
        df_clean_2["WTKG3"] = df_clean_2["WTKG3"].replace({99999: np.nan})

    # 3.3 Các calc-var có 99900/99000 = missing
    for col, miss_code in {
        "_DRNKWEK": 99900,
        "MAXVO2_": 99900, 
        "FC60_": 99900,
        "PAFREQ1_": 99000,
        "PAFREQ2_": 99000,
        "STRFREQ_": 99000,
    }.items():
        if col in df_clean_2.columns:
            df_clean_2[col] = df_clean_2[col].replace({miss_code: np.nan})

    # 3.4 _AGE65YR: 3 = missing
    if "_AGE65YR" in df_clean_2.columns:
        df_clean_2["_AGE65YR"] = df_clean_2["_AGE65YR"].replace({3: np.nan})

    # 3.5 EMPLOY1: 1–8 hợp lệ, 9 = missing
    if "EMPLOY1" in df_clean_2.columns:
        df_clean_2["EMPLOY1"] = df_clean_2["EMPLOY1"].replace({9: np.nan})

    # 3.6 CHILDREN: 1–87, 88 = 0, 99 = missing
    if "CHILDREN" in df_clean_2.columns:
        df_clean_2["CHILDREN"] = (
            df_clean_2["CHILDREN"]
            .replace({88: 0, 99: np.nan})
            .astype("float")
        )

    # 3.7 WEIGHT2 & HEIGHT3: mã missing
    for col in ["WEIGHT2", "HEIGHT3"]:
        if col in df_clean_2.columns:
            df_clean_2[col] = df_clean_2[col].replace({7777: np.nan, 9999: np.nan})

    # 3.8 ALCDAY5: tần suất uống rượu 30 ngày
    if "ALCDAY5" in df_clean_2.columns:
        df_clean_2["ALCDAY5"] = df_clean_2["ALCDAY5"].replace({
            777: np.nan,
            999: np.nan,
            888: 0,   # No drinks in past 30 days
        })

    # 3.9 DRNK3GE5: số lần binge drinking
    if "DRNK3GE5" in df_clean_2.columns:
        df_clean_2["DRNK3GE5"] = df_clean_2["DRNK3GE5"].replace({
            88: 0,
            77: np.nan,
            99: np.nan
        })

    # 3.10 FLSHTMY2: tháng/năm tiêm flu shot
    if "FLSHTMY2" in df_clean_2.columns:
        df_clean_2["FLSHTMY2"] = df_clean_2["FLSHTMY2"].replace({
            777777: np.nan,
            999999: np.nan
        })

    # 3.11 DROCDY3_: drink-occasions-per-day (900 = missing)
    if "DROCDY3_" in df_clean_2.columns:
        df_clean_2["DROCDY3_"] = df_clean_2["DROCDY3_"].replace({900: np.nan})

    return df_clean_2

# ================================
# 3. Drop raw if calculated
# ================================

RAW_TO_CALC = {
    "WEIGHT2":  ["WTKG3", "_BMI5"],
    "HEIGHT3":  ["HTIN4", "HTM4", "_BMI5"],
    "ALCDAY5":  ["DROCDY3_", "_DRNKWEK", "_RFDRHV5", "_RFBING5"],
    "FRUITJU1": ["FTJUDA1_"],
    "FRUIT1":   ["FRUTDA1_"],
    "FVBEANS":  ["BEANDAY_"],
    "FVGREEN":  ["GRENDAY_"],
    "FVORANG":  ["ORNGDAY_"],
    "VEGETAB1": ["VEGEDA1_"],
    "EXEROFT1": ["PAFREQ1_"],
    "EXERHMM1": ["PADUR1_"],
    "EXEROFT2": ["PAFREQ2_"],
    "EXERHMM2": ["PADUR2_"],
    "STRENGTH": ["STRFREQ_", "_PASTRNG"],
}

# Một số biến gốc như WEIGHT2, HEIGHT3, ALCDAY5 đã được BRFSS 
# tổng hợp thành các biến tính sẵn như _BMI5, WTKG3, DRNKWEK, 
# STRFREQ.Khi các biến tính sẵn tồn tại, việc giữ lại biến gốc 
# sẽ làm trùng nghĩa, gây multicollinearity và khiến việc phân tích khó hơn.

def drop_raw_when_calc_exists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop original raw variables when BRFSS provides calculated/derived versions
    of the same information (e.g., WEIGHT2 vs WTKG3). This avoids redundancy
    and reduces multicollinearity.
    """
    df = df.copy()
    for raw_col, calc_cols in RAW_TO_CALC.items():
        if raw_col not in df.columns:
            continue
        if any(c in df.columns for c in calc_cols):
            df = df.drop(columns=[raw_col])
    return df

# ================================
# Check list cột high missing và drop
# ================================

def get_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> list:
    """
    Identify columns whose missing-value ratio exceeds the given threshold.
    Returns a list of column names for inspection before dropping.
    """
    missing_ratio = df.isna().mean()

    high_missing_cols = (
        missing_ratio[missing_ratio > threshold]
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    return high_missing_cols

def drop_columns(
    df: pd.DataFrame,
    cols_to_drop: list,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drop the specified columns from the dataframe.
    Only columns that actually exist in the dataframe are removed.
    Optionally prints the dropped columns.
    """
    
    df = df.copy()
    existing = [c for c in cols_to_drop if c in df.columns]

    if verbose and existing:
        print(f"Dropping {len(existing)} columns:")
        for col in existing:
            print(f"  - {col}")

    return df.drop(columns=existing)


# ==========================================

def build_preprocessed_dataset(
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Pipeline preprocessing:
    - Load
    - Map codebook
    - Drop raw if calc exists
    - Detect & drop high-missing (> threshold)
    """
    df = load_raw_2015()
    df = apply_codebook_cleaning(df)
    df = drop_raw_when_calc_exists(df)

    # Detect 
    high_missing_cols = get_high_missing_columns(df, threshold)
    print("\n=== High-missing columns (> {}): ===".format(threshold))
    print(high_missing_cols)

    # high_missing_cols.remove("some_column_you_want_to_keep")

    # Drop
    df = drop_columns(df, high_missing_cols)

    return df

# ================================

def save_preprocessed_csv(df: pd.DataFrame, filename: str = "clean_2015.csv") -> Path:
    """
    Save the preprocessed dataframe to data/processed/<filename>.
    Creates the directory if it does not exist.
    Returns the full output file path.
    """
    project_root = Path(__file__).resolve()
    for _ in range(6):
        if (project_root / "data").exists():
            break
        project_root = project_root.parent

    save_path = project_root / "data" / "processed"
    save_path.mkdir(parents=True, exist_ok=True)

    full_path = save_path / filename
    df.to_csv(full_path, index=False)

    print(f"Saved cleaned dataset to: {full_path}")
    return full_path

def build_and_save_preprocessed(threshold: float = 0.5, filename: str = "clean_2015.csv") -> pd.DataFrame:
    """
    Run the entire preprocessing pipeline and save the final dataset
    to data/processed/<filename>. Returns the dataframe.
    """
    df = build_preprocessed_dataset(threshold=threshold)
    save_preprocessed_csv(df, filename)
    return df