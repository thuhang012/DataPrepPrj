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
            .replace({88: 0, 99: np.nan, 77: np.nan})
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

def drop_questionnaire_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are related to questionnaire metadata (not actual health data).
    These include version, language, and other administrative fields.
    """
    metadata_cols = [
        "QSTVER",      # Questionnaire version
        "QSTLANG",     # Language of interview
    ]
    
    existing = [c for c in metadata_cols if c in df.columns]
    if existing:
        print(f"\nDropping {len(existing)} questionnaire metadata columns: {existing}")
        df = df.drop(columns=existing)
    
    return df


def drop_high_cardinality_text_columns(df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """
    Drop text/object columns that have more than the specified threshold 
    of unique values. These are typically free-text fields that are hard to model.
    """
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    high_card_text = [
        col for col in text_cols 
        if df[col].nunique() > threshold
    ]
    
    if high_card_text:
        print(f"\nDropping {len(high_card_text)} high-cardinality text columns (>{threshold} unique values):")
        for col in high_card_text:
            print(f"  - {col} ({df[col].nunique()} unique values)")
        df = df.drop(columns=high_card_text)
    
    return df


def fix_boolean_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix encoding issues in boolean columns:
    - Map 1 -> 1, 2 -> 0 (standard BRFSS Yes/No encoding)
    - Map scientific notation values (1.0e+00 -> 1, 5.4e-79 -> 0)
    
    Returns the dataframe with properly encoded boolean columns.
    """
    # Define columns with scientific notation issues
    float_encoded_cols = ['_FRTRESP', '_VEGRESP', '_FRT16', '_VEG23', 'PAMISS1_']
    
    # Identify all boolean columns (2 unique non-null values)
    boolean_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].dropna().nunique()
            if n_unique == 2:
                boolean_cols.append(col)
    
    # 1. Handle standard 1/2 encoded columns
    standard_bool_cols = [col for col in boolean_cols if col not in float_encoded_cols]
    if standard_bool_cols:
        print(f"\nMapping 1->0, 2->1 for {len(standard_bool_cols)} boolean columns")
        df[standard_bool_cols] = df[standard_bool_cols].replace({1: 1, 2: 0, 1.0: 1.0, 2.0: 0.0})
    
    # 2. Handle float-encoded columns (scientific notation)
    for col in float_encoded_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: 1 if x == 1 or x == 1.0 else
                            0 if (isinstance(x, float) and abs(x) < 1e-50) else x
            )
    
    print(f"✓ Fixed encoding for {len(boolean_cols)} boolean columns")
    return df

def feature_grouping(df_prep: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight feature cleaning and grouping:
    - Scale MET / VO2 / FC / PA frequency variables
    - Drop noisy / duplicate / out-of-scope columns
    - Aggregate difficulty indicators
    - Keep compact lifestyle blocks (BMI, SES, smoking, alcohol, diet)
    """
    df = df_prep.copy()
    df = df.replace("5.400000e-79", 0.0)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].mask(df[num_cols].abs() < 1e-50, 0.0)
    original_cols = set(df.columns)

    # ============================================================
    # 0. MET VALUES + PHYSIOLOGICAL METRICS + ACTIVITY FREQUENCY
    # ============================================================

    def process_met_values(series: pd.Series) -> pd.Series:
        """
        Convert stored MET values to true physiological values.
        - Coerce to float
        - Treat extremely small magnitudes as 0 (SAS floating artifact)
        - Apply one implied decimal place (divide by 10)
        """
        # Coerce to float, invalid/parsing errors -> NaN
        series_float = pd.to_numeric(series, errors="coerce")

        # Any tiny value (e.g. 5.400000e-79) is just a floating-point zero artifact
        tiny_mask = series_float.abs() < 1e-50
        series_float[tiny_mask] = 0.0

        # One implied decimal place
        return series_float / 10.0

    scaled_met_cols = []
    for col in ["METVL11_", "METVL21_"]:
        if col in df.columns:
            df[col] = process_met_values(df[col])
            scaled_met_cols.append(col)

    scaled_vo2_cols = []
    for col in ["MAXVO2_", "FC60_"]:
        if col in df.columns:
            df[col] = df[col].astype(float) / 100.0
            scaled_vo2_cols.append(col)

    scaled_freq_cols = []
    if "PAFREQ1_" in df.columns:
        df["PAFREQ1_"] = df["PAFREQ1_"].astype(float) / 1000.0
        scaled_freq_cols.append("PAFREQ1_")

    # ============================================================
    # 1. SIMPLE CLEANING — DROP NOISY / DUPLICATE COLUMNS
    # ============================================================

    cols_to_drop_simple = [
        "SMOKE100", "USENOW3", "EXERANY2", "SEATBELT", "HIVTST6",
        "PCDMDECN", "_RFHLTH", "_HCVU651", "_LTASTH1", "_CASTHM1",
        "_DRDXAR1",
        "_PRACE1", "_MRACE1", "_HISPANC", "_RACE", "_RACEG21",
        "_RACE_G1",
        "_AGEG5YR", "_AGE65YR", "_AGE_G",
        "VETERAN3", "INTERNET", "BLOODCHO",
    ]
    dropped_simple_present = [c for c in cols_to_drop_simple if c in df.columns]
    df = df.drop(columns=cols_to_drop_simple, errors="ignore")

    # ============================================================
    # 1.2 Aggregate physical difficulty indicators
    # ============================================================

    difficulty_cols = ["BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON"]
    existing_diff_cols = [c for c in difficulty_cols if c in df.columns]

    if existing_diff_cols:
        df["PHYSICAL_DIFFICULTY"] = df[existing_diff_cols].sum(axis=1)
        df = df.drop(columns=existing_diff_cols, errors="ignore")

    # ============================================================
    # 2. BEHAVIORAL GROUPING — BMI, SES, SMOKING, ALCOHOL, DIET
    # ============================================================

    # Body size: drop raw height/weight and numeric BMI,
    # keep _BMI5CAT and _RFBMI5
    body_drop = ["HTIN4", "HTM4", "WTKG3", "_BMI5"]
    dropped_body_present = [c for c in body_drop if c in df.columns]

    # Alcohol: keep DRNKANY5, _RFBING5, _RFDRHV5
    alcohol_drop = ["DROCDY3_", "_DRNKWEK"]
    dropped_alcohol_present = [c for c in alcohol_drop if c in df.columns]

    # Fruit & veg: keep _FRUTSUM, _VEGESUM, _FRTLT1, _VEGLT1
    diet_drop = [
        "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_",
        "VEGEDA1_", "_MISFRTN", "_MISVEGN", "_FRTRESP", "_VEGRESP",
        "_FRT16", "_VEG23", "_FRUITEX", "_VEGETEX",
    ]
    dropped_diet_present = [c for c in diet_drop if c in df.columns]

    # Apply behavioral drops
    cols_to_drop_behavior = body_drop + alcohol_drop + diet_drop
    cols_to_drop_behavior = [c for c in cols_to_drop_behavior if c in df.columns]
    df = df.drop(columns=cols_to_drop_behavior)

    # ============================================================
    # 3. SHORT TEXT REPORT
    # ============================================================

    print("\n================ Feature grouping / cleaning report ================\n")

    if scaled_met_cols:
        print(f"✓ Scaled MET variables (÷10, fix 5.4e-79): {scaled_met_cols}")
    if scaled_vo2_cols:
        print(f"✓ Scaled VO2 / functional capacity (÷100): {scaled_vo2_cols}")
    if scaled_freq_cols:
        print(f"✓ Scaled PA frequency (÷1000): {scaled_freq_cols}")

    if existing_diff_cols:
        print(f"✓ Combined difficulty indicators {existing_diff_cols} → 'PHYSICAL_DIFFICULTY'")

    if dropped_simple_present:
        print(f"✓ Dropped admin / duplicate / out-of-scope columns: {dropped_simple_present}")

    if dropped_body_present:
        print(f"✓ Dropped raw body-size columns: {dropped_body_present} "
              f"(keep _BMI5CAT and _RFBMI5)")

    if dropped_alcohol_present:
        print(f"✓ Dropped detailed alcohol volume columns: {dropped_alcohol_present} "
              f"(keep DRNKANY5, _RFBING5, _RFDRHV5)")

    if dropped_diet_present:
        print(f"✓ Dropped detailed fruit/veg and quality flags: {dropped_diet_present} "
              f"(keep _FRUTSUM, _VEGESUM, _FRTLT1, _VEGLT1)")

    print("\n===================================================================\n")

    return df

def classify_variables(df: pd.DataFrame) -> tuple:
    """
    Classify columns into numeric, categorical, and boolean types.
    
    Uses domain knowledge to correctly classify BRFSS variables:
    - True numeric: counts and days where magnitude matters (PHYSHLTH, MENTHLTH, POORHLTH, CHILDREN)
    - Ordinal: ordered categories like health status, education, income (keep as categorical)
    - Nominal: unordered categories like race groups (keep as categorical)
    - Boolean: binary 0/1 variables
    
    Returns:
        tuple: (numeric_cols, categorical_cols, boolean_cols)
    """
    numeric_vars = []
    categorical_vars = []
    boolean_vars = []
    
    # Define true numeric columns (counts, days where magnitude matters)
    TRUE_NUMERIC = [
        'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'CHILDREN',
        # Calculated numeric variables
        '_DRNKWEK', 'DROCDY3_', 
        # Weight, height, BMI continuous values
        'WTKG3', 'HTIN4', 'HTM4', '_BMI5',
        # Age continuous (if exists)
        '_AGE80',
        # Fruit/vegetable servings per day
        'FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_',
        '_FRUTSUM', '_VEGESUM',
        # Physical activity minutes/METs
        'METVL11_', 'METVL21_', 'MAXVO2_', 'FC60_',
        'ACTIN11_', 'ACTIN21_', 'PADUR1_', 'PAFREQ1_', 'PAMIN11_', 'PA1MIN_',
        'PAVIG11_', 'PA1VIGM_', '_MINAC11', '_MINAC21',
    ]
    
    # Define ordinal/nominal categories (keep as categorical even if numeric codes)
    ORDINAL_NOMINAL = [
        # Ordinal health/socioeconomic
        'GENHLTH', 'EDUCA', 'SEATBELT', 'CHECKUP1', 'INCOME2', 'EMPLOY1',
        'PERSDOC2', 'CHOLCHK', 'DIABETE3', 'MARITAL', 'SMOKDAY2', 'USENOW3',
        'TRNSGNDR',
        # Age/BMI/Education/Income bins (ordinal categories)
        '_AGEG5YR', '_BMI5CAT', '_EDUCAG', '_INCOMG', '_AGE_G', '_AGE65YR',
        # Race groups (nominal)
        '_PRACE1', '_MRACE1', '_HISPANC', '_RACE', '_RACEG21', '_RACEGR3', '_RACE_G1',
        'SEX', '_CHLDCNT',
        # Physical activity categories
        '_PACAT1', '_PA150R2', '_PA300R2', '_PA30021',
        # Smoking categories
        '_SMOKER3',
        # Other calculated categories
        'IMFVPLAC', 'BPHIGH4',
    ]

    for col in df.columns:
        # Skip object/categorical types (text)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_vars.append(col)
            continue

        valid_series = df[col].dropna()
        n_unique = valid_series.nunique()

        # Boolean: exactly 2 unique values
        if n_unique == 2:
            boolean_vars.append(col)
            continue
        
        # Force true numeric columns
        if col in TRUE_NUMERIC:
            numeric_vars.append(col)
            continue
        
        # Force ordinal/nominal as categorical
        if col in ORDINAL_NOMINAL:
            categorical_vars.append(col)
            continue

        # For remaining columns, use heuristics
        # Numeric: has decimal values
        if pd.api.types.is_float_dtype(df[col]) and (valid_series % 1 != 0).any():
            numeric_vars.append(col)
            continue

        # Default: high cardinality = numeric, low = categorical
        if n_unique > 50:
            numeric_vars.append(col)
        else:
            categorical_vars.append(col)

    print(f"\n=== Variable Classification ===")
    print(f"Numeric: {len(numeric_vars)}")
    print(f"Categorical: {len(categorical_vars)}")
    print(f"Boolean: {len(boolean_vars)}")
    
    return numeric_vars, categorical_vars, boolean_vars


# def build_and_save_preprocessed(threshold: float = 0.5, filename: str = "clean_2015.csv") -> pd.DataFrame:
#     """
#     Run the entire preprocessing pipeline and save the final dataset
#     to data/processed/<filename>. Returns the dataframe.
    
#     Pipeline steps:
#     1. Load raw data
#     2. Apply codebook cleaning (missing value mappings)
#     3. Drop raw columns when calculated versions exist
#     4. Drop high-missing columns
#     5. Drop questionnaire metadata columns
#     6. Drop high-cardinality text columns
#     7. Fix boolean encoding (1/2 -> 0/1)
#     8. Classify variables into numeric/categorical/boolean
#     """
#     df = build_preprocessed_dataset(threshold=threshold)
    
#     # Additional preprocessing steps
#     df = drop_questionnaire_metadata_columns(df)
#     df = drop_high_cardinality_text_columns(df, threshold=50)
#     df = fix_boolean_encoding(df)
    
#     # Classify variables
#     numeric_cols, categorical_cols, boolean_cols = classify_variables(df)
    
#     # Save the final preprocessed dataframe
#     save_preprocessed_csv(df, filename)
    
#     return df