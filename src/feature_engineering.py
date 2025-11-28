from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union


# Hanle missing logic


def handle_missing(df: pd.DataFrame, impute_medium_cols: bool = False) -> pd.DataFrame:
    """
    Applies missingness strategies (Flags, Median Imputation, Zero/Mode Imputation) 
    to features based on the mechanism and magnitude of missingness.
    
    CRITICAL: This function must be run AFTER feature creation and capping. 
    Assumes data is clean of BRFSS special codes (77, 88, 99).
    """
    df = df.copy()

    print("--- 1. Flagging Medium Missingness (NMAR/MNAR) ---")
    # Strategy A: Missing Indicator Flags (Medium Missingness: 9%-18%)
    medium_nan_raw_cols = ["INCOME2", "CHOLCHK", "PNEUVAC3", "FLUSHOT6"]
    
    for raw_col in medium_nan_raw_cols:
        if raw_col in df.columns:
            flag_name = f"{raw_col}_Missing_flag"
            df[flag_name] = df[raw_col].isna().astype("float")
            print(f" - Created flag {flag_name}")

    print("\n--- 2. Imputation for Count Variables (Median) ---")
    # Strategy B: Median Imputation for Count/Continuous Variables (MAR assumption)
    
    # 2.1. Health Day Counts
    health_count_cols = ["PHYSHLTH", "MENTHLTH", "POORHLTH"] 
    
    # 2.2. Diet Sums (Mức thiếu 9-11%) - Thêm vào nhóm Impute bằng Median
    diet_sum_cols = ["_VEGESUM", "_FRUTSUM"]
    
    for col in health_count_cols + diet_sum_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f" - Imputed {col} with Median ({median_val:.2f})")
    
    # --- 3. Physical Activity (High Systematic Missingness: 33%-36%) ---
    print("\n--- 3. Handling Physical Activity (NMAR - Systematic) ---")
    
    pa_detailed_raw_cols = [
        "PAMIN11_","PAMIN21_", "PAVIG21_", "PADUR1_", "PAFREQ1_",
        "_MINAC21", "_MINAC11", "ACTIN21_", "ACTIN11_", 
        "METVL21_", "METVL11_" # Thêm các biến PA dẫn xuất khác
    ]
    pa_present = [c for c in pa_detailed_raw_cols if c in df.columns]
            
    # Sub-Strategy C.1: Create Group Missing Flag 
    if pa_present:
        # SỬA LỖI: Dùng .any(axis=1) để gắn cờ 1.0 nếu BẤT KỲ cột chi tiết nào bị thiếu.
        df["PA_Detail_Input_Missing_flag"] = df[pa_present].isna().any(axis=1).astype("float")
        print(f" - Created PA_Detail_Input_Missing_flag (using .any logic)")
    
    # Sub-Strategy C.2: Zero/One Imputation
    # Dùng list chứa cả biến thô và biến kỹ thuật liên quan đến hoạt động thể chất
    pa_features_to_impute = [
        "AnyAerobicActivity_flag", "AnyStrengthActivity_flag", 
        "Inactive_flag", "AnyExercise_flag"
    ] + pa_present
    
    for col in pa_features_to_impute:
        if col in df.columns and df[col].isna().any():
            missing_mask = df[col].isna()
            fill_val = 1.0 if "inactive" in col.lower() else 0.0
            df.loc[missing_mask, col] = fill_val
            print(f" - Imputed {col} with {fill_val} (Zero/One Imputation)")


    # --- 4. Chronic Disease Flags (Low Missingness: 0.01%-4%) ---
    print("\n--- 4. Handling Chronic Disease Flags (Low-nan) ---")
    # Strategy D: Mode Imputation (0.0) + Missing Flag
    low_nan_cols = [
        "Hypertension_flag", "Diabetes", "HadCHD", "HadStroke", "HadMI", 
        "AnyCVD_flag", "Obese_flag", "Underweight_flag", "OverweightOrObese_flag",
        "CurrentSmoker", "FormerSmoker", "NeverSmoker", "Asthma_Remission_flag", 
        "CurrentAsthma_flag", "EverAsthma_flag", "COPD_flag", 
        "QLACTLM2", "USEEQUIP", "FLUSHOT6", "PNEUVAC3" 
    ]
    
    for col in low_nan_cols:
        if col in df.columns and df[col].isna().any():
            missing_mask = df[col].isna()
            
            # 1. Create Flag
            df[f"{col}_Missing_flag"] = missing_mask.astype("float")
            # 2. Impute with Mode (0.0) - Conservative
            df.loc[missing_mask, col] = 0.0
            print(f" - Imputed {col} with 0.0 and created missing flag.")
    
    # --- 5. Optional Impute for Raw Features (suitable for NN ỏ Linear Model) ---
    if impute_medium_cols:
        print("\n--- 5. Imputing Medium-nan Raw Features (Optional) ---")
        # Thay thế NaN cho các biến thô đã được gắn cờ (để mô hình có thể xử lý)
        for col in medium_nan_raw_cols:
            if col in df.columns:
                # Dùng giá trị mới (-1) để tạo ra một loại "hạng mục thiếu"
                df[col] = df[col].fillna(-1).astype("float")
                print(f" - Imputed raw {col} with -1.")

    return df

# Outlier handling
def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers based on:
      - Domain knowledge (WHO, BRFSS, AHA…)
      - Winsorization 99.9% for continuous skewed vars
      - Not touching binary / categorical
    """
    df = df.copy()
    cols = set(df.columns)

    # 1. HEALTH DAYS (BRFSS official max 30 days)
    health_days = ["PHYSHLTH", "MENTHLTH", "POORHLTH"]
    for col in health_days:
        if col in cols:
            df[col] = df[col].clip(0, 30)

    # 2. FRUIT / VEGETABLE SERVINGS 
    FOOD_CAP = 20  # reasonable upper bound 
    food_vars = ["_VEGESUM", "_FRUTSUM"]

    for col in food_vars:
        if col in cols:
            df[col] = df[col].clip(upper=FOOD_CAP)

    # 3. PHYSICAL ACTIVITY MINUTES – WHO physiology constraint
    pa_minutes = [
        "PAMIN21_", "PAMIN11_", "_MINAC21", "_MINAC11",
        "PAVIG21_", "PAVIG11_", "PADUR1_"
    ]
    PA_MAX = 2100  # WHO extreme physiological boundary

    for col in pa_minutes:
        if col in cols:
            if df[col].notna().any():
                q999 = df[col].quantile(0.999)
                ub = min(q999, PA_MAX)
                df[col] = df[col].clip(upper=ub)

    # 4. STRENGTH TRAINING FREQUENCY – AHA guidelines
    if "STRFREQ_" in cols:
        q999 = df["STRFREQ_"].quantile(0.999)
        # AHA: 2–3 days/week recommended → extreme bound set as 14/week (2 per day)
        ST_MAX = 14
        ub = min(q999, ST_MAX)
        df["STRFREQ_"] = df["STRFREQ_"].clip(upper=ub)

    other_numeric_cols = [
        "METVL21_", "METVL11_", "MAXVO2_", "FC60_", 
        "CHILDREN", "Age_x_CurrentSmoker"
    ]
    
    for col in other_numeric_cols:
        if col in cols and df[col].notna().any():
            q999 = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=q999)
            print(f" - Q99.9% Capped {col} at {q999:.2f}")

    return df

#=========================================
# AGE
#=========================================

def add_age_features(df):
    """
    Add age-based features from BRFSS variable `_AGE80`.
    Creates:
        - Age: numeric age
        - AgeGroup: categorical 18-24, 25-34, ..., 65+
        - Age_65plus: binary flag
        - AgeBucket: coarse age groups (Young, Middle, Senior)
    """
    df = df.copy()

    if "_AGE80" not in df.columns:
        print("[WARN] _AGE80 not found. Age features skipped.")
        return df
    
    print("\n" + "="*60)
    print("AGE FEATURES")
    print("="*60)

    # 1. Numeric age
    df["Age"] = pd.to_numeric(df["_AGE80"], errors="coerce")

    # 2. Age groups
    age_bins = [0, 25, 35, 45, 55, 65, 200]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        ordered=True
    )

    # 3. Age 65+ flag
    mask_age = df["Age"].isna()
    df["Age_65plus"] = (df["Age"] >= 65).astype("float")
    df.loc[mask_age, "Age_65plus"] = np.nan

    # AgeBucket
    df["AgeBucket"] = pd.cut(
        df["Age"],
        bins=[0, 45, 65, 200],
        labels=["Young", "Middle", "Senior"],
        right=False,
        ordered=True
    )

    return df

#===========================================
# SEX
#===========================================

def add_sex_features(df):
    """
    Create single binary sex feature for ML.
    Creates:
        - IsMale: 1 = male, 0 = female, NaN = missing
    """
    df = df.copy()

    if "SEX" not in df.columns:
        print("[WARN] SEX not found. Sex features skipped.")
        return df

    print("\n" + "="*40)
    print("SEX FEATURES")
    print("="*40)

    mask = df["SEX"].isna()
    # BRFSS: 1 = Male, 2 = Female
    df["IsMale"] = (df["SEX"] == 1).astype("float")
    df.loc[mask, "IsMale"] = np.nan

    return df

#===========================================
# RACE
#===========================================
def add_race_features(df):
    """
    Coding:
        1 = White only, Non-Hispanic
        2 = Black only, Non-Hispanic
        3 = Other race only, Non-Hispanic
        4 = Multiracial, Non-Hispanic
        5 = Hispanic
    Creates:
        - Race_White
        - Race_Black
        - Race_Hispanic
    Notes:
        - Other non-Hispanic (3,4) are implicit baseline (all flags = 0)
        - Keeps NaN for missing
    """
    df = df.copy()

    if "_RACEGR3" not in df.columns:
        print("[WARN] _RACEGR3 not found. Race features skipped.")
        return df

    print("\n" + "="*60)
    print("RACE FEATURES")
    print("="*60)

    mask = df["_RACEGR3"].isna()

    # Binary flags directly from numeric code
    df["Race_White"] = (df["_RACEGR3"] == 1).astype("float")
    df["Race_Black"] = (df["_RACEGR3"] == 2).astype("float")
    df["Race_Hispanic"] = (df["_RACEGR3"] == 5).astype("float")

    df.loc[mask, ["Race_White", "Race_Black", "Race_Hispanic"]] = np.nan

    # Codes 3 & 4 => Other non-Hispanic baseline (all 3 flags = 0)

    return df

def add_employ_features(df):
    """
    Add a useful employment-based flag for ML.
    Creates:
        - UnableToWork_flag: 1 if EMPLOY1 == 8, else 0 (NaN stays NaN)
    Notes:
        - Keep EMPLOY1 raw (to be encoded later in statistical FE).
        - This flag highlights a clinically important subgroup.
    """
    df = df.copy()

    if "EMPLOY1" not in df.columns:
        print("[WARN] EMPLOY1 not found. Employment feature skipped.")
        return df

    print("\n" + "="*60)
    print("EMPLOYMENT FEATURES")
    print("="*60)

    # ==============================
    # 8 = Unable to work (high risk signal)
    mask = df["EMPLOY1"].isna()
    df["UnableToWork_flag"] = (df["EMPLOY1"] == 8).astype("float")
    df.loc[mask, "UnableToWork_flag"] = np.nan

    return df

# ===================================
# BMI
# ===================================

def add_bmi_features(df):
    """
    Add BMI-related features from BRFSS BMI variables.
    Uses:
        - _RFBMI5  : BRFSS obesity risk flag (1 = obese, 2 = not obese)
        - _BMI5CAT : BMI category (1=underweight, 2=normal, 3=overweight, 4=obese)
    Creates:
        - Obese_flag                : 1 = obese, 0 = not obese
        - OverweightOrObese_flag    : 1 = BMI category 3 or 4
        - Underweight_flag (optional): 1 = BMI category 1
    """
    df = df.copy()

    has_bmi_cat = "_BMI5CAT" in df.columns
    has_rfbmi5  = "_RFBMI5" in df.columns

    if not has_bmi_cat and not has_rfbmi5:
        print("[WARN] Neither _BMI5CAT nor _RFBMI5 found. BMI features skipped.")
        return df

    print("\n" + "="*60)
    print("BMI FEATURES")
    print("="*60)

    # 1. Obese_flag (cat 4)
    if has_bmi_cat:
        mask_bmi = df["_BMI5CAT"].isna()
        df["Obese_flag"] = (df["_BMI5CAT"] == 4).astype("float")
        df.loc[mask_bmi, "Obese_flag"] = np.nan
    else:
        df["Obese_flag"] = np.nan

    # 2. OverweightOrObese_flag (cat >= 3 or _RFBMI5 == 1)
    if has_bmi_cat:
        mask_bmi = df["_BMI5CAT"].isna()
        df["OverweightOrObese_flag"] = (df["_BMI5CAT"] >= 3).astype("float")
        df.loc[mask_bmi, "OverweightOrObese_flag"] = np.nan
    elif has_rfbmi5:
        mask_rf = df["_RFBMI5"].isna()
        df["OverweightOrObese_flag"] = (df["_RFBMI5"] == 1).astype("float")
        df.loc[mask_rf, "OverweightOrObese_flag"] = np.nan
    else:
        df["OverweightOrObese_flag"] = np.nan

    # 3. Underweight_flag (cat 1)
    if has_bmi_cat:
        mask_bmi = df["_BMI5CAT"].isna()
        df["Underweight_flag"] = (df["_BMI5CAT"] == 1).astype("float")
        df.loc[mask_bmi, "Underweight_flag"] = np.nan
    else:
        df["Underweight_flag"] = np.nan

    return df


# ====================================
# Hypertension
# ====================================

def add_hypertension_features(df):
    """
    Add a clean hypertension indicator.
    Uses:
        - _RFHYPE5 (preferred): 1 = hypertensive, 0/2 = not, NaN = missing
        - BPHIGH4 (fallback): 1 = chronic HTN, others treated as non-HTN
    Creates:
        - Hypertension_flag: 1 = hypertensive, 0 = not, NaN = missing
    """
    df = df.copy()

    has_rfhype5 = "_RFHYPE5" in df.columns
    has_bphigh4 = "BPHIGH4" in df.columns

    if not has_rfhype5 and not has_bphigh4:
        print("[WARN] Neither _RFHYPE5 nor BPHIGH4 found. Hypertension features skipped.")
        return df

    print("\n" + "="*60)
    print("HYPERTENSION FEATURES")
    print("="*60)

    # 1. Main flag
    if has_rfhype5:
        # _RFHYPE5: 1 = yes HTN, 0/2 = no, others/missing = NaN
        mask = df["_RFHYPE5"].isna()
        df["Hypertension_flag"] = (df["_RFHYPE5"] == 1).astype("float")
        df.loc[mask, "Hypertension_flag"] = np.nan
    else:
        # Fallback: BPHIGH4 = 1 is chronic HTN
        mask = df["BPHIGH4"].isna()
        df["Hypertension_flag"] = (df["BPHIGH4"] == 1).astype("float")
        df.loc[mask, "Hypertension_flag"] = np.nan

    return df

# ==================================
# HighChol
# =====================================

def add_highchol_features(df):
    """
    Create a clean high-cholesterol flag from BRFSS risk variable.
    Creates:
        - HighChol_flag : 1 = high cholesterol, 0 = not, NaN = missing

    Raw columns (_RFCHOL, TOLDHI2) are kept for dropping later.
    """
    df = df.copy()

    if "_RFCHOL" not in df.columns:
        print("[WARN] _RFCHOL not found. HighChol_flag skipped.")
        return df

    print("\n" + "="*60)
    print("HIGH CHOLESTEROL FEATURES")
    print("="*60)

    mask = df["_RFCHOL"].isna()
    df["HighChol_flag"] = (df["_RFCHOL"] == 1).astype("float")
    df.loc[mask, "HighChol_flag"] = np.nan

    return df

# ====================================
# Diabetes
# ====================================

def add_diabetes_features(df):
    """
    Create clean diabetes-related flags from BRFSS DIABETE3.
    Uses:
        - DIABETE3:
            1 = diabetes
            2 = diabetes during pregnancy only
            3 = no
            4 = pre-diabetes / borderline
            5 = pre-diabetes (other wording)
            NaN = unknown / refused
    Creates:
        - Diabetes    : 1 = diagnosed diabetes (code 1)
        - PreDiabetes : 1 = pre-diabetes / borderline (codes 4 or 5)
    Raw DIABETE3 is kept for later dropping if desired.
    """
    df = df.copy()

    if "DIABETE3" not in df.columns:
        print("[WARN] DIABETE3 not found. Diabetes features skipped.")
        return df

    print("\n" + "="*60)
    print("DIABETES FEATURES")
    print("="*60)

    mask = df["DIABETE3"].isna()

    # 1. Diabetes (chronic diabetes only)
    df["Diabetes"] = (df["DIABETE3"] == 1).astype("float")

    # 2. PreDiabetes (borderline / pre-diabetes)
    df["PreDiabetes"] = df["DIABETE3"].isin([4, 5]).astype("float")

    df.loc[mask, ["Diabetes", "PreDiabetes"]] = np.nan

    return df


# =========================================
# CVD hist
# =========================================

def add_cvd_history_features(df):
    """
    Create cardiovascular history flags from BRFSS CVD variables.
    Uses:
        - CVDINFR4 : ever had MI (heart attack)
        - CVDCRHD4 : ever had angina / CHD
        - CVDSTRK3 : ever had stroke
        (1 = yes, 2 = no, NaN = unknown/refused after cleaning)

    Creates:
        - HadMI       : 1 = yes MI, 0 = no, NaN = missing
        - HadCHD      : 1 = yes CHD/angina, 0 = no, NaN = missing
        - HadStroke   : 1 = yes stroke, 0 = no, NaN = missing
        - AnyCVD_flag :
            1 if any of HadMI/HadCHD/HadStroke == 1
            0 if all three are known and all 0
            NaN otherwise (i.e. any missing and no 1)
    """
    df = df.copy()

    needed_cols = ["CVDINFR4", "CVDCRHD4", "CVDSTRK3"]
    missing = [c for c in needed_cols if c not in df.columns]
    if len(missing) == len(needed_cols):
        print("[WARN] No CVD history columns found. CVD features skipped.")
        return df

    print("\n" + "="*60)
    print("CARDIOVASCULAR HISTORY FEATURES")
    print("="*60)

    # HadMI
    if "CVDINFR4" in df.columns:
        mask = df["CVDINFR4"].isna()
        df["HadMI"] = (df["CVDINFR4"] == 1).astype("float")
        df.loc[mask, "HadMI"] = np.nan

    # HadCHD
    if "CVDCRHD4" in df.columns:
        mask = df["CVDCRHD4"].isna()
        df["HadCHD"] = (df["CVDCRHD4"] == 1).astype("float")
        df.loc[mask, "HadCHD"] = np.nan

    # HadStroke
    if "CVDSTRK3" in df.columns:
        mask = df["CVDSTRK3"].isna()
        df["HadStroke"] = (df["CVDSTRK3"] == 1).astype("float")
        df.loc[mask, "HadStroke"] = np.nan

    # -------- AnyCVD_flag with strict logic --------
    flag_cols = [c for c in ["HadMI", "HadCHD", "HadStroke"] if c in df.columns]
    if flag_cols:
        sum_flags = df[flag_cols].sum(axis=1, skipna=True)
        count_known = df[flag_cols].notna().sum(axis=1)
        n_flags = len(flag_cols)

        df["AnyCVD_flag"] = np.nan

        # 1) Any 1 -> 1
        df.loc[sum_flags > 0, "AnyCVD_flag"] = 1.0

        # 2) All known and all 0 -> 0
        df.loc[(sum_flags == 0) & (count_known == n_flags), "AnyCVD_flag"] = 0.0
        # 3) Else: vẫn là NaN

    return df

# ================================
# Pulmonary & respiratory
# ================================

def add_respiratory_features(df):
    """
    Create pulmonary/respiratory disease flags.

    Uses:
        - CHCCOPD1 : 1 = COPD/emphysema/chronic bronchitis, 2 = no, NaN = unknown
        - _ASTHMS1 : 1 = current asthma, 2 = former asthma, 3 = never, NaN = unknown

    Creates:
        - COPD_flag             : 1 = has COPD, 0 = no, NaN = missing
        - CurrentAsthma_flag    : 1 = current asthma, 0 = no, NaN = missing
        - EverAsthma_flag       : 1 = ever asthma (current or former), 0 = never, NaN = missing
        - Asthma_Remission_flag : 1 = former asthma (now resolved), 0 otherwise, NaN = missing
    """
    df = df.copy()

    print("\n" + "="*60)
    print("RESPIRATORY FEATURES")
    print("="*60)

    # -------- COPD_flag from CHCCOPD1 --------
    if "CHCCOPD1" in df.columns:
        mask_copd = df["CHCCOPD1"].isna()
        df["COPD_flag"] = (df["CHCCOPD1"] == 1).astype("float")
        df.loc[mask_copd, "COPD_flag"] = np.nan
    else:
        print("[WARN] CHCCOPD1 not found. COPD_flag skipped.")

    # -------- Asthma flags from _ASTHMS1 --------
    if "_ASTHMS1" in df.columns:
        mask_asth = df["_ASTHMS1"].isna()

        # 1. Current asthma
        df["CurrentAsthma_flag"] = (df["_ASTHMS1"] == 1).astype("float")
        df.loc[mask_asth, "CurrentAsthma_flag"] = np.nan

        # 2. Ever asthma (current or former)
        df["EverAsthma_flag"] = df["_ASTHMS1"].isin([1, 2]).astype("float")
        df.loc[mask_asth, "EverAsthma_flag"] = np.nan

        # 3. Asthma remission (had asthma but not now)
        df["Asthma_Remission_flag"] = (df["_ASTHMS1"] == 2).astype("float")
        df.loc[mask_asth, "Asthma_Remission_flag"] = np.nan
    else:
        print("[WARN] _ASTHMS1 not found. Asthma-related flags skipped.")

    return df

# =================================
# Self-rated health
# =================================

def add_selfrated_health_features(df):
    """
    Self-rated general health from GENHLTH.
    Uses:
        - GENHLTH: 1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor, NaN=missing
    Creates:
        - PoorHealth_flag      : 1 if Fair/Poor (4 or 5), 0 if 1–3, NaN if missing
        - VeryPoorHealth_flag  : 1 if Poor (5), 0 otherwise, NaN if missing
    Keeps raw GENHLTH for potential ordinal use.
    """
    df = df.copy()

    if "GENHLTH" not in df.columns:
        print("[WARN] GENHLTH not found. Self-rated health features skipped.")
        return df

    print("\n" + "="*60)
    print("SELF-RATED HEALTH FEATURES")
    print("="*60)

    mask = df["GENHLTH"].isna()

    # Fair or Poor vs better
    df["PoorHealth_flag"] = df["GENHLTH"].isin([4, 5]).astype("float")
    df.loc[mask, "PoorHealth_flag"] = np.nan

    # Poor only (optional, finer split)
    df["VeryPoorHealth_flag"] = (df["GENHLTH"] == 5).astype("float")
    df.loc[mask, "VeryPoorHealth_flag"] = np.nan

    return df

# =================================
# Poor Health Day
# =================================

def add_poor_health_day_features(df):
    """
    Poor health days: physical, mental, and activity limitation.

    Uses:
        - PHYSHLTH : # of days physical health not good (0–30, NaN = unknown)
        - MENTHLTH : # of days mental health not good (0–30, NaN = unknown)
        - POORHLTH : # of days poor health kept from usual activities

    Creates:
        - PoorPhysical14plus_flag : 1 if PHYSHLTH >= 14, 0 if <14, NaN if missing
        - PoorMental14plus_flag   : 1 if MENTHLTH >= 14, 0 if <14, NaN if missing
        - AnyActivityLimitation_flag : 1 if POORHLTH > 0, 0 if ==0, NaN if missing

    Raw day-count variables are kept for possible numeric use.
    """
    df = df.copy()

    print("\n" + "="*60)
    print("POOR HEALTH DAYS FEATURES")
    print("="*60)

    # -------- Physical poor health (PHYSHLTH) --------
    if "PHYSHLTH" in df.columns:
        mask_phys = df["PHYSHLTH"].isna()
        df["PoorPhysical14plus_flag"] = (df["PHYSHLTH"] >= 14).astype("float")
        df.loc[mask_phys, "PoorPhysical14plus_flag"] = np.nan
    else:
        print("[WARN] PHYSHLTH not found. Physical poor-health features skipped.")

    # -------- Mental poor health (MENTHLTH) --------
    if "MENTHLTH" in df.columns:
        mask_ment = df["MENTHLTH"].isna()
        df["PoorMental14plus_flag"] = (df["MENTHLTH"] >= 14).astype("float")
        df.loc[mask_ment, "PoorMental14plus_flag"] = np.nan
    else:
        print("[WARN] MENTHLTH not found. Mental poor-health features skipped.")

    # -------- Activity limitation (POORHLTH) --------
    if "POORHLTH" in df.columns:
        mask_poor = df["POORHLTH"].isna()
        df["AnyActivityLimitation_flag"] = (df["POORHLTH"] > 0).astype("float")
        df.loc[mask_poor, "AnyActivityLimitation_flag"] = np.nan
    else:
        print("[WARN] POORHLTH not found. Activity limitation feature skipped.")

    return df

# ===================================
# Smoking
# ===================================

def add_smoking_features(df):
    """
    Create smoking status flags from BRFSS derived tobacco variables.
    Uses:
        - _SMOKER3 : 1=Current daily, 2=Current some days,
                    3=Former, 4=Never, NaN=unknown

        - _RFSMOK3 (optional): 1=current smoker, 2=not, NaN=unknown

    Creates:
        - CurrentSmoker
        - FormerSmoker
        - NeverSmoker
    """
    df = df.copy()

    print("\n" + "="*60)
    print("SMOKING FEATURES")
    print("="*60)

    if "_SMOKER3" not in df.columns:
        print("[WARN] _SMOKER3 not found. Smoking features skipped.")
        return df

    mask = df["_SMOKER3"].isna()

    # 1. Current smoker (code 1 or 2)
    df["CurrentSmoker"] = df["_SMOKER3"].isin([1, 2]).astype("float")
    df.loc[mask, "CurrentSmoker"] = np.nan

    # 2. Former smoker (code 3)
    df["FormerSmoker"] = (df["_SMOKER3"] == 3).astype("float")
    df.loc[mask, "FormerSmoker"] = np.nan

    # 3. Never smoker (code 4)
    df["NeverSmoker"] = (df["_SMOKER3"] == 4).astype("float")
    df.loc[mask, "NeverSmoker"] = np.nan

    return df

# ==============================
# Physical Activity
# ==============================

def add_physical_activity_features(df):
    """
    Create physical activity features from EXRACT11/21 activity type codes.

    Uses:
        - EXRACT11: aerobic activity type 
                    88 = none, NaN = unknown, all other codes = some activity
        - EXRACT21: strength activity type 
                    88 = none, NaN = unknown, all other codes = some activity

    Creates:
        - AnyAerobicActivity_flag
        - AnyStrengthActivity_flag
        - AnyExercise_flag
    """
    df = df.copy()

    print("\n" + "="*60)
    print("PHYSICAL ACTIVITY FEATURES")
    print("="*60)

    # --------------------
    # Aerobic activity
    # --------------------
    if "EXRACT11" in df.columns:
        df["AnyAerobicActivity_flag"] = np.nan

        # explicit none

        # valid activities (not 88, not NaN)
        mask_valid = df["EXRACT11"].notna()
        df.loc[mask_valid, "AnyAerobicActivity_flag"] = 1.0
    else:
        print("[WARN] EXRACT11 not found.")

    # --------------------
    # Strength activity
    # --------------------
    if "EXRACT21" in df.columns:
        df["AnyStrengthActivity_flag"] = np.nan

        df.loc[df["EXRACT21"] == 88, "AnyStrengthActivity_flag"] = 0.0

        mask_valid2 = df["EXRACT21"].notna() & (df["EXRACT21"] != 88)
        df.loc[mask_valid2, "AnyStrengthActivity_flag"] = 1.0
    else:
        print("[WARN] EXRACT21 not found.")

    # --------------------
    # Combined flag
    # --------------------
    if "AnyAerobicActivity_flag" in df.columns or "AnyStrengthActivity_flag" in df.columns:
        df["AnyExercise_flag"] = np.nan

        # any = 1
        df.loc[
            (df.get("AnyAerobicActivity_flag") == 1.0) |
            (df.get("AnyStrengthActivity_flag") == 1.0),
            "AnyExercise_flag"
        ] = 1.0

        # both known and both = 0
        both_known = (
            df.get("AnyAerobicActivity_flag").isin([0.0, 1.0]) &
            df.get("AnyStrengthActivity_flag").isin([0.0, 1.0])
        )

        df.loc[both_known & df["AnyExercise_flag"].isna(), "AnyExercise_flag"] = 0.0

    return df

# ==============================
# Access
# ==============================

def add_access_features(df):
    """
    Access to care features: insurance, primary doctor, cost barrier, checkup recency.

    Uses:
        - HLTHPLN1 : 1 = has health insurance, 2 = no, NaN = unknown
        - PERSDOC2 : 1/2 = has >=1 personal doctor, 3 = no, NaN = unknown
        - MEDCOST  : 1 = could not see doctor due to cost, 2 = no, NaN = unknown
        - CHECKUP1 : 1 = within past year, 2–4/8 = longer/never, NaN = unknown

    Creates:
        - HasInsurance_flag
        - HasPrimaryDoctor_flag
        - CouldNotAffordCare_flag
        - RecentCheckup_flag
    """
    df = df.copy()

    print("\n" + "="*60)
    print("ACCESS TO CARE FEATURES")
    print("="*60)

    # ----------------- Insurance -----------------
    if "HLTHPLN1" in df.columns:
        mask_ins = df["HLTHPLN1"].isna()
        df["HasInsurance_flag"] = (df["HLTHPLN1"] == 1).astype("float")
        df.loc[mask_ins, "HasInsurance_flag"] = np.nan
    else:
        print("[WARN] HLTHPLN1 not found. HasInsurance_flag skipped.")

    # ----------------- Primary doctor -----------------
    if "PERSDOC2" in df.columns:
        mask_doc = df["PERSDOC2"].isna()
        df["HasPrimaryDoctor_flag"] = df["PERSDOC2"].isin([1, 2]).astype("float")
        df.loc[mask_doc, "HasPrimaryDoctor_flag"] = np.nan
    else:
        print("[WARN] PERSDOC2 not found. HasPrimaryDoctor_flag skipped.")

    # ----------------- Cost barrier -----------------
    if "MEDCOST" in df.columns:
        mask_cost = df["MEDCOST"].isna()
        df["CouldNotAffordCare_flag"] = (df["MEDCOST"] == 1).astype("float")
        df.loc[mask_cost, "CouldNotAffordCare_flag"] = np.nan
    else:
        print("[WARN] MEDCOST not found. CouldNotAffordCare_flag skipped.")

    # ----------------- Recent checkup -----------------
    if "CHECKUP1" in df.columns:
        mask_chk = df["CHECKUP1"].isna()
        df["RecentCheckup_flag"] = (df["CHECKUP1"] == 1).astype("float")
        df.loc[mask_chk, "RecentCheckup_flag"] = np.nan
    else:
        print("[WARN] CHECKUP1 not found. RecentCheckup_flag skipped.")

    return df

def add_riskfactor_features(df):
    """
    Tier-2: risk factor cluster.
    Uses (if present):
        - Hypertension_flag
        - HighChol
        - Diabetes
        - Obese_flag
        - CurrentSmoker

    Creates:
        - RiskFactorCount       : # of risk factors present (0..N, NaN if all missing)
        - RiskFactor_missing_any: 1 if any RF missing, else 0
    EDA: check CVD rate by RiskFactorCount (0,1,2,3,4+).
    """
    df = df.copy()

    print("\n" + "="*70)
    print("INTERACTION – RISK FACTOR CLUSTER")
    print("="*70)

    rf_cols = [c for c in [
        "Hypertension_flag",
        "HighChol_flag",
        "Diabetes",
        "Obese_flag",
        "CurrentSmoker"  # hoặc CurrentSmoker_flag nếu bạn dùng tên đó
    ] if c in df.columns]

    if len(rf_cols) == 0:
        print("[WARN] No risk factor columns found. Skipping.")
        return df

    print(f"Using risk factor columns: {rf_cols}")

    # all-missing mask
    sub = df[rf_cols]
    mask_all_nan = sub.isna().all(axis=1)

    # count positives, treating NaN as 0 but fixing rows with all NaN
    df["RiskFactorCount"] = sub.fillna(0).sum(axis=1)
    df.loc[mask_all_nan, "RiskFactorCount"] = np.nan

    # any missing among RFs
    df["RiskFactor_missing_any"] = sub.isna().any(axis=1).astype("float")

    # ------- logging -------
    print("\nRiskFactorCount distribution:")
    print(df["RiskFactorCount"].value_counts(dropna=False).sort_index())

    print("\nRiskFactor_missing_any distribution:")
    print(df["RiskFactor_missing_any"].value_counts(dropna=False))

    print("\nSample (RF cols + RiskFactorCount):")
    print(df[rf_cols + ["RiskFactorCount", "RiskFactor_missing_any"]].head())

    return df

def add_metabolic_syndrome_features(df):
    """
    Tier-2: metabolic syndrome cluster.
    Uses:
        - Diabetes
        - Hypertension_flag
        - HighChol
        - Obese_flag

    Creates:
        - MetabolicSyndrome_flag : 1 if >=3 RFs present, 0 otherwise, NaN if all missing
        - MetSyn_missing_flag    : 1 if any of 4 components missing, else 0
    EDA: tabulate CVD vs MetabolicSyndrome_flag.
    """
    df = df.copy()

    print("\n" + "="*70)
    print("INTERACTION – METABOLIC SYNDROME")
    print("="*70)

    met_cols = [c for c in [
        "Diabetes",
        "Hypertension_flag",
        "HighChol",
        "Obese_flag"
    ] if c in df.columns]

    if len(met_cols) < 2:
        print(f"[WARN] Too few metabolic columns found ({met_cols}). Skipping.")
        return df

    print(f"Using metabolic components: {met_cols}")

    sub = df[met_cols]
    mask_all_nan = sub.isna().all(axis=1)

    # >=3 factors present (strict)
    sum_pos = sub.fillna(0).sum(axis=1)
    df["MetabolicSyndrome_flag"] = (sum_pos >= 3).astype("float")
    df.loc[mask_all_nan, "MetabolicSyndrome_flag"] = np.nan

    df["MetSyn_missing_flag"] = sub.isna().any(axis=1).astype("float")

    # ------- logging -------
    print("\nMetabolicSyndrome_flag distribution:")
    print(df["MetabolicSyndrome_flag"].value_counts(dropna=False))

    print("\nMetSyn_missing_flag distribution:")
    print(df["MetSyn_missing_flag"].value_counts(dropna=False))

    print("\nSample (metabolic + MetabolicSyndrome_flag):")
    print(df[met_cols + ["MetabolicSyndrome_flag", "MetSyn_missing_flag"]].head())

    return df

def _strict_interaction(df, cols, name):
    """
    Helper: strict AND over binary columns with missing-aware logic.
    Logic:
      - 1  if ALL cols == 1
      - 0  if ALL cols are known (0 or 1) AND NOT ALL are 1
      - NaN otherwise (i.e. any col is NaN AND not all cols are 1)
    """
    sub = df[cols]
    n_cols = len(cols)

    # Count how many are 1s and how many are known (not NaN)
    sum_pos = sub.sum(axis=1, skipna=True)
    count_known = sub.notna().sum(axis=1)

    # Initialize result to NaN
    df[name] = np.nan

    # 1. Set to 1.0 if all components are 1
    df.loc[(sum_pos == n_cols) & (count_known == n_cols), name] = 1.0

    # 2. Set to 0.0 if all components are known (count_known == n_cols) 
    #    AND the result is not already 1.0 (i.e., sum_pos < n_cols)
    df.loc[(count_known == n_cols) & df[name].isna(), name] = 0.0

    # Missing Flag (this part is fine)
    df[name + "_missing_flag"] = (count_known < n_cols).astype("float")
    
    return df


def add_binary_interactions(df):
    """
    Tier-2: key binary×binary interactions.
    Uses (if present):
        - Obese_flag
        - Hypertension_flag
        - Diabetes
        - HighChol
        - AnyExercise_flag  (to derive Inactive_flag)

    Creates:
        - Obese_and_Inactive_flag (+ _missing_flag)
        - Obese_and_Hypertensive_flag (+ _missing_flag)
        - HTN_and_Diabetes_flag (+ _missing_flag)
        - Diabetes_and_HighChol_flag (+ _missing_flag)

    EDA: compare CVD rate by each interaction flag.
    """
    df = df.copy()

    print("\n" + "="*70)
    print("INTERACTION – BINARY COMBOS")
    print("="*70)

    # -------- derive Inactive_flag from AnyExercise_flag --------
    if "AnyExercise_flag" in df.columns:
        df["Inactive_flag"] = np.nan
        df.loc[df["AnyExercise_flag"] == 0, "Inactive_flag"] = 1.0
        df.loc[df["AnyExercise_flag"] == 1, "Inactive_flag"] = 0.0
    else:
        print("[WARN] AnyExercise_flag not found. Obese_and_Inactive_flag skipped.")

    # list of (cols, new_name)
    combos = []

    if {"Obese_flag", "Inactive_flag"}.issubset(df.columns):
        combos.append((["Obese_flag", "Inactive_flag"], "Obese_and_Inactive_flag"))

    if {"Obese_flag", "Hypertension_flag"}.issubset(df.columns):
        combos.append((["Obese_flag", "Hypertension_flag"], "Obese_and_Hypertensive_flag"))

    if {"Hypertension_flag", "Diabetes"}.issubset(df.columns):
        combos.append((["Hypertension_flag", "Diabetes"], "HTN_and_Diabetes_flag"))

    if {"Diabetes", "HighChol_flag"}.issubset(df.columns):
        combos.append((["Diabetes", "HighChol_flag"], "Diabetes_and_HighChol_flag"))

    if not combos:
        print("[WARN] No valid binary interaction combos found.")
        return df

    for cols, name in combos:
        print(f"\nCreating interaction: {name} from {cols}")
        _strict_interaction(df, cols, name)
        print(f"{name} distribution:")
        print(df[name].value_counts(dropna=False))

    return df

def add_numeric_interactions(df):
    """
    Tier-2: numeric × binary interactions.
    Uses:
        - Age
        - CurrentSmoker (or CurrentSmoker_flag)

    Creates:
        - Age_x_CurrentSmoker
    EDA: compare Age_x_CurrentSmoker distribution by CVD outcome.
    """
    df = df.copy()

    print("\n" + "="*70)
    print("INTERACTION – NUMERIC × BINARY")
    print("="*70)

    smoker_col = None
    for cand in ["CurrentSmoker", "CurrentSmoker_flag"]:
        if cand in df.columns:
            smoker_col = cand
            break

    if "Age" not in df.columns or smoker_col is None:
        print(f"[WARN] Age or smoker column not found. Skipping Age_x_CurrentSmoker.")
        return df

    df["Age_x_CurrentSmoker"] = df["Age"] * df[smoker_col]

    print("\nAge_x_CurrentSmoker summary:")
    print(df["Age_x_CurrentSmoker"].describe())

    return df


class BRFSSPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete BRFSS pipeline
    Usage:
        
    from src.feature_engineering import BRFSSPreprocessor
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df[[c for c in df.columns if c != target]], 
        df[target], 
        test_size=0.2, 
        stratify=df[target],
        random_state=42
    )
    # 2. Fit on TRAIN only
    preprocessor = BRFSSPreprocessor(impute_medium_cols=True) # Set to False if training tree models
    X_train_clean = preprocessor.fit_transform(X_train)  
    # 3. Transform TEST using TRAIN stats only
    X_test_clean = preprocessor.transform(X_test)        
    """
    def __init__(self, impute_medium_cols: bool = False):
        self.impute_medium_cols = impute_medium_cols
        # These get learned on TRAIN only
        self.medians_ = {}           # For imputation
        self.quantiles_ = {}         # For outlier capping

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # 1. Learn medians for imputation (train only!)
        median_cols = ["PHYSHLTH", "MENTHLTH", "POORHLTH", "_VEGESUM", "_FRUTSUM"]
        for col in median_cols:
            if col in X.columns:
                self.medians_[col] = X[col].median()

        # 2. Learn 99.9th percentiles for capping (train only!)
        quantile_cols = [
            "PAMIN11_", "PAMIN21_", "_MINAC11", "_MINAC21",
            "PAVIG11_", "PAVIG21_", "PADUR1_", "STRFREQ_",
            "METVL21_", "METVL11_", "MAXVO2_", "FC60_", "CHILDREN"
        ]
        for col in quantile_cols:
            if col in X.columns:
                self.quantiles_[col] = X[col].quantile(0.999)

        print(f" Fitted on {len(X):,} train samples")
        print(f"   → Learned {len(self.medians_)} medians")
        print(f"   → Learned {len(self.quantiles_)} quantiles")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Feature engineering (safe — pure logic)
        feature_funcs = [
            add_age_features, add_sex_features, add_race_features, add_employ_features,
            add_bmi_features, add_hypertension_features, add_highchol_features,
            add_diabetes_features, add_cvd_history_features, add_respiratory_features,
            add_selfrated_health_features, add_poor_health_day_features,
            add_smoking_features, add_physical_activity_features, add_access_features,
            add_riskfactor_features, add_metabolic_syndrome_features,
            add_binary_interactions, add_numeric_interactions
        ]
        for func in feature_funcs:
            X = func(X)

        # 2. MISSING HANDLING — uses TRAIN medians only
        X = self._safe_missing_handling(X)

        # 3. OUTLIER CAPPING — uses TRAIN quantiles only
        X = self._safe_outlier_capping(X)

        return X

    def _safe_missing_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Missing handling using ONLY train medians — ZERO leakage."""
        df = df.copy()

        # === CRITICAL: Use TRAIN medians only ===
        for col, train_median in self.medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(train_median)

        # === Rest of your logic (flags, zero/one imputation) ===
        # This part is already safe — keep exactly as-is
        medium_cols = ["INCOME2", "CHOLCHK", "PNEUVAC3", "FLUSHOT6"]
        for col in medium_cols:
            if col in df.columns:
                df[f"{col}_Missing_flag"] = df[col].isna().astype(float)

        # PA systematic missingness
        pa_cols = ["PAMIN11_","PAMIN21_","PAVIG21_","PADUR1_","PAFREQ1_",
                   "_MINAC21","_MINAC11","ACTIN21_","ACTIN11_","METVL21_","METVL11_"]
        pa_present = [c for c in pa_cols if c in df.columns]
        if pa_present:
            df["PA_Detail_Input_Missing_flag"] = df[pa_present].isna().any(axis=1).astype(float)

        # Zero/one imputation for PA flags
        pa_impute_cols = ["AnyAerobicActivity_flag","AnyStrengthActivity_flag",
                          "Inactive_flag","AnyExercise_flag"] + pa_present
        for col in pa_impute_cols:
            if col in df.columns and df[col].isna().any():
                fill_val = 1.0 if "inactive" in col.lower() else 0.0
                df[col] = df[col].fillna(fill_val)

        # Chronic disease flags
        chronic_cols = ["Hypertension_flag","Diabetes","HadCHD","HadStroke","HadMI",
                        "AnyCVD_flag","Obese_flag","Underweight_flag","OverweightOrObese_flag",
                        "CurrentSmoker","FormerSmoker","NeverSmoker","COPD_flag",
                        "CurrentAsthma_flag","EverAsthma_flag","Asthma_Remission_flag"]
        for col in chronic_cols:
            if col in df.columns and df[col].isna().any():
                df[f"{col}_Missing_flag"] = df[col].isna().astype(float)
                df[col] = df[col].fillna(0.0)

        # Optional -1 imputation
        if self.impute_medium_cols:
            for col in medium_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(-1)

        return df

    def _safe_outlier_capping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Outlier capping using ONLY train quantiles — ZERO leakage."""
        df = df.copy()

        # === CRITICAL: Hard caps (same for train/test) ===
        hard_caps = {
            "PHYSHLTH": 30, "MENTHLTH": 30, "POORHLTH": 30,
            "_VEGESUM": 20, "_FRUTSUM": 20,
        }
        for col, cap in hard_caps.items():
            if col in df.columns:
                df[col] = df[col].clip(0, cap)

        # === CRITICAL: Dynamic caps using TRAIN quantiles only ===
        for col, train_q999 in self.quantiles_.items():
            if col in df.columns:
                # Apply train quantile (never compute on test!)
                df[col] = df[col].clip(upper=train_q999)

        # === Hard physiological limits (safety net) ===
        phys_limits = {
            "PAMIN11_": 2100, "PAMIN21_": 2100,
            "_MINAC11": 2100, "_MINAC21": 2100,
            "PAVIG11_": 2100, "PAVIG21_": 2100,
            "PADUR1_": 2100, "STRFREQ_": 14
        }
        for col, limit in phys_limits.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=limit)

        return df