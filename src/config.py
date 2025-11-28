from __future__ import annotations

import numpy as np
import pandas as pd

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
        df.loc[df["EXRACT11"] == 88, "AnyAerobicActivity_flag"] = 0.0

        # valid activities (not 88, not NaN)
        mask_valid = df["EXRACT11"].notna() & (df["EXRACT11"] != 88)
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
        "HighChol",
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
      - 1  if ALL cols == 1
      - 0  if at least one col == 0 and not all NaN
      - NaN if all cols are NaN

    Also creates:
      - {name}_missing_flag : 1 if any component is NaN, else 0
    """
    sub = df[cols]
    mask_all_nan = sub.isna().all(axis=1)

    out = (sub.fillna(0).sum(axis=1) == len(cols)).astype("float")
    out[mask_all_nan] = np.nan

    df[name] = out
    df[name + "_missing_flag"] = sub.isna().any(axis=1).astype("float")


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
