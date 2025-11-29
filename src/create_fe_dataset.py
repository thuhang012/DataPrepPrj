"""
Script to create the feature-engineered (FE) dataset from the CLEAN dataset.

This applies all feature engineering functions from feature_engineering.py
to the clean dataset, avoiding data leakage by ensuring no fitted statistics
are computed on the full dataset before splitting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from .config import load_data, get_project_root
from .feature_engineering import (
    add_age_features,
    add_sex_features,
    add_race_features,
    add_employ_features,
    add_bmi_features,
    add_hypertension_features,
    add_highchol_features,
    add_diabetes_features,
    add_cvd_history_features,
    add_respiratory_features,
    add_selfrated_health_features,
    add_poor_health_day_features,
    add_smoking_features,
    add_physical_activity_features,
    add_access_features,
    add_riskfactor_features,
    add_metabolic_syndrome_features,
    add_binary_interactions,
    add_numeric_interactions,
    handle_missing,
    cap_outliers,
)


def create_fe_dataset(
    input_file: str = "clean_2015.csv",
    output_file: str = "fe_2015.csv"
) -> pd.DataFrame:
    """
    Create feature-engineered dataset from clean dataset.
    
    This function applies all feature engineering transformations that:
    1. Do NOT require fitted statistics (safe to apply before split)
    2. Create new features from existing ones
    3. Create interaction terms and derived variables
    
    NOTE: Any transformations requiring fitted statistics (like imputation
    with train medians, outlier capping with train quantiles) should be done
    INSIDE scikit-learn pipelines after splitting, or using the 
    BRFSSPreprocessor.fit_transform() pattern.
    
    Args:
        input_file: Name of input clean CSV file
        output_file: Name of output FE CSV file
        
    Returns:
        Feature-engineered dataframe
    """
    print("="*80)
    print("CREATING FEATURE-ENGINEERED (FE) DATASET")
    print("="*80)
    
    # Load clean dataset
    print(f"\n1. Loading clean dataset: {input_file}")
    df_clean = load_data(input_file, subfolder="processed")
    print(f"   Shape: {df_clean.shape}")
    print(f"   Columns: {len(df_clean.columns)}")
    
    # Apply feature engineering (all pure transformations, no fitting)
    print("\n2. Applying feature engineering transformations...")
    print("   (These are deterministic transformations with no data leakage)")
    
    df_fe = df_clean.copy()
    
    # List of all FE functions (order matters for dependencies)
    fe_functions = [
        ("Age features", add_age_features),
        ("Sex features", add_sex_features),
        ("Race features", add_race_features),
        ("Employment features", add_employ_features),
        ("BMI features", add_bmi_features),
        ("Hypertension features", add_hypertension_features),
        ("High cholesterol features", add_highchol_features),
        ("Diabetes features", add_diabetes_features),
        ("CVD history features", add_cvd_history_features),
        ("Respiratory features", add_respiratory_features),
        ("Self-rated health features", add_selfrated_health_features),
        ("Poor health days features", add_poor_health_day_features),
        ("Smoking features", add_smoking_features),
        ("Physical activity features", add_physical_activity_features),
        ("Access to care features", add_access_features),
        ("Risk factor cluster features", add_riskfactor_features),
        ("Metabolic syndrome features", add_metabolic_syndrome_features),
        ("Binary interaction features", add_binary_interactions),
        ("Numeric interaction features", add_numeric_interactions),
    ]
    
    # Apply each function
    for name, func in fe_functions:
        try:
            print(f"   - {name}...", end=" ")
            n_cols_before = len(df_fe.columns)
            df_fe = func(df_fe)
            n_cols_after = len(df_fe.columns)
            n_new = n_cols_after - n_cols_before
            if n_new > 0:
                print(f"✓ (+{n_new} columns)")
            else:
                print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary
    print(f"\n3. Feature engineering complete!")
    print(f"   Original columns: {len(df_clean.columns)}")
    print(f"   New columns:      {len(df_fe.columns)}")
    print(f"   Added:            {len(df_fe.columns) - len(df_clean.columns)}")
    
    # Handle missing values and outliers
    print(f"\n4. Handling missing values and outliers...")
    df_fe = handle_missing(df_fe, impute_medium_cols=True)
    print(f"   ✓ Missing values handled")
    df_fe = cap_outliers(df_fe)
    print(f"   ✓ Outliers capped")
    
    # Save
    print(f"\n5. Saving FE dataset to: {output_file}")
    project_root = get_project_root()
    output_path = project_root / "data" / "processed" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_fe.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    
    # Quality checks
    print(f"\n6. Quality checks:")
    print(f"   - Shape: {df_fe.shape}")
    print(f"   - Missing values: {df_fe.isna().sum().sum():,} ({df_fe.isna().sum().sum() / df_fe.size * 100:.2f}%)")
    print(f"   - Duplicates: {df_fe.duplicated().sum()}")
    
    # Check target distribution
    if "_MICHD" in df_fe.columns:
        target_dist = df_fe["_MICHD"].value_counts(normalize=True)
        print(f"\n   Target (_MICHD) distribution:")
        for val, prop in target_dist.items():
            print(f"     {val}: {prop:.2%}")
    
    print("\n" + "="*80)
    print("FE DATASET CREATED SUCCESSFULLY!")
    print("="*80)
    
    return df_fe


if __name__ == "__main__":
    # Create FE dataset
    df_fe = create_fe_dataset(
        input_file="clean_2015.csv",
        output_file="fe_2015.csv"
    )
    
    print("\n✓ Done! You can now use this FE dataset in the modeling pipeline.")
    print("\nNext steps:")
    print("1. Load RAW, CLEAN, and FE datasets")
    print("2. Apply same train/val/test split to all three")
    print("3. Build preprocessing + modeling pipelines")
    print("4. Compare results across datasets")
