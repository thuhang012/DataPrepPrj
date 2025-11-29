from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def get_project_root():
    """Get the project root directory."""
    return PROJECT_ROOT


def load_data(filename: str, subfolder: str = ""):
    """
    Load data from CSV file.
    
    Args:
        filename: Name of the CSV file
        subfolder: Optional subfolder within data/ (e.g., 'raw', 'processed')
    
    Returns:
        DataFrame
    """
    if subfolder:
        file_path = PROJECT_ROOT / "data" / subfolder / filename
    else:
        file_path = PROJECT_ROOT / "data" / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)
