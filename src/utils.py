from pathlib import Path
import pandas as pd

def get_project_root():
    path = Path(__file__).resolve()
    for _ in range(8):
        path = path.parent
        if (path / "data").exists():
            return path
    raise RuntimeError("Project root not found!")

def load_data(filename: str, subfolder: str = ""):
    root = get_project_root()
    if subfolder:
        file_path = root / "data" / subfolder / filename
    else:
        file_path = root / "data" / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)
