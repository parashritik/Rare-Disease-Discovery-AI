from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MOD_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "cleaned"

def ensure_directories():
    for d in [DATA_DIR, MOD_DIR, OUT_DIR]:
        d.mkdir(exist_ok=True)