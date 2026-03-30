"""
ml/shared/paths.py

Path constants for ML data files.

Directory structure:
    data/
        features/
            NVDA/
                features_w60.csv
                features_w45.csv
                ...
            AMD/
                features_w60.csv
                ...
        models/
            NVDA/
                session_direction.json
                session_direction_w60.json
                calibrated_w60.pkl
                ...
            AMD/
                session_direction.json
                calibrated.pkl
                ...
        reports/
            NVDA/
                directional_report.txt
                walkforward.png
                ...
            AMD/
                ...
"""

from pathlib import Path

# Base data directory (relative to repo root)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Subdirectories
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"


def features_path(symbol: str, window: int = 60) -> Path:
    """
    Get path to features CSV for a symbol.

    Args:
        symbol: Stock symbol (e.g., "NVDA")
        window: Prediction window in minutes (default: 60)

    Returns:
        Path to features CSV, e.g., data/features/NVDA/features_w60.csv
    """
    symbol = symbol.upper()
    return FEATURES_DIR / symbol / f"features_w{window}.csv"


def model_path(symbol: str, model_type: str = "session_direction", window: int = 60) -> Path:
    """
    Get path to model file.

    Args:
        symbol: Stock symbol (e.g., "NVDA")
        model_type: Model type (default: "session_direction")
        window: Prediction window in minutes (default: 60)

    Returns:
        Path to model JSON, e.g., data/models/NVDA/session_direction_w60.json
    """
    symbol = symbol.upper()
    return MODELS_DIR / symbol / f"{model_type}_w{window}.json"


def calibration_path(symbol: str, window: int = 60) -> Path:
    """
    Get path to calibration pickle file.

    Args:
        symbol: Stock symbol (e.g., "NVDA")
        window: Prediction window in minutes (default: 60)

    Returns:
        Path to calibration pickle, e.g., data/models/NVDA/calibrated_w60.pkl
    """
    symbol = symbol.upper()
    return MODELS_DIR / symbol / f"calibrated_w{window}.pkl"


def report_path(symbol: str, report_type: str, extension: str = "txt") -> Path:
    """
    Get path to report file.

    Args:
        symbol: Stock symbol (e.g., "NVDA")
        report_type: Report type (e.g., "directional_w60", "walkforward")
        extension: File extension (default: "txt")

    Returns:
        Path to report file, e.g., data/reports/NVDA/directional_w60_report.txt
    """
    symbol = symbol.upper()
    filename = f"{report_type}_report.{extension}" if extension == "txt" else f"{report_type}.{extension}"
    return REPORTS_DIR / symbol / filename


def ensure_dirs(symbol: str):
    """
    Ensure all data directories exist for a symbol.

    Args:
        symbol: Stock symbol (e.g., "NVDA")
    """
    symbol = symbol.upper()
    (FEATURES_DIR / symbol).mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / symbol).mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / symbol).mkdir(parents=True, exist_ok=True)
