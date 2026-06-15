from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


def is_s3_path(path: str | Path) -> bool:
    """Return whether a path points to S3."""
    return str(path).startswith("s3://")


def require_fsspec():
    """Import fsspec lazily so local-only workflows do not need S3 deps."""
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError(
            "S3 paths require fsspec and s3fs. Install them with "
            "`pip install fsspec s3fs`."
        ) from exc
    return fsspec


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV, Excel, or Parquet file from local disk or S3."""
    path_text = str(path)
    suffix = Path(path_text).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path_text)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_text)
    if suffix == ".parquet":
        return pd.read_parquet(path_text)

    raise ValueError(f"Unsupported file type: {suffix}")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe as CSV or Parquet to local disk or S3."""
    path_text = str(path)
    suffix = Path(path_text).suffix.lower()

    if not is_s3_path(path_text):
        Path(path_text).parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(path_text, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path_text, index=False)
        return

    raise ValueError(f"Unsupported output type: {suffix}")


def save_numpy_array(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to local disk or S3 as .npy."""
    path_text = str(path)

    if is_s3_path(path_text):
        fsspec = require_fsspec()
        buffer = BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        with fsspec.open(path_text, "wb") as output_file:
            output_file.write(buffer.read())
        return

    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_dataframe_copies(df: pd.DataFrame, paths: list[str | Path]) -> None:
    """Save the same dataframe to one or more destinations."""
    for path in paths:
        save_dataframe(df, path)


def save_numpy_array_copies(array: np.ndarray, paths: list[str | Path]) -> None:
    """Save the same NumPy array to one or more destinations."""
    for path in paths:
        save_numpy_array(array, path)
