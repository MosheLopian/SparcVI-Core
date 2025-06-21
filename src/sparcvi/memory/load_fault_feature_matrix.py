import pandas as pd
import os

def load_fault_feature_matrix(csv_path):
    """
    Load and validate the fault-feature matrix from a CSV file.
    Returns:
        pd.DataFrame: Feature x Fault matrix with signed integer scores.
    Raises:
        ValueError: If matrix contains missing values or invalid data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Matrix file not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=False)

    if "Feature Name" not in df.columns:
        raise ValueError("Missing 'Feature Name' column in fault-feature matrix.")

    df.set_index("Feature Name", inplace=True)

    if df.isnull().values.any():
        raise ValueError("Fault-feature matrix contains missing values.")

    if not all(df.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt))):
        raise ValueError("All impact scores must be numeric.")

    return df