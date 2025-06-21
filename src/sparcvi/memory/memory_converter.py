import pandas as pd
import numpy as np
import os
import argparse

def convert_matrix_to_npy(csv_path, output_dir):
    # Load CSV
    df = pd.read_csv(csv_path, index_col=False)
    if "Feature Name" not in df.columns:
        raise ValueError("CSV must contain 'Feature Name' as the first column.")

    df.set_index("Feature Name", inplace=True)

    # Convert to numpy array
    matrix = df.to_numpy(dtype=np.int8)

    # Save .npy file
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, "fault_feature_matrix.npy")
    np.save(npy_path, matrix)

    # Save feature and fault labels
    df.index.to_series().to_csv(os.path.join(output_dir, "feature_names.csv"), index=False)
    pd.Series(df.columns).to_csv(os.path.join(output_dir, "fault_names.csv"), index=False)

    print(f"[✓] Matrix saved to: {npy_path}")
    print(f"    Shape: {matrix.shape}")

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to fault_feature_matrix.csv")
    parser.add_argument("--out", required=True, help="Output directory for .npy files")
    args = parser.parse_args()

    convert_matrix_to_npy(args.csv, args.out)

