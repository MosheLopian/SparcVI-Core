from src.sparcvi.memory.memory_converter import convert_matrix_to_npy

def main():
    # Step 1: Always convert latest CSV matrix to .npy
    print("🧠 Converting fault-feature matrix to .npy...")
    convert_matrix_to_npy(
        csv_path="src/sparcvi/config/fault_feature_matrix.csv",
        output_dir="src/sparcvi/data/fault_memory_maps"
    )
    print("✅ Memory conversion complete.")

    # Step 2: Continue with rest of the system logic
    print("🚀 Starting SPARCVI diagnostic system...")
    # TODO: Add risk scoring, vector matching, and diagnosis pipeline here

if __name__ == "__main__":
    main()