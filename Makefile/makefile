# Makefile for SPARCVI memory build

# Configurable paths
CSV=src/sparcvi/config/fault_feature_matrix.csv
OUT=src/sparcvi/data/fault_memory_maps

# Command: convert CSV matrix to .npy
build-memory:
	@echo "🔧 Building fault memory from CSV..."
	@python src/sparcvi/memory/memory_converter.py --csv $(CSV) --out $(OUT)
	@echo "✅ Memory built at: $(OUT)/fault_feature_matrix.npy"
