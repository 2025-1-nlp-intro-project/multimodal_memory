#!/bin/bash
# Script to run the entire pipeline: data download, data generation, training, and evaluation

# Display help information
function show_help {
  echo "Usage: ./run_pipeline.sh [OPTIONS]"
  echo ""
  echo "Run the complete Visual Dialogue fine-tuning pipeline"
  echo ""
  echo "Options:"
  echo "  -c, --config CONFIG_FILE   Path to config file (default: config.yaml)"
  echo "  -o, --output OUTPUT_DIR    Directory to store outputs (default: outputs)"
  echo "  -d, --data DATA_DIR        Directory to store data (default: data)"
  echo "  -s, --skip-download        Skip dataset download step"
  echo "  -g, --skip-generation      Skip data generation step"
  echo "  -t, --skip-training        Skip model training step"
  echo "  -e, --skip-evaluation      Skip model evaluation step"
  echo "  -h, --help                 Show this help message"
  echo ""
  echo "Example:"
  echo "  ./run_pipeline.sh --config configs/my_config.yaml --output my_experiment"
}

# Default values
CONFIG_FILE="config.yaml"
OUTPUT_DIR="outputs"
DATA_DIR="data"
SKIP_DOWNLOAD=false
SKIP_GENERATION=false
SKIP_TRAINING=false
SKIP_EVALUATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -d|--data)
      DATA_DIR="$2"
      shift 2
      ;;
    -s|--skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    -g|--skip-generation)
      SKIP_GENERATION=true
      shift
      ;;
    -t|--skip-training)
      SKIP_TRAINING=true
      shift
      ;;
    -e|--skip-evaluation)
      SKIP_EVALUATION=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p $OUTPUT_DIR

# Start logging
LOG_FILE="$OUTPUT_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================"
echo "  Visual Dialogue Fine-tuning Pipeline"
echo "======================================================"
echo "Configuration file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Data directory: $DATA_DIR"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "======================================================"

# 1. Download datasets
if [ "$SKIP_DOWNLOAD" = false ]; then
  echo ""
  echo "STEP 1: Downloading datasets..."
  echo "------------------------------------------------------"
  
  python data/download_data.py --data_dir "$DATA_DIR"
  
  if [ $? -ne 0 ]; then
    echo "Error: Dataset download failed!"
    exit 1
  fi
  
  echo "Dataset download completed successfully."
else
  echo "Skipping dataset download step."
fi

# 2. Generate training data
if [ "$SKIP_GENERATION" = false ]; then
  echo ""
  echo "STEP 2: Generating training data..."
  echo "------------------------------------------------------"
  
  python src/data_generation/data_generator.py --config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR/data"
  
  if [ $? -ne 0 ]; then
    echo "Error: Data generation failed!"
    exit 1
  fi
  
  echo "Data generation completed successfully."
else
  echo "Skipping data generation step."
fi

# 3. Fine-tune the model
if [ "$SKIP_TRAINING" = false ]; then
  echo ""
  echo "STEP 3: Fine-tuning model..."
  echo "------------------------------------------------------"
  
  python src/training/finetune.py --config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR/model"
  
  if [ $? -ne 0 ]; then
    echo "Error: Model training failed!"
    exit 1
  fi
  
  echo "Model training completed successfully."
else
  echo "Skipping model training step."
fi

# 4. Evaluate the model
if [ "$SKIP_EVALUATION" = false ]; then
  echo ""
  echo "STEP 4: Evaluating model..."
  echo "------------------------------------------------------"
  
  python src/evaluation/evaluation.py --config "$CONFIG_FILE" --model_path "$OUTPUT_DIR/model" --output_path "$OUTPUT_DIR/evaluation"
  
  if [ $? -ne 0 ]; then
    echo "Error: Model evaluation failed!"
    exit 1
  fi
  
  echo "Model evaluation completed successfully."
else
  echo "Skipping model evaluation step."
fi

# Pipeline complete
echo ""
echo "======================================================"
echo "Pipeline execution completed successfully!"
echo "End time: $(date)"
echo "Results are available in: $OUTPUT_DIR"
echo "======================================================"