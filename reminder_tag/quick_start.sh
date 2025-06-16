#!/bin/bash
# Quick Start Script for Visual Dialogue Fine-tuning Project

echo "======================================================"
echo "  Visual Dialogue Fine-tuning Project Quick Start"
echo "======================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if conda/pip is available
if ! command -v conda &> /dev/null && ! command -v pip &> /dev/null; then
    echo "Error: conda or pip is required but not found."
    exit 1
fi

echo "✓ Python environment check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "visdial_env" ]; then
    echo "Creating virtual environment..."
    if command -v conda &> /dev/null; then
        conda create -n visdial python=3.9 -y
        echo "Activate with: conda activate visdial"
    else
        python3 -m venv visdial_env
        echo "Activate with: source visdial_env/bin/activate"
    fi
fi

# Create necessary directories
echo "Setting up project directories..."
mkdir -p data/{visdial/{data,images},coco/{train2014,val2014,annotations}}
mkdir -p outputs/{models,data,evaluation,logs}
mkdir -p configs
mkdir -p src/{data_generation,training,inference,evaluation,utils}
mkdir -p scripts
mkdir -p tests

echo "✓ Directory structure created"

# Create a sample configuration if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "Creating sample configuration file..."
    cp config.yaml configs/sample_config.yaml 2>/dev/null || echo "Note: Copy config.yaml to configs/ manually"
fi

# Make scripts executable
echo "Setting script permissions..."
chmod +x run_pipeline.sh 2>/dev/null
chmod +x scripts/*.sh 2>/dev/null

echo ""
echo "======================================================"
echo "  Setup Complete! Next Steps:"
echo "======================================================"
echo ""
echo "1. Activate your virtual environment:"
echo "   conda activate visdial  # or source visdial_env/bin/activate"
echo ""
echo "2. Install dependencies:"
echo "   pip install -r requirements.txt"
echo "   pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
echo ""
echo "3. Download datasets:"
echo "   python data/download_data.py"
echo ""
echo "4. Run the complete pipeline:"
echo "   ./run_pipeline.sh"
echo ""
echo "Or run individual steps:"
echo "   python example_training.py --config config.yaml"
echo "   python evaluation_script.py --predictions_path outputs/predictions.json --ground_truth_path data/visdial/data/visdial_1.0_val.json"
echo ""
echo "For more details, see README.md"
echo "======================================================"