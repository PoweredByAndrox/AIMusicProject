#!/bin/bash
# Install All-In-One music structure analyzer in Docker container

echo "=========================================="
echo "Installing All-In-One Music Structure Analyzer"
echo "=========================================="

# Check CUDA version
echo ""
echo "Checking CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else '')"

# Install NATTEN (Neighborhood Attention)
echo ""
echo "Installing NATTEN..."
# For CUDA 12.1 + PyTorch 2.1
pip install natten -f https://shi-labs.com/natten/wheels/cu121/torch2.1/index.html 2>/dev/null || \
pip install natten -f https://shi-labs.com/natten/wheels/cu118/torch2.1/index.html 2>/dev/null || \
pip install natten -f https://shi-labs.com/natten/wheels/cu121/torch2.0/index.html 2>/dev/null || \
pip install natten 2>/dev/null || \
echo "NATTEN installation failed - will try to continue anyway"

# Install allin1
echo ""
echo "Installing allin1..."
pip install allin1

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import allin1; print('✓ allin1 installed successfully')" || echo "✗ allin1 installation failed"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
