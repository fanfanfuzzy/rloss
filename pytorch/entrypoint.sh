#!/usr/bin/env bash
set -e

# rloss Environment startup script for A100 + Ubuntu 22.04

echo "=== rloss Environment for A100 + Ubuntu 22.04 ==="
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# CUDA availability check (exit code 0 if available, 1 otherwise)
if python - << 'EOF'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
EOF
then
    echo "CUDA available: True"
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")')"
else
    echo "CUDA available: False"
fi

# Usage instructions
cat << 'USAGE'

Available commands:
  python test_environment.py      # Test environment setup  
  cd pytorch-deeplab_v3_plus      # Enter main directory  
  jupyter lab --ip=0.0.0.0 --allow-root  # Start Jupyter Lab

USAGE

# Execute passed command
exec "${@}"
