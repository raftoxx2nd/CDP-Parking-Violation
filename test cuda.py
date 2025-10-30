import torch
import sys

print(f"Python Executable: {sys.executable}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Install Location: {torch.__file__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")