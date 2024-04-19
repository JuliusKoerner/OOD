import torch

# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()

# Print whether CUDA is available or not
print("CUDA is available:" if is_cuda_available else "CUDA is not available.")
