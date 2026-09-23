import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

x = torch.tensor([1,2,3])
print(x.device)