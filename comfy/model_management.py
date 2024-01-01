import torch
def get_torch_device():
    return "cuda"
def soft_empty_cache():
    torch.cuda.empty_cache()