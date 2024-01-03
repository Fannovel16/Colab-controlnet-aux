import torch
def get_torch_device():
    return torch.device("cuda")
def soft_empty_cache():
    torch.cuda.empty_cache()