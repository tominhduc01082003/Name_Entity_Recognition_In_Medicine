import torch
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    print(f"GPU: {device_name} ({total_vram} GB VRAM)")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Ko phát hiện GPU")
