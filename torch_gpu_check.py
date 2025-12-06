import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")