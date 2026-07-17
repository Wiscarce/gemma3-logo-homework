echo "===== Python ====="
python --version

echo "===== OS ====="
cat /etc/os-release

echo "===== GPU ====="
nvidia-smi

echo "===== Torch ====="
python -c "
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
"

echo "===== Packages ====="
pip show transformers
pip show peft
pip show ms-swift
pip show datasets
pip show accelerate