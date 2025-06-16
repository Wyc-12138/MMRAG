# 在 Python 中执行以下命令
import torch
print(torch.__version__)          # 预期输出：2.1.2+
print(torch.cuda.is_available())  # 必须为 True
print(torch.version.cuda)         # 应与 CUDA 驱动版本匹配