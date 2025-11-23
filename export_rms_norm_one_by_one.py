import torch
import torch.nn as nn
from pathlib import Path
from llms_from_scratch.qwen3_fixed_32_seq_len import QWEN_CONFIG_06_B_FIXED_32

class VarianceRsqrtOp(nn.Module):
    """合并方差计算、rsqrt 计算和乘法:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    rsqrt_result = torch.rsqrt(variance + self.eps)
    norm_x = x * rsqrt_result
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rsqrt_result = torch.rsqrt(variance + self.eps)
        # 显式扩展 rsqrt_result 到与 x 相同的形状，避免广播导致的NPU算子不兼容
        rsqrt_result = rsqrt_result.expand_as(x)
        norm_x = x * rsqrt_result
        return norm_x

out_path = "output"
Path(out_path).mkdir(parents=True, exist_ok=True)

# 创建模型实例
eps = 1e-6
variance_rsqrt_op = VarianceRsqrtOp(eps=eps)
variance_rsqrt_op.eval()

# 设置设备
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")
variance_rsqrt_op.to(device)

# 定义输入: [batch_size, seq_len, emb_dim]
emb_dim = QWEN_CONFIG_06_B_FIXED_32['emb_dim']  # 使用模型配置中的emb_dim (1024)
seq_len = 32
batch_size = 1
dummy_input = torch.randn(batch_size, seq_len, emb_dim, device=device, dtype=torch.float32)

# 定义ONNX导出路径
onnx_path = Path(out_path, "variance_rsqrt_op.onnx")

print("Exporting variance + rsqrt + multiply operator to ONNX format...")
try:
    # 导出到ONNX
    torch.onnx.export(
        variance_rsqrt_op,
        (dummy_input,),
        str(onnx_path),
        input_names=['x'],
        output_names=['norm_x'],
        dynamic_axes={
            'x': {0: 'batch_size', 1: 'seq_len'},
            'norm_x': {0: 'batch_size', 1: 'seq_len'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Variance + rsqrt + multiply operator successfully exported to ONNX format: {onnx_path}")
    print(f"ONNX model file size: {onnx_path.stat().st_size / 1024:.2f} KB")
    print("Input shape: [batch_size, seq_len, emb_dim]")
    print("Output shape: [batch_size, seq_len, emb_dim]")
    print(f"Epsilon value: {eps}")
except Exception as e:
    print(f"Error exporting to ONNX: {e}")
    import traceback
    traceback.print_exc()
