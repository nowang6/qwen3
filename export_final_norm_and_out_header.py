from pathlib import Path
import torch
from safetensors.torch import load_file
import torch.nn as nn

from llms_from_scratch.qwen3_fixed_32_seq_len import Qwen3Model, QWEN_CONFIG_06_B_FIXED_32, load_weights_into_qwen, RMSNorm


class FinalNormAndOutHead(nn.Module):
    """Wrapper module that combines final_norm and out_head"""
    def __init__(self, final_norm, out_head):
        super().__init__()
        self.final_norm = final_norm
        self.out_head = out_head
    
    def forward(self, x):
        # x: [batch_size, 32, emb_dim]
        x = self.final_norm(x)  # [batch_size, 32, emb_dim]
        logits = self.out_head(x)  # [batch_size, 32, vocab_size]
        return logits


model_path = "models/Qwen3-0.6B"
output_path = "output"

model_file = Path(model_path, "model.safetensors")

print("Loading model with fixed sequence length of 32...")
model = Qwen3Model(QWEN_CONFIG_06_B_FIXED_32)
weights_dict = load_file(model_file)
load_weights_into_qwen(model, QWEN_CONFIG_06_B_FIXED_32, weights_dict)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")
model.to(device)
model.eval()  # Set to evaluation mode

# Export final_norm and out_head combined to ONNX format
print("Exporting final_norm and out_head to ONNX format...")
try:
    # Create wrapper module combining final_norm and out_head
    final_norm_and_out_head = FinalNormAndOutHead(model.final_norm, model.out_head)
    final_norm_and_out_head.eval()

    # Define dummy input: embeddings [batch_size, 32, emb_dim]
    dummy_input = torch.randn(1, 32, QWEN_CONFIG_06_B_FIXED_32['emb_dim'], device=device, dtype=torch.float32)

    # Define ONNX export path
    onnx_path = Path(output_path, "qwen3_0.6b_final_norm_and_out_head.onnx")

    # Export to ONNX with 1 input: (x,)
    torch.onnx.export(
        final_norm_and_out_head,
        (dummy_input,),  # Single input argument
        str(onnx_path),
        input_names=['embeddings'],
        output_names=['logits'],
        dynamic_axes={
            'embeddings': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Final norm and out head successfully exported to ONNX format: {onnx_path}")
    print(f"ONNX model file size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
    print("Note: The exported ONNX model requires 1 input: embeddings [batch_size, 32, emb_dim]")
    print("      Output: logits [batch_size, 32, vocab_size]")
except Exception as e:
    print(f"Error exporting to ONNX: {e}")
    import traceback
    traceback.print_exc()

