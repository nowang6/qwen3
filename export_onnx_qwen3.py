"""Export Qwen3 model to ONNX format.
Reference: export_onnx.py for Qwen2 model export pattern.
"""

import os
import argparse
import torch
import onnx

from utils import load_qwen3_model_and_tokenizer
from llms_from_scratch.qwen3 import QWEN_CONFIG_06_B



onnx_model_path = "out.onnx"


def export_onnx(
    model : None,
    device_str: str,
    seq_length: int,
    batch_size: int,
    model_config: dict,
):
    """Export Qwen3 model to ONNX format.
    
    Args:
        device_str: Device to use (npu/cuda/cpu)
        dtype: Data type (float16/float32)
        model_path: Path to the model directory
        onnx_model_path: Output ONNX model path
        seq_length: Sequence length for export
        batch_size: Batch size for export
        model_config: Model configuration dict
    """
   

    device = torch.device(device_str)
    model.eval()

    # Prepare input
    input_names = ["input_ids"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size", 1: "seq_length"},
    }
    
    # Create dummy input
    input_ids = torch.randint(
        0, 
        model_config["vocab_size"], 
        (batch_size, seq_length),
        dtype=torch.long
    ).to(device)
    
    print(f"Model input shape: {input_ids.shape}")
    print(f"Exporting to ONNX format...")
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            f=onnx_model_path,
            args=(input_ids,),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            opset_version=14,
            export_params=True,
            verbose=False
        )
    
    print(f"ONNX model exported to: {onnx_model_path}")
    
    # Verify the exported model
    try:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except Exception as e:
        print(f"Warning: ONNX model verification failed: {e}")


if __name__ == "__main__":
    model, tokenier = load_qwen3_model_and_tokenizer(model_path="models/Qwen3-0.6B")
    
    export_onnx(
        model=model,
        device_str="cpu",
        seq_length=2048,
        batch_size=1,
        model_config=QWEN_CONFIG_06_B,
    )

