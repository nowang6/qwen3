"""Export Qwen3 model to ONNX format.
Reference: export_onnx.py for Qwen2 model export pattern.
"""

import os
import argparse
import torch
import onnx

from utils import load_qwen3_model_and_tokenizer
from llms_from_scratch.qwen3 import QWEN_CONFIG_06_B


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
onnx_model_dir = os.path.join(output_dir, "onnx")
if not os.path.exists(onnx_model_dir):
    os.mkdir(onnx_model_dir)
if len(os.listdir(onnx_model_dir)) > 0:
    print("found some file in {}, will clear it".format(onnx_model_dir))
    for temp_file in os.listdir(onnx_model_dir):
        temp_path = os.path.join(onnx_model_dir, temp_file)
        os.remove(temp_path)


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device_str",
        type=str,
        choices=["npu", "cuda", "cpu"],
        help="support npu, cuda, cpu",
        default="cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="support float16/float32, if use CPU, only support fp32",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="model and tokenizer path",
        default=os.path.join(project_dir, "models", "Qwen3-0.6B")
    )
    parser.add_argument(
        "--onnx_model_path",
        help="output onnx path",
        type=str,
        default=os.path.join(onnx_model_dir, "qwen3_0.6b.onnx")
    )
    parser.add_argument(
        "--seq_length",
        help="sequence length for export",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size for export",
        type=int,
        default=1,
    )
    return parser.parse_args()


def export_onnx(
    device_str: str,
    dtype: str,
    model_path: str,
    onnx_model_path: str,
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
    if device_str == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            raise ImportError("torch_npu is required for NPU device but not installed")
    if dtype == "float16":
        assert device_str.lower() != "cpu", "cpu not support fp16"
    elif dtype == "float32":
        pass
    else:
        raise Exception("unsupport dtype")

    device = torch.device(device_str)
    
    # Load model using utils function
    print(f"Loading model from {model_path}...")
    # Note: load_qwen3_model_and_tokenizer will auto-detect device (cuda/cpu)
    # We'll move it to the specified device afterwards
    model, tokenizer, loaded_device = load_qwen3_model_and_tokenizer(
        model_path=model_path,
        use_instruct_model=False,
        use_reasoning_model=False  # Use base model for export
    )
    
    # Move model to specified device (override auto-detected device)
    if device != loaded_device:
        print(f"Moving model from {loaded_device} to {device}...")
        model = model.to(device)
    if dtype == "float16":
        model = model.half()
    elif dtype == "float32":
        model = model.float()
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
    args = parser_arguments()
    
    # Use QWEN_CONFIG_06_B from the model
    # You can modify this to use different configs based on model_path
    model_config = QWEN_CONFIG_06_B
    
    print("Model configuration:")
    print(f"  vocab_size: {model_config['vocab_size']}")
    print(f"  context_length: {model_config['context_length']}")
    print(f"  emb_dim: {model_config['emb_dim']}")
    print(f"  n_layers: {model_config['n_layers']}")
    print(f"  n_heads: {model_config['n_heads']}")
    print(f"  n_kv_groups: {model_config['n_kv_groups']}")
    print(f"\nBegin export ONNX...")
    
    export_onnx(
        device_str=args.device_str,
        dtype=args.dtype,
        model_path=args.model_path,
        onnx_model_path=args.onnx_model_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        model_config=model_config,
    )

