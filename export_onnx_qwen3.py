"""Export Qwen3 model to ONNX format.
Reference: export_onnx.py for Qwen2 model export pattern.
"""

import os
import argparse
import torch
import torch.nn as nn
import onnx

from utils import load_qwen3_model_and_tokenizer
from llms_from_scratch.qwen3 import QWEN_CONFIG_06_B
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model as Qwen3ModelKVCache
from llms_from_scratch.kv_cache.qwen3 import load_weights_into_qwen
from llms_from_scratch.kv_cache.utils import KVCache
from safetensors.torch import load_file
from pathlib import Path


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
        default="models/Qwen3-0.6B"
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
    parser.add_argument(
        "--kv_cache_length",
        help="kv-cache length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--use_kv_cache",
        help="export model with KV cache support",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


class Qwen3ModelWithKVCacheWrapper(nn.Module):
    """Wrapper for Qwen3Model to export KV cache as tensors for ONNX."""
    
    def __init__(self, model, n_layers, n_kv_groups, head_dim, kv_cache_length, dtype):
        super().__init__()
        self.model = model
        self.n_layers = n_layers
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.kv_cache_length = kv_cache_length
        self.dtype = dtype
        
    def forward(self, input_ids, past_key_values):
        """
        Forward pass with KV cache as tensor input.
        
        Args:
            input_ids: (batch_size, seq_length) token ids
            past_key_values: (batch_size, kv_len, n_layers * 2 * n_kv_groups, head_dim)
                            Flattened KV cache from all layers
        
        Returns:
            logits: (batch_size, seq_length, vocab_size)
            out_key_values: (batch_size, kv_len + seq_length, n_layers * 2 * n_kv_groups, head_dim)
        """
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        # Convert past_key_values tensor to KVCache object
        cache = KVCache(self.n_layers)
        kv_len = past_key_values.shape[1]
        
        # Reshape past_key_values: (batch, kv_len, n_layers * 2 * n_kv_groups, head_dim)
        # -> list of (keys, values) tuples for each layer
        past_key_values_reshaped = past_key_values.reshape(
            batch_size, kv_len, self.n_layers, 2, self.n_kv_groups, self.head_dim
        )
        
        # Split into keys and values for each layer
        for layer_idx in range(self.n_layers):
            layer_kv = past_key_values_reshaped[:, :, layer_idx, :, :, :]
            # layer_kv shape: (batch, kv_len, 2, n_kv_groups, head_dim)
            keys = layer_kv[:, :, 0, :, :].transpose(1, 2)  # (batch, n_kv_groups, kv_len, head_dim)
            values = layer_kv[:, :, 1, :, :].transpose(1, 2)  # (batch, n_kv_groups, kv_len, head_dim)
            cache.update(layer_idx, (keys, values))
        
        # Set position tracking based on KV cache length
        # The model's current_pos should be set to kv_len before forward pass
        self.model.current_pos = kv_len
        
        # Forward pass
        logits = self.model(input_ids, cache=cache)
        
        # Collect output KV cache from all layers
        all_keys = []
        all_values = []
        for layer_idx in range(self.n_layers):
            keys, values = cache.get(layer_idx)
            # keys, values shape: (batch, n_kv_groups, new_kv_len, head_dim)
            # Transpose back to (batch, new_kv_len, n_kv_groups, head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            all_keys.append(keys)
            all_values.append(values)
        
        # Concatenate all layers: (batch, new_kv_len, n_layers * 2 * n_kv_groups, head_dim)
        new_kv_len = all_keys[0].shape[1]
        out_kv_list = []
        for layer_idx in range(self.n_layers):
            out_kv_list.append(all_keys[layer_idx])  # (batch, new_kv_len, n_kv_groups, head_dim)
            out_kv_list.append(all_values[layer_idx])  # (batch, new_kv_len, n_kv_groups, head_dim)
        
        # Stack and reshape: (batch, new_kv_len, n_layers * 2, n_kv_groups, head_dim)
        out_kv_stacked = torch.stack(out_kv_list, dim=2)
        out_kv_stacked = out_kv_stacked.reshape(
            batch_size, new_kv_len, self.n_layers * 2 * self.n_kv_groups, self.head_dim
        )
        
        return logits, out_kv_stacked


def export_onnx(
    device_str: str,
    dtype: str,
    model_path: str,
    onnx_model_path: str,
    seq_length: int,
    batch_size: int,
    model_config: dict,
    kv_cache_length: int = 2048,
    use_kv_cache: bool = False,
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
        kv_cache_length: Length of KV cache
        use_kv_cache: Whether to export with KV cache support
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
    
    # Load model
    print(f"Loading model from {model_path}...")
    if use_kv_cache:
        # Load KV cache version of the model
        model_file = Path(model_path, "model.safetensors")
        model = Qwen3ModelKVCache(model_config)
        weights_dict = load_file(model_file)
        load_weights_into_qwen(model, model_config, weights_dict)
        tokenizer = None  # We don't need tokenizer for export
    else:
        # Load model using utils function (no KV cache)
        model, tokenizer, loaded_device = load_qwen3_model_and_tokenizer(
            model_path=model_path,
            use_instruct_model=False,
            use_reasoning_model=False
        )
    
    # Move model to specified device
    model = model.to(device)
    if dtype == "float16":
        model = model.half()
        torch_dtype = torch.float16
    elif dtype == "float32":
        model = model.float()
        torch_dtype = torch.float32
    
    # Wrap model with KV cache wrapper if needed
    if use_kv_cache:
        head_dim = model_config.get("head_dim") or (model_config["emb_dim"] // model_config["n_heads"])
        model = Qwen3ModelWithKVCacheWrapper(
            model=model,
            n_layers=model_config["n_layers"],
            n_kv_groups=model_config["n_kv_groups"],
            head_dim=head_dim,
            kv_cache_length=kv_cache_length,
            dtype=torch_dtype
        )
    
    model.eval()

    # Prepare input and output names
    if use_kv_cache:
        input_names = ["input_ids", "past_key_values"]
        output_names = ["logits", "out_key_values"]
        
        head_dim = model_config.get("head_dim") or (model_config["emb_dim"] // model_config["n_heads"])
        n_kv_groups = model_config["n_kv_groups"]
        n_layers = model_config["n_layers"]
        
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "past_key_values": {0: "batch_size", 1: "kv_len"},
            "logits": {0: "batch_size", 1: "seq_length"},
            "out_key_values": {0: "batch_size", 1: "kv_len"},
        }
        
        # Create dummy inputs
        seq_len = 1  # For incremental inference
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)
        past_key_values = torch.rand(
            (batch_size, kv_cache_length, n_layers * 2 * n_kv_groups, head_dim),
            dtype=torch_dtype
        ).to(device)
        
        input_args = (input_ids, past_key_values)
        
        print(f"Model input shapes:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  past_key_values: {past_key_values.shape}")
    else:
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
        
        input_args = (input_ids,)
        print(f"Model input shape: {input_ids.shape}")
    
    print(f"Exporting to ONNX format...")
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            f=onnx_model_path,
            args=input_args,
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
        kv_cache_length=args.kv_cache_length,
        use_kv_cache=args.use_kv_cache,
    )

