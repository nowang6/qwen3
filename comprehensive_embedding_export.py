#!/usr/bin/env python3
"""
Comprehensive PyTorch Embedding to ONNX Export Script
Supports both float32 and float16 export formats with configurable parameters
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
from typing import Optional, Union, Dict, Any
import json

def create_embedding_model(
    vocab_size: int = 1000,
    embedding_dim: int = 128,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    dtype: torch.dtype = torch.float32
) -> nn.Module:
    """Create a configurable PyTorch embedding model."""

    class EmbeddingModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, padding_idx, max_norm,
                     norm_type, scale_grad_by_freq, sparse, dtype):
            super().__init__()
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse,
                dtype=dtype
            )

        def forward(self, input_ids):
            return self.embedding(input_ids)

    return EmbeddingModel(vocab_size, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, dtype)

def export_embedding_to_onnx(
    model: nn.Module,
    output_path: str,
    vocab_size: int = 1000,
    max_seq_length: int = 512,
    batch_size: int = 1,
    export_dtype: str = "float32",
    opset_version: int = 14,
    dynamic_batch: bool = True,
    dynamic_sequence: bool = True,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export PyTorch embedding model to ONNX format with configurable parameters.

    Args:
        model: PyTorch embedding model
        output_path: Path to save ONNX model
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length for dummy input
        batch_size: Batch size for dummy input
        export_dtype: Export data type ("float32" or "float16")
        opset_version: ONNX opset version
        dynamic_batch: Whether to enable dynamic batch size
        dynamic_sequence: Whether to enable dynamic sequence length
        custom_metadata: Custom metadata to add to ONNX model

    Returns:
        Dict containing export information
    """

    print(f"Exporting PyTorch embedding model to ONNX: {output_path}")
    print(f"  Export dtype: {export_dtype}")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic batch: {dynamic_batch}")
    print(f"  Dynamic sequence: {dynamic_sequence}")

    # Set model to evaluation mode
    model.eval()

    # Convert model to appropriate dtype
    if export_dtype == "float16":
        model = model.half()
        dummy_dtype = torch.float16
    else:
        dummy_dtype = torch.float32

    # Create dummy input
    dummy_input = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, max_seq_length),
        dtype=torch.int32
    )

    # Prepare dynamic axes
    dynamic_axes = {}
    if dynamic_batch or dynamic_sequence:
        dynamic_axes["input_ids"] = {}
        dynamic_axes["embeddings"] = {}
        if dynamic_batch:
            dynamic_axes["input_ids"][0] = "batch_size"
            dynamic_axes["embeddings"][0] = "batch_size"
        if dynamic_sequence:
            dynamic_axes["input_ids"][1] = "sequence_length"
            dynamic_axes["embeddings"][1] = "sequence_length"

    # Export to ONNX
    start_time = time.time()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input_ids"],
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            verbose=False
        )
    export_time = time.time() - start_time

    print(f"✓ ONNX export completed in {export_time:.3f}s!")
    print(f"  Model saved to: {output_path}")

    # Add custom metadata if provided
    if custom_metadata:
        try:
            import onnx
            onnx_model = onnx.load(output_path)

            # Create metadata properties
            for key, value in custom_metadata.items():
                meta = onnx_model.metadata_props.add()
                meta.key = str(key)
                meta.value = str(value)

            onnx.save(onnx_model, output_path)
            print(f"✓ Added custom metadata: {list(custom_metadata.keys())}")
        except Exception as e:
            print(f"⚠ Failed to add metadata: {e}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")

        # Get model info
        model_info = {
            "export_path": output_path,
            "export_dtype": export_dtype,
            "opset_version": opset_version,
            "export_time": export_time,
            "verification_passed": True,
            "input_shape": [batch_size, max_seq_length],
            "output_shape": [batch_size, max_seq_length, model.embedding.embedding_dim],
            "vocab_size": vocab_size,
            "embedding_dim": model.embedding.embedding_dim,
            "parameters": sum(p.numel() for p in model.parameters())
        }

        return model_info

    except Exception as e:
        print(f"⚠ ONNX model verification failed: {e}")
        return {"verification_passed": False, "error": str(e)}

def test_exported_model(
    onnx_path: str,
    test_inputs: Optional[list] = None,
    expected_dtype: str = "float32"
) -> Dict[str, Any]:
    """
    Test the exported ONNX model with various inputs.

    Args:
        onnx_path: Path to ONNX model
        test_inputs: List of test inputs (if None, generates random inputs)
        expected_dtype: Expected output data type

    Returns:
        Dict containing test results
    """
    try:
        import onnxruntime as ort
        import numpy as np

        print(f"\nTesting exported ONNX model: {onnx_path}")

        # Create ONNX runtime session
        session = ort.InferenceSession(onnx_path)

        # Get input and output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        print(f"  Input name: {input_info.name}, shape: {input_info.shape}, dtype: {input_info.type}")
        print(f"  Output name: {output_info.name}, shape: {output_info.shape}, dtype: {output_info.type}")

        # Generate test inputs if not provided
        if test_inputs is None:
            # Test with different sequence lengths
            test_lengths = [1, 5, 10, 20]
            vocab_size = 1000  # Default vocab size
            test_inputs = []
            for length in test_lengths:
                test_input = np.random.randint(0, vocab_size, size=(1, length), dtype=np.int32)
                test_inputs.append(test_input)

        results = []

        for i, test_input in enumerate(test_inputs):
            print(f"\n  Test {i+1}: Input shape {test_input.shape}")

            # Run inference
            start_time = time.time()
            outputs = session.run([output_info.name], {input_info.name: test_input})
            inference_time = time.time() - start_time

            embeddings = outputs[0]

            test_result = {
                "test_id": i+1,
                "input_shape": test_input.shape,
                "output_shape": embeddings.shape,
                "output_dtype": str(embeddings.dtype),
                "inference_time": inference_time,
                "embeddings_mean": float(np.mean(embeddings)),
                "embeddings_std": float(np.std(embeddings))
            }

            results.append(test_result)

            print(f"    ✓ Output shape: {embeddings.shape}")
            print(f"    ✓ Output dtype: {embeddings.dtype}")
            print(f"    ✓ Inference time: {inference_time:.4f}s")
            print(f"    ✓ Embeddings mean: {test_result['embeddings_mean']:.6f}, std: {test_result['embeddings_std']:.6f}")

        # Summary
        avg_inference_time = np.mean([r["inference_time"] for r in results])
        print(f"\n✓ All tests completed!")
        print(f"  Average inference time: {avg_inference_time:.4f}s")
        print(f"  Tests passed: {len(results)}")

        return {
            "tests_passed": len(results),
            "results": results,
            "average_inference_time": avg_inference_time,
            "model_path": onnx_path
        }

    except ImportError:
        print("⚠ onnxruntime not available for testing")
        return {"error": "onnxruntime not available"}
    except Exception as e:
        print(f"⚠ Testing failed: {e}")
        return {"error": str(e)}

def compare_models(
    pytorch_model: nn.Module,
    onnx_path: str,
    test_input: torch.Tensor,
    export_dtype: str = "float32"
) -> Dict[str, Any]:
    """
    Compare PyTorch and ONNX model outputs.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        test_input: Input tensor for testing
        export_dtype: Export data type

    Returns:
        Dict containing comparison results
    """
    try:
        import onnxruntime as ort
        import numpy as np

        print(f"\nComparing PyTorch vs ONNX outputs...")

        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)

        # Convert to numpy
        pytorch_np = pytorch_output.numpy()
        if export_dtype == "float16":
            pytorch_np = pytorch_np.astype(np.float16)

        # Get ONNX output
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        onnx_output = session.run([output_name], {input_name: test_input.numpy().astype(np.int32)})[0]

        # Compare shapes
        shapes_match = pytorch_np.shape == onnx_output.shape
        print(f"  Shapes match: {shapes_match}")
        print(f"  PyTorch shape: {pytorch_np.shape}")
        print(f"  ONNX shape: {onnx_output.shape}")

        # Calculate metrics
        pytorch_flat = pytorch_np.flatten()
        onnx_flat = onnx_output.flatten()

        # Cosine similarity
        cosine_sim = np.dot(pytorch_flat, onnx_flat) / (
            np.linalg.norm(pytorch_flat) * np.linalg.norm(onnx_flat)
        )

        # Mean squared error
        mse = np.mean((pytorch_flat - onnx_flat) ** 2)

        # Mean absolute error
        mae = np.mean(np.abs(pytorch_flat - onnx_flat))

        # Relative error
        relative_error = mae / (np.mean(np.abs(pytorch_flat)) + 1e-8)

        print(f"  Cosine similarity: {cosine_sim:.6f}")
        print(f"  Mean squared error: {mse:.8f}")
        print(f"  Mean absolute error: {mae:.8f}")
        print(f"  Relative error: {relative_error:.6f}")

        # Quality assessment
        quality = "excellent" if cosine_sim > 0.999 else \
                 "good" if cosine_sim > 0.99 else \
                 "fair" if cosine_sim > 0.95 else "poor"

        print(f"  Output quality: {quality}")

        return {
            "shapes_match": shapes_match,
            "cosine_similarity": float(cosine_sim),
            "mse": float(mse),
            "mae": float(mae),
            "relative_error": float(relative_error),
            "quality": quality,
            "export_dtype": export_dtype
        }

    except Exception as e:
        print(f"⚠ Comparison failed: {e}")
        return {"error": str(e)}

def main():
    """Main function demonstrating comprehensive embedding export."""

    print("=" * 80)
    print("Comprehensive PyTorch Embedding to ONNX Export Demo")
    print("Supports Float32 and Float16 Export Formats")
    print("=" * 80)

    # Configuration
    config = {
        "vocab_size": 1000,
        "embedding_dim": 256,
        "max_seq_length": 128,
        "batch_size": 1,
        "opset_version": 14,
        "output_dir": Path("output")
    }

    # Create output directory
    config["output_dir"].mkdir(exist_ok=True)

    # Test both float32 and float16 exports
    export_configs = [
        {
            "name": "float32",
            "dtype": torch.float32,
            "export_dtype": "float32",
            "filename": "embedding_model_float32.onnx"
        },
        {
            "name": "float16",
            "dtype": torch.float16,
            "export_dtype": "float16",
            "filename": "embedding_model_float16.onnx"
        }
    ]

    results = []

    for export_config in export_configs:
        print(f"\n{'='*60}")
        print(f"Exporting {export_config['name'].upper()} Model")
        print(f"{'='*60}")

        # Step 1: Create PyTorch model
        print(f"\n1. Creating PyTorch {export_config['name']} embedding model...")
        model = create_embedding_model(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            dtype=export_config["dtype"]
        )

        # Initialize with random weights
        with torch.no_grad():
            model.embedding.weight.normal_(mean=0.0, std=0.02)

        print(f"✓ Model created successfully!")
        print(f"  Vocab size: {config['vocab_size']}")
        print(f"  Embedding dim: {config['embedding_dim']}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Step 2: Export to ONNX
        print(f"\n2. Exporting to ONNX ({export_config['export_dtype']})...")
        onnx_path = config["output_dir"] / export_config["filename"]

        export_info = export_embedding_to_onnx(
            model=model,
            output_path=str(onnx_path),
            vocab_size=config["vocab_size"],
            max_seq_length=config["max_seq_length"],
            batch_size=config["batch_size"],
            export_dtype=export_config["export_dtype"],
            opset_version=config["opset_version"],
            dynamic_batch=True,
            dynamic_sequence=True,
            custom_metadata={
                "model_type": "embedding",
                "export_format": export_config["export_dtype"],
                "vocab_size": str(config["vocab_size"]),
                "embedding_dim": str(config["embedding_dim"]),
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

        # Step 3: Test exported model
        print(f"\n3. Testing exported ONNX model...")
        test_results = test_exported_model(
            str(onnx_path),
            expected_dtype=export_config["export_dtype"]
        )

        # Step 4: Compare with PyTorch (for verification)
        print(f"\n4. Comparing PyTorch vs ONNX outputs...")
        test_input = torch.randint(0, config["vocab_size"], (1, 10), dtype=torch.int32)
        comparison_results = compare_models(
            model, str(onnx_path), test_input, export_config["export_dtype"]
        )

        # Store results
        result = {
            "export_config": export_config,
            "export_info": export_info,
            "test_results": test_results,
            "comparison_results": comparison_results
        }
        results.append(result)

    # Final summary
    print(f"\n{'='*80}")
    print("Export Summary")
    print(f"{'='*80}")

    for i, result in enumerate(results):
        export_config = result["export_config"]
        export_info = result["export_info"]
        comparison = result["comparison_results"]

        print(f"\n{i+1}. {export_config['name'].upper()} Model:")
        print(f"   ONNX file: {export_info['export_path']}")
        print(f"   Export time: {export_info['export_time']:.3f}s")
        print(f"   Parameters: {export_info['parameters']:,}")
        print(f"   Quality: {comparison.get('quality', 'N/A')}")
        print(f"   Cosine similarity: {comparison.get('cosine_similarity', 'N/A'):.6f}")

    print(f"\n{'='*80}")
    print("All exports completed successfully!")
    print(f"Models saved to: {config['output_dir']}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()