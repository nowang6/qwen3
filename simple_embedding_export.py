#!/usr/bin/env python3
"""
Simple PyTorch Embedding to ONNX Export Script
Demonstrates how to export a PyTorch embedding layer to ONNX format using .venv environment
"""

import torch
import torch.nn as nn
from pathlib import Path
import time

def create_simple_embedding_model(vocab_size: int = 1000, embedding_dim: int = 128):
    """Create a simple PyTorch embedding model for demonstration."""
    class SimpleEmbeddingModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, input_ids):
            return self.embedding(input_ids)

    return SimpleEmbeddingModel(vocab_size, embedding_dim)

def export_embedding_to_onnx(
    model: nn.Module,
    output_path: str,
    vocab_size: int = 1000,
    max_seq_length: int = 512,
    batch_size: int = 1
):
    """Export PyTorch embedding model to ONNX format."""

    print(f"Exporting PyTorch embedding model to ONNX: {output_path}")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input for export
    dummy_input = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, max_seq_length),
        dtype=torch.int32
    )

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input_ids"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14,
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )

    print(f"✓ ONNX export completed successfully!")
    print(f"  Model saved to: {output_path}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")
    except Exception as e:
        print(f"⚠ ONNX model verification failed: {e}")

def test_exported_model(onnx_path: str, vocab_size: int = 1000, embedding_dim: int = 128):
    """Test the exported ONNX model."""
    try:
        import onnxruntime as ort
        import numpy as np

        print(f"\nTesting exported ONNX model...")

        # Create ONNX runtime session
        session = ort.InferenceSession(onnx_path)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Create test input
        test_input = np.random.randint(0, vocab_size, size=(1, 10), dtype=np.int32)

        # Run inference
        start_time = time.time()
        outputs = session.run([output_name], {input_name: test_input})
        inference_time = time.time() - start_time

        embeddings = outputs[0]

        print(f"✓ ONNX inference successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Output dtype: {embeddings.dtype}")
        print(f"  Inference time: {inference_time:.4f}s")

        return embeddings

    except ImportError:
        print("⚠ onnxruntime not available for testing")
        return None

def main():
    """Main function to demonstrate PyTorch embedding to ONNX export."""

    print("=" * 60)
    print("PyTorch Embedding to ONNX Export Demo")
    print("=" * 60)

    # Configuration
    vocab_size = 1000
    embedding_dim = 128
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "simple_embedding_model.onnx"

    # Step 1: Create PyTorch embedding model
    print(f"\n1. Creating PyTorch embedding model...")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Embedding dimension: {embedding_dim}")

    model = create_simple_embedding_model(vocab_size, embedding_dim)
    
    # odel = model.half()

    # Initialize with random weights
    with torch.no_grad():
        model.embedding.weight.normal_(mean=0.0, std=0.02)

    print(f"✓ Model created successfully!")

    # Step 2: Test PyTorch model
    print(f"\n2. Testing PyTorch model...")
    test_input = torch.randint(0, vocab_size, (1, 5), dtype=torch.int32)
    with torch.no_grad():
        pytorch_output = model(test_input)

    print(f"✓ PyTorch model test successful!")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {pytorch_output.shape}")
    print(f"  Output dtype: {pytorch_output.dtype}")

    # Step 3: Export to ONNX
    print(f"\n3. Exporting to ONNX format...")
    export_embedding_to_onnx(model, str(onnx_path), vocab_size)

    # Step 4: Test exported model
    print(f"\n4. Testing exported ONNX model...")
    onnx_output = test_exported_model(str(onnx_path), vocab_size, embedding_dim)

    # Step 5: Compare outputs (if both available)
    if onnx_output is not None:
        print(f"\n5. Comparing PyTorch vs ONNX outputs...")

        # Convert PyTorch output to numpy for comparison
        pytorch_np = pytorch_output.numpy()

        # Take a sample for comparison (first 5 tokens)
        if onnx_output.shape[1] >= 5:
            onnx_sample = onnx_output[0, :5, :]
            pytorch_sample = pytorch_np[0, :5, :]

            # Calculate cosine similarity
            import numpy as np
            pytorch_flat = pytorch_sample.flatten()
            onnx_flat = onnx_sample.flatten()

            cosine_sim = np.dot(pytorch_flat, onnx_flat) / (
                np.linalg.norm(pytorch_flat) * np.linalg.norm(onnx_flat)
            )

            print(f"✓ Comparison completed!")
            print(f"  Cosine similarity: {cosine_sim:.6f}")

            if cosine_sim > 0.99:
                print("  ✓ Outputs are very similar (similarity > 0.99)")
            else:
                print("  ⚠ Outputs differ somewhat")

    print(f"\n" + "=" * 60)
    print("Export completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
    