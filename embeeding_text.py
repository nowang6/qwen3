
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
import onnxruntime as ort
import numpy as np
import onnx
import onnxslim

from llms_from_scratch.qwen3 import Qwen3Tokenizer, QWEN_CONFIG_06_B


def load_embedding_weights(model_path: Path) -> nn.Embedding:
    cfg = QWEN_CONFIG_06_B
    emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

    model_file = model_path / "model.safetensors"
    print(f"Loading embedding weights from {model_file} ...")
    start = time.time()
    weights_dict = load_file(model_file)
    torch.manual_seed(0)

    with torch.no_grad():
        emb.weight.copy_(weights_dict["model.embed_tokens.weight"])

    elapsed = time.time() - start
    print(f"Loaded embedding weights in {elapsed:.2f}s")
    return emb


def test_embedding(prompt: str, model_path: Path, use_float32: bool = False) -> torch.Tensor:
    emb = load_embedding_weights(model_path)

    # Convert to float32 if needed for ONNX compatibility
    if use_float32:
        emb.weight.data = emb.weight.data.to(torch.float32)

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=model_path / "tokenizer.json",
        repo_id=str(model_path),
        apply_chat_template=False,
        add_generation_prompt=False,
    )

    token_ids = tokenizer.encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Tokenized length: {len(token_ids)}")

    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    embeddings = emb(token_tensor)
    print(f"Embedding output shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    return embeddings


def export_embedding_to_onnx(
    model_path: Path,
    onnx_output_path: Path,
    opset_version: int = 10,
) -> None:
    """Export the Qwen3 token embedding layer to ONNX."""
    emb = load_embedding_weights(model_path)
    emb.eval()

    # Keep original float16 precision for ONNX export
    # emb.weight.data remains in float16

    # Dummy input: a small sequence of token ids
    # Use int32 instead of int64 (torch.long) for OMG/ATC compatibility
    dummy_input = torch.randint(
        low=0,
        high=QWEN_CONFIG_06_B["vocab_size"],
        size=(1, 10),  # (batch_size, seq_len) - seq_len fixed to 10
        dtype=torch.int32,
    )

    print(f"Exporting embedding to ONNX: {onnx_output_path}")
    torch.onnx.export(
        emb,
        dummy_input,
        onnx_output_path,
        input_names=["token_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "token_ids": {0: "batch_size"},  # seq_len fixed to 10
            "embeddings": {0: "batch_size"},  # seq_len fixed to 10
        },
        opset_version=opset_version,
        # Specify output type as float16
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        verbose=False,
    )
    print("ONNX export finished.")


def slim_onnx_model(
    onnx_model_path: Path,
    slim_output_path: Optional[Path] = None,
) -> Path:
    """Use onnxslim to simplify the exported ONNX model."""
    if slim_output_path is None:
        slim_output_path = onnx_model_path.with_suffix(".slim.onnx")

    print(f"Loading ONNX model for slimming: {onnx_model_path}")
    model = onnx.load(str(onnx_model_path))

    print("Running onnxslim.slim ...")
    slimmed_model = onnxslim.slim(model)

    # Fix Embedding input type from int64 to int32 for OMG/ATC compatibility
    for input_tensor in slimmed_model.graph.input:
        if input_tensor.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            print(f"Converting input '{input_tensor.name}' from int64 to int32")
            input_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT32

    print(f"Saving slimmed model to {slim_output_path}")
    onnx.save(slimmed_model, str(slim_output_path))

    print("ONNX slimming finished.")
    return slim_output_path


def test_embedding_with_onnx(
    prompt: str,
    model_path: Path,
    onnx_model_path: Path
) -> None:
    """Test embeddings using exported ONNX model."""
    # Load tokenizer
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=model_path / "tokenizer.json",
        repo_id=str(model_path),
        apply_chat_template=False,
        add_generation_prompt=False,
    )

    # Tokenize input
    token_ids = tokenizer.encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Tokenized length: {len(token_ids)}")

    # Prepare input for ONNX (use int32 for OMG/ATC compatibility)
    token_tensor = torch.tensor([token_ids], dtype=torch.int32)  # Add batch dimension

    # Load ONNX model
    print(f"Loading ONNX model from {onnx_model_path}...")
    session = ort.InferenceSession(str(onnx_model_path))

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Debug: print input/output info
    print(f"Input name: {input_name}, type: {session.get_inputs()[0].type}")
    print(f"Output name: {output_name}, type: {session.get_outputs()[0].type}")
    print(f"Input tensor shape: {token_tensor.shape}, dtype: {token_tensor.dtype}")

    print("Running ONNX inference...")
    start_time = time.time()

    # Convert to appropriate numpy type for ONNX (int32 for OMG/ATC compatibility)
    if token_tensor.dtype == torch.int32:
        input_data = token_tensor.numpy().astype(np.int32)
    elif token_tensor.dtype == torch.int64:
        input_data = token_tensor.numpy().astype(np.int64)
    else:
        input_data = token_tensor.numpy()

    onnx_output = session.run([output_name], {input_name: input_data})
    elapsed_time = time.time() - start_time

    embeddings_onnx = torch.from_numpy(onnx_output[0])

    print(f"ONNX embedding output shape: {embeddings_onnx.shape}")
    print(f"ONNX embedding dtype: {embeddings_onnx.dtype}")
    print(f"ONNX inference time: {elapsed_time:.4f}s")

    # Compare with PyTorch implementation (using original float16)
    print("\nComparing with PyTorch implementation...")
    embeddings_pytorch = test_embedding(prompt, model_path, use_float32=True)

    # Calculate similarity
    similarity = torch.nn.functional.cosine_similarity(
        embeddings_pytorch.flatten(),
        embeddings_onnx[0].flatten(),
        dim=0
    )

    print(f"Cosine similarity between PyTorch and ONNX outputs: {similarity.item():.6f}")

    # Check if outputs are close
    if torch.allclose(embeddings_pytorch, embeddings_onnx[0], atol=1e-5):
        print("✓ ONNX and PyTorch outputs match within tolerance")
    else:
        print("⚠ ONNX and PyTorch outputs differ slightly")

    return embeddings_onnx


if __name__ == "__main__":
    MODEL_DIR = Path("models/Qwen3-0.6B")
    SAMPLE_PROMPT = "给我介绍一下大型语言模型。"

    # 1) 测试一次 embedding
    print("=" * 50)
    print("Testing PyTorch embedding:")
    print("=" * 50)
    embeddings = test_embedding(SAMPLE_PROMPT, MODEL_DIR)
    print(f"First token embedding snippet: {embeddings[0][:5]}")

    # 2) 导出 ONNX
    print("\n" + "=" * 50)
    print("Exporting to ONNX:")
    print("=" * 50)
    ONNX_PATH = Path("output/qwen3_0.6B_embedding.onnx")
    export_embedding_to_onnx(MODEL_DIR, ONNX_PATH)

    # 2.5) onnxslim 简化
    print("\n" + "=" * 50)
    print("Slimming ONNX model:")
    print("=" * 50)
    SLIM_ONNX_PATH = Path("output/qwen3_0.6B_embedding.slim.onnx")
    slim_onnx_model(ONNX_PATH, SLIM_ONNX_PATH)

    # 3) 测试 ONNX 模型
    print("\n" + "=" * 50)
    print("Testing ONNX model:")
    print("=" * 50)
    test_embedding_with_onnx(SAMPLE_PROMPT, MODEL_DIR, SLIM_ONNX_PATH)