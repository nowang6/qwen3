import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from llms_from_scratch.qwen3 import (
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)


def select_providers():
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


def main():
    model_path = Path("models", "Qwen3-0.6B")
    onnx_model_path = Path("output/qwen3_0.6b.onnx")

    print("Loading tokenizer...")
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=Path(model_path, "tokenizer.json"),
        repo_id=str(model_path),
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )

    print("Initializing ONNX Runtime session...")
    session = ort.InferenceSession(
        onnx_model_path.as_posix(),
        providers=select_providers(),
    )
    output_name = session.get_outputs()[0].name

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    print(f"Input prompt: {prompt}")
    print(f"Input token count: {len(input_token_ids)}")

    torch_context_limit = QWEN_CONFIG_06_B["context_length"]
    actual_context_size = min(len(input_token_ids) + 150, 2048)
    actual_context_size = min(actual_context_size, torch_context_limit)
    print(
        f"Using context size: {actual_context_size} "
        f"(max: {torch_context_limit})"
    )

    max_new_tokens = 150
    eos_id = tokenizer.eos_token_id

    tokens = np.array(input_token_ids, dtype=np.int64)

    def run_session(token_array: np.ndarray) -> np.ndarray:
        inputs = {"input_ids": token_array[np.newaxis, :]}
        logits = session.run([output_name], inputs)[0]
        return logits

    print("\nStarting ONNX Runtime generation...")
    print("Generated text: ", end="", flush=True)
    start = time.time()

    for _ in range(max_new_tokens):
        window = tokens[-actual_context_size:]
        logits = run_session(window)
        next_token_id = int(np.argmax(logits[0, -1, :]))

        tokens = np.append(tokens, next_token_id)

        if eos_id is not None and next_token_id == eos_id:
            break

        token_text = tokenizer.decode([next_token_id])
        print(token_text, end="", flush=True)

    elapsed = time.time() - start
    print()
    print(f"Time: {elapsed:.2f} sec")
    if elapsed > 0:
        print(f"{int(len(tokens) / elapsed)} tokens/sec")

    output_text = tokenizer.decode(tokens.tolist())
    print("\n\nOutput text:\n")
    print(output_text + "...")


if __name__ == "__main__":
    main()

