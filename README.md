python export_onnx_qwen3.py --device_str cpu --dtype float32 --model_path models/Qwen3-0.6B --onnx_model_path output/onnx/qwen3_0.6b.onnx




```
./omg \
  --framework=5 \
  --model="qwen3_0.6B_embedding.onnx" \
  --output="qwen3_0.6B_embedding" \
  --input_shape="token_ids:1,-1" \
  --dynamic_dims "1;32"

```



```sh

./omg \
--om=qwen3_0.6B_embedding.om  \
--json=model.json \
--mode=1

```



```sh
./omg \
  --framework=5 \
  --model="simple_embedding_model_fp32.onnx" \
  --output="simple_embedding_model_fp32" \
  --input_shape="input_ids:1,32"

./omg \
  --framework=5 \
  --model="qwen3_0.6b_transformer_block_0.onnx" \
  --output="qwen3_0.6b_transformer_block_0" \
  --input_shape="embeddings:1,32,1024"
```