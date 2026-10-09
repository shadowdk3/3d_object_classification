# ONNX

## Environmet

- Python: Python 3.12
- wsl: Ubuntu 24.04
- ONNX: 1.17.0
- ONNX Runtime GPU: 1.20.2
- ONNX script: 0.7.2

## Preset

```
python -m pip install onnx==1.17.0 onnxruntime-gpu==1.20.2
```

## Configure CUDA Libraries

Add the NVIDIA Python package library directories to `LD_LIBRARY_PATH` so ONNX Runtime can locate CUDA and cuDNN libraries.

```
export LD_LIBRARY_PATH="$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -type d -name lib -printf '%p:')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

**Run this command in each new terminal session before GPU inference.**

## Check ONNX Runtime

```
python -c "import onnxruntime as ort; print('ONNX Runtime:', ort.__version__); print('Available providers:', ort.get_available_providers())"
```

Confirm that CUDAExecutionProvider appears in the available providers.

## Convert .pth to .onnx

Export the trained PointNet checkpoint using the project's export script:

```
python -m utils.export_onnx models/pointnet_model_best.pth
```

Input model:
`models/pointnet_model_best.pth`

Output model:
`models/pointnet_model_best.onnx`

The export script uses ONNX 17 and supports dynamic batch sizes. The model input shape is [batch_size, 1024, 3], with 40 output class scores for ModelNet40.

## Run ONNX Inference on GPU

```
python -m src.evaluate_onnx
```

Expected output includes:

```
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Actual: car
Predicted: car
```

CUDA is available as an execution provider, with CPU as a fallback.

## Troubleshooting

If CUDA or cuDNN libraries cannot be found:

Re-run the LD_LIBRARY_PATH command in Section 2.
Check available providers.
Inspect missing shared libraries:
ldd "$VIRTUAL_ENV/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so" | grep "not found"

If this command reports missing libraries, resolve those dependencies before retrying inference.

Note: Re-export the .pth checkpoint whenever the trained PyTorch model changes. Otherwise, reuse the existing .onnx file.



