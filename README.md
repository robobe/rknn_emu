# MobileNetV2 UAV123 classifier

This project fine-tunes MobileNetV2 on UAV123 target crops and runs it with the RKNN Toolkit2 PC simulator. It classifies the annotated target as one of ten dataset categories: bike, bird, boat, building, car, group, person, truck, UAV, or wakeboard.

## Train and play a sequence

Requires Linux x86_64, Python 3.10–3.12, `uv`, and UAV123_10fps at `~/datasets/Dataset_UAV123_10fps`.

Install the RKNN environment, then create a separate CUDA environment for training:

```sh
uv sync
uv venv --python 3.12 .venv-gpu
uv pip install --python .venv-gpu/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch torchvision numpy pillow
```

Train with CUDA and run the RKNN simulator in its own environment:

```sh
.venv-gpu/bin/python train_uav.py --epochs 5 --stride 10
uv run rknn-mobilenet --sequence car1
```

Check that training sees the GPU with `.venv-gpu/bin/python -c 'import torch; print(torch.cuda.is_available())'`. If it prints `False`, the machine's NVIDIA driver or CUDA wheel is not usable. The project environment uses CPU PyTorch to match RKNN Toolkit2; the separate training environment avoids that version constraint. Training fine-tunes the last MobileNetV2 blocks and classification head, balances training samples by category, and selects the checkpoint with the best held-out balanced accuracy. Adjust training with `--epochs`, `--stride`, and `--batch-size`.

Playback uses the UAV123 ground-truth box to crop the target for classification and draws that box in green. The model predicts the target category and percentage; it does not predict the box. The percentage is softmax confidence across the ten UAV123 categories, not a guarantee of correctness. Use any annotated sequence folder, such as `person8`, `car1`, `uav1`, or `bike1`; `--fps` controls playback speed.

On the current training run, held-out balanced accuracy was 63.8% (42.0% majority-class baseline), and the RKNN simulator classified `person8/000001.jpg` as `person` at 97.6%. These are dataset-specific results, not a general object-detection benchmark.

The first run exports and converts the fine-tuned model to ONNX and RKNN in `models/`. Toolkit2's simulator runs on the PC and does not measure RK3588 NPU performance.
