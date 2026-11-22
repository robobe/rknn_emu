from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uav_data import CLASSES, FRAMES, box_for_frame, crop_target, sequences
from uav_model import create_model


ROOT = Path(__file__).resolve().parent
DATASET = Path.home() / "datasets/Dataset_UAV123_10fps/UAV123_10fps/data_seq/UAV123_10fps"
MODEL_DIR = ROOT / "models"
WEIGHTS_PATH = MODEL_DIR / "uav123_mobilenetv2.pth"
ONNX_PATH = MODEL_DIR / "uav123_mobilenetv2.onnx"
RKNN_PATH = MODEL_DIR / "uav123_mobilenetv2_rk3588.rknn"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="person8", help="UAV123 sequence to run (default: person8)")
    parser.add_argument("--fps", type=float, default=10, help="Playback speed (default: 10 FPS)")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    all_sequences = sequences()
    exact = [record for record in all_sequences if record.name == args.sequence]
    selected = exact or [record for record in all_sequences if record.folder == args.sequence]
    if not selected:
        raise SystemExit(f"Sequence not found in UAV123 annotations: {args.sequence}")
    frames = sorted((FRAMES / selected[0].folder).glob("*.jpg"))
    frames = [path for path in frames if any(record.start <= int(path.stem) <= record.end for record in selected)]
    if not frames:
        raise SystemExit(f"No annotated frames found for {args.sequence}")
    if not WEIGHTS_PATH.is_file():
        raise SystemExit("Trained model is missing; first run: uv run python train_uav.py")

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    if checkpoint["classes"] != CLASSES:
        raise SystemExit("Model classes differ from this project; retrain with uv run python train_uav.py")
    model = create_model().eval()
    model.load_state_dict(checkpoint["state_dict"])
    if not ONNX_PATH.exists():
        torch.onnx.export(model, torch.zeros(1, 3, 224, 224), ONNX_PATH, input_names=["input"], output_names=["output"], opset_version=12)
    import matplotlib.pyplot as plt

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        assert rknn.config(
            target_platform="rk3588",
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
        ) == 0
        assert rknn.load_onnx(model=str(ONNX_PATH)) == 0
        assert rknn.build(do_quantization=False) == 0
        assert rknn.export_rknn(str(RKNN_PATH)) == 0
        assert rknn.init_runtime() == 0  # PC simulator, no NPU attached
        plt.ion()
        figure, axes = plt.subplots()
        axes.axis("off")
        display = None
        from matplotlib.patches import Rectangle

        target_box = Rectangle((0, 0), 1, 1, fill=False, edgecolor="lime", linewidth=2)
        for frame_number, image_path in enumerate(frames, start=1):
            frame = Image.open(image_path).convert("RGB")
            annotation = box_for_frame(selected, int(image_path.stem))
            input_image = crop_target(frame, annotation[1]) if annotation and annotation[1] else None
            if input_image:
                input_tensor = np.asarray(input_image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
                input_tensor = np.ascontiguousarray((input_tensor - MEAN) / STD)
                outputs = rknn.inference(inputs=[input_tensor], data_format=["nchw"])
                scores = np.asarray(outputs[0]).reshape(-1)
                probabilities = np.exp(scores - np.max(scores))
                prediction = int(np.argmax(scores))
                confidence = probabilities[prediction] / probabilities.sum() * 100
                title = f"{CLASSES[prediction]} — {confidence:.1f}%"
            else:
                title = "Target not visible"
            pixels = np.asarray(frame)
            if display is None:
                display = axes.imshow(pixels)
                axes.add_patch(target_box)
            else:
                display.set_data(pixels)
            if annotation and annotation[1]:
                x, y, width, height = annotation[1]
                target_box.set_xy((x, y))
                target_box.set_width(width)
                target_box.set_height(height)
                target_box.set_visible(True)
            else:
                target_box.set_visible(False)
            axes.set_title(f"{title}  |  {frame_number}/{len(frames)}")
            figure.canvas.draw_idle()
            plt.pause(1 / args.fps)
        plt.ioff()
        plt.show()
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
