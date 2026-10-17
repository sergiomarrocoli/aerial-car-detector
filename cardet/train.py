"""Fine-tune a detector on the dataset from `cardet.dataset`.

python3 -m cardet.train data/yolo data/models/yolocar-11n.pt [name=gentle] [epochs=30] [freeze=10] [lr0=0.0003] ...
python3 -m cardet.train resume runs/detect/runs/<name>/weights/last.pt
Extra key=value arguments override the defaults below and go straight to Ultralytics. `resume` carries on an
interrupted run; on Apple GPUs (device=mps) Ultralytics sometimes crashes with a shape-mismatch error in its label
assigner, so long runs are wrapped in a retry loop that resumes.
"""
import sys

from ultralytics import YOLO

DEFAULTS = dict(
    epochs=60, imgsz=1536, batch=2, device="mps", project="runs", name="newtown", exist_ok=True,
    optimizer="AdamW", lr0=0.002, flipud=0.5, fliplr=0.5, degrees=0, patience=0, workers=2, plots=False, verbose=False,
)


def parse(extra):
    out = {}
    for item in extra:
        key, value = item.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        out[key] = value
    return out


if __name__ == "__main__":
    if sys.argv[1] == "resume":
        YOLO(sys.argv[2]).train(resume=True)
        sys.exit()
    data, base = sys.argv[1], sys.argv[2]
    YOLO(base).train(data=f"{data}/data.yaml", **{**DEFAULTS, **parse(sys.argv[3:])})
