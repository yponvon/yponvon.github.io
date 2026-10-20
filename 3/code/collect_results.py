"""Copy a notebook's results into images/results/ with the names index.html uses.

usage: python collect_results.py <results folder or .zip from the Part A or Part B notebook> [out folder]
"""
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TIMES = ["0_02", "0_04", "0_06", "0_10", "0_14", "0_25", "0_50"]
EDIT_NAMES = ["campanile", "my_image_1", "my_image_2", "web_image", "drawing_1", "drawing_2"]
TEXT_EDIT_NAMES = ["campanile", "my_image_1", "my_image_2"]


def tag(t):
    # 0.25 -> "t025", the suffix the webpage uses for each noise level
    return "t" + f"{float(t):.2f}".replace(".", "")


def main(src, out):
    out.mkdir(parents=True, exist_ok=True)
    copies = {}  # output name -> source file, relative to the results folder

    copies["p11_original"] = "training_inputs/original.png"
    for t, label in [("0.25", "t_0_25"), ("0.5", "t_0_5"), ("0.75", "t_0_75")]:
        copies[f"p11_noisy_{tag(t)}"] = f"training_inputs/{label}.png"
        copies[f"p13_onestep_{tag(t)}"] = f"one_step_{t}/one_step.png"
        copies[f"p14_velocity_{tag(t)}"] = f"velocity_{float(t):.2f}.png"

    # 1.2 picks its own best sigma, so read it from the file name
    sigmas = {}
    for t in ["0.25", "0.5", "0.75"]:
        blurred = next((src / f"blur_{t}").glob("blur_sigma_*.png"))
        whole, frac = re.search(r"sigma_(\d+)_(\d+)", blurred.name).groups()  # blur_sigma_8_0 -> 8.0
        sigmas[t] = float(f"{whole}.{frac}")
        copies[f"p12_blur_{tag(t)}"] = str(blurred.relative_to(src))

    for frame in sorted((src / "trajectory").glob("t_*.png")):
        copies["p15_" + tag(frame.stem[2:].replace("_", "."))] = f"trajectory/{frame.name}"
    for method in ["blur", "one_step", "euler"]:
        copies[f"p15_cmp_{method}"] = f"comparison/{method}.png"

    for i in range(5):
        copies[f"p16_sample{i}"] = f"conditional_{i}.png"
        copies[f"p21_sample{i}"] = f"cfg_{i}.png"

    for name in EDIT_NAMES:
        copies[f"p31_{name}_original"] = f"edit_{name}/original.png"
        for t in TIMES:
            copies[f"p31_{name}_t{t.replace('_', '')}"] = f"edit_{name}/t_{t}.png"
    for name in TEXT_EDIT_NAMES:
        for t in TIMES:
            copies[f"p32_{name}_t{t.replace('_', '')}"] = f"text_edit_{name}/t_{t}.png"

    for i in range(2):
        copies[f"p33_anagram{i}"] = f"anagram_{i}.png"
        copies[f"p34_hybrid{i}"] = f"hybrid_{i}.png"

    missing = [rel for rel in copies.values() if not (src / rel).is_file()]
    if missing:
        sys.exit("Missing from the results folder:\n  " + "\n  ".join(missing))

    for name, rel in copies.items():
        image = Image.open(src / rel).convert("RGB")
        image.save(out / f"{name}.jpg", quality=88, optimize=True)
        if name.startswith("p33_"):
            # the anagram's second reading is the same image turned upside down
            image.rotate(180).save(out / f"{name}_flipped.jpg", quality=88, optimize=True)
        if name.startswith("p34_"):
            # the notebook views the hybrid at 64x64 to see the distant prompt
            image.resize((64, 64), Image.Resampling.LANCZOS).save(out / f"{name}_small.png")

    metrics = json.loads((src / "metrics.json").read_text())
    summary = {"blur_sigma": sigmas, "reconstruction_mse": metrics.get("reconstruction_mse"),
               "gpu": metrics.get("gpu"), "peak_memory_gib": metrics.get("peak_memory_gib")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(copies)} images to {out}")
    print(json.dumps(summary, indent=2))


def main_partb(src, out):
    # Part B's notebook already names its figures b12_..., b26_... and they are small,
    # sharp-edged MNIST grids, so they are copied as PNG/GIF without re-encoding.
    out.mkdir(parents=True, exist_ok=True)
    figures = sorted(p for p in src.iterdir() if p.name.startswith("b") and p.suffix in (".png", ".gif"))
    for p in figures:
        shutil.copy(p, out / p.name)
    metrics = json.loads((src / "metrics.json").read_text())
    (out / "partb_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"copied {len(figures)} figures to {out}")
    print(json.dumps({k: v for k, v in metrics.items() if "loss" in k or isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    src = Path(sys.argv[1]).expanduser()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "images" / "results"
    if src.suffix == ".zip":
        unzipped = Path(tempfile.mkdtemp())
        zipfile.ZipFile(src).extractall(unzipped)
        src = unzipped
    if (src / "b12_noise_levels.png").is_file():
        main_partb(src, out)
    else:
        main(src, out)
