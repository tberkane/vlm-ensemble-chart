"""Convert raw TinyChart prediction files in outputs/TinyChart/ into proper
outputs/predict/ run directories so they can be evaluated and ensembled
through the standard pipeline.

Each `wb_preds_N.json` -> a directory with predictions.json (key rename
imagename -> image, plus zero token counts) and a config.yaml.
"""

import json
import re
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

SRC_DIR = project_root / "outputs" / "TinyChart"
DST_BASE = project_root / "outputs" / "predict"
INPUT_IMAGES_DIR = "data/World Bank v2/test/png"


def convert_one(src_path: Path, dst_dir: Path) -> None:
    with src_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    converted = []
    for entry in raw:
        img = entry.get("imagename") or entry.get("image")
        ans = entry.get("answer", "")
        if img is None:
            continue
        # TinyChart raw outputs use " | " between cells; normalize to TSV.
        ans = re.sub(r"\s*\|\s*", "\t", ans)
        converted.append(
            {
                "image": img,
                "answer": ans,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    with (dst_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)

    cfg = {
        "input_images_dir": INPUT_IMAGES_DIR,
        "model": "TinyChart",
        "temperature": 0.0,
        "prompt": "CHART_EXTRACTION_PROMPT",
        "run_dir": None,
        "use_wandb": False,
        "wandb_project": "vlm-ensemble-chart-extraction",
    }
    with (dst_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    src_files = sorted(SRC_DIR.glob("wb_preds_*.json"),
                       key=lambda p: int(p.stem.split("_")[-1]))
    if not src_files:
        raise FileNotFoundError(f"No wb_preds_*.json files in {SRC_DIR}")

    print(f"Converting {len(src_files)} TinyChart prediction files...")
    out_dirs = []
    for src in src_files:
        run_idx = int(src.stem.split("_")[-1])
        dst_name = f"2026-05-04_TinyChart_WBv2_run{run_idx:02d}"
        dst_dir = DST_BASE / dst_name
        convert_one(src, dst_dir)
        out_dirs.append(dst_dir)
        print(f"  {src.name} -> {dst_dir.relative_to(project_root)}")

    manifest = project_root / "outputs" / "TinyChart" / "converted_run_dirs.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump([str(d.relative_to(project_root)) for d in out_dirs], f, indent=2)
    print(f"\nWrote manifest: {manifest.relative_to(project_root)}")


if __name__ == "__main__":
    main()
