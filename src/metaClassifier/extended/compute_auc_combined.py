#!/usr/bin/env python3
"""
Compute auc_combined from OOF predictions and update summary JSON.

Usage examples:

1) Single build dir (auto-detect OOF/summary paths):
   python -m metaClassifier.extended.compute_auc_combined \
     --build_dir /abs/path/to/.../builds/Disease_ASD/catboost

2) Explicit OOF path (and optional summary to update):
   python -m metaClassifier.extended.compute_auc_combined \
     --oof_file /abs/path/to/oof_predictions_catboost.csv \
     --summary_file /abs/path/to/summary_catboost.json

3) Batch over a builds root (all Disease_*/*/catboost by default):
   python -m metaClassifier.extended.compute_auc_combined \
     --builds_root /abs/path/to/.../builds \
     --model_name catboost
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Optional, Tuple, List

import pandas as pd
from sklearn.metrics import roc_auc_score


def _infer_paths_from_build_dir(build_dir: Path, model_name: str) -> Tuple[Path, Path]:
    out_dir = build_dir / "2_final_model" / "extended_final_cv"
    oof_file = out_dir / f"oof_predictions_{model_name}.csv"
    summary_file = out_dir / f"summary_{model_name}.json"
    return oof_file, summary_file


def _compute_auc_from_oof(oof_path: Path) -> float:
    df = pd.read_csv(oof_path)
    y_true = df["true_value"].values
    y_prob = df["prob_1"].values
    return float(roc_auc_score(y_true, y_prob))


def _update_summary(summary_path: Path, auc_combined: float) -> None:
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    else:
        summary = {}
    summary["auc_combined"] = float(auc_combined)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _iter_build_dirs(builds_root: Path, model_name: str) -> List[Path]:
    # Expect pattern .../builds/Disease_*/{model_name}
    return sorted(builds_root.glob(f"Disease_*/{model_name}"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute auc_combined from OOF predictions and update summary JSON")
    p.add_argument("--build_dir", type=str, default=None, help="Single build directory (e.g., .../builds/Disease_ASD/catboost)")
    p.add_argument("--builds_root", type=str, default=None, help="Root of builds to batch process (e.g., .../builds)")
    p.add_argument("--model_name", type=str, default="catboost", help="Model name used in filenames, default catboost")
    p.add_argument("--oof_file", type=str, default=None, help="Explicit OOF CSV path (overrides build_dir inference)")
    p.add_argument("--summary_file", type=str, default=None, help="Explicit summary JSON path to update (optional)")
    return p.parse_args()


def handle_single(build_dir: Optional[str], model_name: str, oof_file: Optional[str], summary_file: Optional[str]) -> None:
    if oof_file is None:
        if not build_dir:
            raise ValueError("Either --build_dir or --oof_file must be provided for single processing")
        bd = Path(build_dir).resolve()
        oof_path, summary_path = _infer_paths_from_build_dir(bd, model_name)
    else:
        oof_path = Path(oof_file).resolve()
        summary_path = Path(summary_file).resolve() if summary_file else None

    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")

    auc_combined = _compute_auc_from_oof(oof_path)
    print(json.dumps({
        "oof_file": str(oof_path),
        "auc_combined": auc_combined
    }, indent=2, ensure_ascii=False))

    if summary_path is None and build_dir:
        # If build_dir provided and summary not explicit, infer it
        _, summary_path = _infer_paths_from_build_dir(Path(build_dir), model_name)

    if summary_path is not None:
        _update_summary(summary_path, auc_combined)
        print(f"Updated summary: {summary_path}")


def handle_batch(builds_root: str, model_name: str) -> None:
    root = Path(builds_root).resolve()
    targets = _iter_build_dirs(root, model_name)
    if not targets:
        print(f"No build directories found under: {root}")
        return
    results = []
    for bd in targets:
        oof_path, summary_path = _infer_paths_from_build_dir(bd, model_name)
        if not oof_path.exists():
            print(f"Skip (no OOF): {oof_path}")
            continue
        auc_c = _compute_auc_from_oof(oof_path)
        _update_summary(summary_path, auc_c)
        results.append({
            "build_dir": str(bd),
            "oof_file": str(oof_path),
            "summary_file": str(summary_path),
            "auc_combined": auc_c,
        })
        print(f"OK: {bd.name} | auc_combined={auc_c:.6f}")
    print(json.dumps({"processed": results}, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    # Batch mode
    if args.builds_root:
        handle_batch(args.builds_root, args.model_name)
        return
    # Single mode
    handle_single(args.build_dir, args.model_name, args.oof_file, args.summary_file)


if __name__ == "__main__":
    main()


