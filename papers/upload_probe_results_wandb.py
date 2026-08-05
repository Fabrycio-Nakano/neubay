#!/usr/bin/env python3
"""Publish existing probe CSV/JSON outputs to Weights & Biases.

This does not rerun probes. It creates one W&B run per world-model seed and
uploads the existing result tables plus a small set of aggregate metrics.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASET = "Go2JoystickFlatTerrain-direction-expert-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="flnalmada-ufg")
    parser.add_argument("--project", default="world-model-probes")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seeds", default="0,2")
    parser.add_argument("--run-tag", default="scaled_mlp_v2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "probe_outputs",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def paths_for_seed(output_dir, dataset, seed, run_tag):
    stem = "go2__{}__seed{}__member__{}".format(dataset, seed, run_tag)
    return {
        "results": output_dir / (stem + "__probe_results.csv"),
        "diagnostics": output_dir / (stem + "__diagnostic_probe_results.csv"),
        "controls": output_dir / (stem + "__controls.csv"),
        "summary": output_dir / (stem + "__summary.json"),
    }


def finite_mean(frame, column):
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def build_metrics(results, diagnostics, controls, summary):
    valid = results[results["status"] == "ok"]
    skipped = results[results["status"] != "ok"]
    metrics = {
        "probe/valid_rows": int(len(valid)),
        "probe/skipped_rows": int(len(skipped)),
        "probe/ridge_r2_mean": finite_mean(valid, "ridge_r2"),
        "probe/mlp_r2_mean": finite_mean(valid, "mlp_r2"),
        "probe/mlp_minus_ridge_mean": finite_mean(valid, "delta_mlp_minus_ridge"),
        "diagnostic/ridge_r2_mean": finite_mean(diagnostics, "ridge_r2"),
        "diagnostic/mlp_r2_mean": finite_mean(diagnostics, "mlp_r2"),
        "control/ridge_r2_mean": finite_mean(controls, "ridge_r2"),
        "runtime/elapsed_seconds": float(summary["elapsed_seconds"]),
        "data/transitions_used": int(summary["num_transitions_used"]),
        "data/transitions_total": int(summary["num_transitions_total"]),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def table_from_dataframe(wandb, frame):
    clean = frame.where(pd.notnull(frame), None)
    return wandb.Table(columns=list(clean.columns), data=clean.values.tolist())


def publish_seed(args, seed):
    paths = paths_for_seed(args.output_dir, args.dataset, seed, args.run_tag)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing probe artifacts: {}".format(", ".join(missing)))

    results = pd.read_csv(paths["results"])
    diagnostics = pd.read_csv(paths["diagnostics"])
    controls = pd.read_csv(paths["controls"])
    summary = json.loads(paths["summary"].read_text())
    metrics = build_metrics(results, diagnostics, controls, summary)

    print(
        "seed={} results={} diagnostics={} controls={} elapsed={:.2f}s".format(
            seed,
            len(results),
            len(diagnostics),
            len(controls),
            summary["elapsed_seconds"],
        )
    )
    if args.dry_run:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    import wandb

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.dataset,
        job_type="retrospective-probe-upload",
        name="probe-{}-model-seed-{}".format(args.run_tag, seed),
        tags=["probe", "retrospective-upload", "ridge", "mlp", "joystick"],
        config={
            "domain": "go2",
            "dataset": args.dataset,
            "model_seed": seed,
            "run_tag": args.run_tag,
            "aggregate": summary["args"]["aggregate"],
            "member_index": summary["args"]["member_index"],
            "sample_size": summary["num_transitions_used"],
            "probe_seeds": summary["probe_seeds"],
            "checkpoint": summary["checkpoint"],
            "retroactive_upload": True,
        },
        reinit=True,
    )
    try:
        wandb.log(
            {
                **metrics,
                "tables/probe_results": table_from_dataframe(wandb, results),
                "tables/diagnostic_results": table_from_dataframe(wandb, diagnostics),
                "tables/controls": table_from_dataframe(wandb, controls),
            }
        )

        artifact = wandb.Artifact(
            name="joystick-probe-results-seed-{}-{}".format(seed, args.run_tag),
            type="probe-results",
            metadata={"dataset": args.dataset, "model_seed": seed},
        )
        for path in paths.values():
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        run.summary.update(metrics)
        print("published seed={} run={}".format(seed, run.url))
    finally:
        run.finish()


def main():
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    for seed in seeds:
        publish_seed(args, seed)


if __name__ == "__main__":
    main()
