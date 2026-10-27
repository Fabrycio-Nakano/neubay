#!/usr/bin/env python3
"""Rerun the Go2 rollouts that previously survived at least 100 steps.

The first execution freezes the selected jobs in an incremental manifest.  Any
later execution resumes that snapshot, so edits to the source manifests cannot
silently change a run that is already in progress.

The runner intentionally executes one video at a time.  Each rollout runs in a
fresh subprocess, is written to a staging MP4, is checked with ffprobe, and is
only then atomically promoted to its final path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = ROOT / ".venv-go2-video/bin/python"
DEFAULT_GENERATOR = ROOT / "scripts/local/generate_go2_video.py"
DEFAULT_EXPERIMENT_MANIFEST = (
    ROOT / "videos/experiment_variants/go2-full8-20260824-r1/manifest.csv"
)
DEFAULT_VARIANTS_MANIFEST = ROOT / "videos/variants/manifest.csv"
DEFAULT_LEGACY_CHECKPOINT_ROOT = (
    ROOT / "offline_agent/ckpt/go2/Go2JoystickFlatTerrain-direction-expert-v1"
)
DEFAULT_OUTPUT_ROOT = ROOT / "videos/go2_survivors_1000"

SOURCE_MIN_STEPS = 100
TARGET_STEPS = 1000
EXPECTED_TOTAL = 154
EXPECTED_SOURCE_COUNTS = {
    "go2-full8-20260824-r1": 149,
    "legacy-direction-expert-v1": 5,
}
SCHEMA_VERSION = 1

MANIFEST_FIELDS = [
    "schema_version",
    "job_id",
    "source_group",
    "source_manifest",
    "dataset",
    "condition",
    "policy_seed",
    "simulation_seed",
    "vx",
    "vy",
    "yaw_rate",
    "source_steps",
    "selection_min_steps",
    "target_steps",
    "render_every",
    "width",
    "height",
    "camera",
    "steps",
    "reached_target",
    "reproduced_source_steps",
    "termination",
    "duration_seconds",
    "frame_count",
    "status",
    "attempts",
    "return_code",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "generator",
    "generator_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "spec_sha256",
    "source_video",
    "video",
    "receipt",
    "job_log",
    "video_bytes",
    "video_sha256",
    "error",
]

_STOP_REQUESTED = False
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None
_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


class RunnerError(RuntimeError):
    """An actionable runner or input error."""


@dataclass(frozen=True)
class Job:
    job_id: str
    source_group: str
    source_manifest: Path
    dataset: str
    condition: str
    policy_seed: int
    simulation_seed: int
    vx: float
    vy: float
    yaw_rate: float
    source_steps: int
    selection_min_steps: int
    target_steps: int
    render_every: int
    width: int
    height: int
    camera: str
    generator: Path
    generator_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    spec_sha256: str
    source_video: Path
    video: Path
    receipt: Path
    job_log: Path


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    codec: str
    video_bytes: int
    sha256: str


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def resolve_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def resolve_executable(path: Path | str) -> Path:
    """Make an executable path absolute without dereferencing venv symlinks."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return Path(os.path.abspath(value))


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def safe_component(value: str, field: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise RunnerError(f"Unsafe {field}: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RunnerError(f"Unsupported characters in {field}: {value!r}")
    return value


def required(row: dict[str, str], field: str, source: Path) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise RunnerError(f"Missing {field!r} in {source}")
    return value


def parse_int(row: dict[str, str], field: str, source: Path) -> int:
    value = required(row, field, source)
    try:
        return int(value)
    except ValueError as error:
        raise RunnerError(f"Invalid integer {field}={value!r} in {source}") from error


def parse_float(row: dict[str, str], field: str, source: Path) -> float:
    value = required(row, field, source)
    try:
        return float(value)
    except ValueError as error:
        raise RunnerError(f"Invalid float {field}={value!r} in {source}") from error


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RunnerError(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise RunnerError(f"Manifest has no header: {path}")
        return [dict(row) for row in reader]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_file_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_SHA256_CACHE:
        _FILE_SHA256_CACHE[key] = file_sha256(path)
    return _FILE_SHA256_CACHE[key]


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=MANIFEST_FIELDS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def calculate_spec_sha256(
    *,
    source_group: str,
    dataset: str,
    condition: str,
    policy_seed: int,
    simulation_seed: int,
    vx: float,
    vy: float,
    yaw_rate: float,
    source_steps: int,
    selection_min_steps: int,
    target_steps: int,
    render_every: int,
    width: int,
    height: int,
    camera: str,
    generator: Path,
    generator_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "source_group": source_group,
        "dataset": dataset,
        "condition": condition,
        "policy_seed": policy_seed,
        "simulation_seed": simulation_seed,
        "vx": vx,
        "vy": vy,
        "yaw_rate": yaw_rate,
        "source_steps": source_steps,
        "selection_min_steps": selection_min_steps,
        "target_steps": target_steps,
        "render_every": render_every,
        "width": width,
        "height": height,
        "camera": camera,
        "generator": display_path(generator),
        "generator_sha256": generator_sha256,
        "checkpoint": display_path(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def make_job(
    *,
    row: dict[str, str],
    source_group: str,
    source_manifest: Path,
    dataset: str,
    generator: Path,
    checkpoint: Path,
    output_root: Path,
    selection_min_steps: int,
    target_steps: int,
    render_every: int,
    width: int,
    height: int,
    camera: str,
) -> Job:
    dataset = safe_component(dataset, "dataset")
    condition = safe_component(required(row, "condition", source_manifest), "condition")
    policy_seed = parse_int(row, "policy_seed", source_manifest)
    source_steps = parse_int(row, "steps", source_manifest)
    simulation_seed = parse_int(row, "simulation_seed", source_manifest)
    vx = parse_float(row, "vx", source_manifest)
    vy = parse_float(row, "vy", source_manifest)
    yaw_rate = parse_float(row, "yaw_rate", source_manifest)
    generator = generator.resolve()
    checkpoint = checkpoint.resolve()
    try:
        generator_sha256 = cached_file_sha256(generator)
        checkpoint_sha256 = cached_file_sha256(checkpoint)
    except OSError as error:
        raise RunnerError(f"Cannot fingerprint inputs for {dataset}/{condition}: {error}") from error
    spec_sha256 = calculate_spec_sha256(
        source_group=source_group,
        dataset=dataset,
        condition=condition,
        policy_seed=policy_seed,
        simulation_seed=simulation_seed,
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        source_steps=source_steps,
        selection_min_steps=selection_min_steps,
        target_steps=target_steps,
        render_every=render_every,
        width=width,
        height=height,
        camera=camera,
        generator=generator,
        generator_sha256=generator_sha256,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    source_video = resolve_path(required(row, "video", source_manifest))
    filename = f"seed{policy_seed}_{condition}.mp4"
    video = output_root / source_group / dataset / filename
    receipt = video.with_suffix(video.suffix + ".done.json")
    job_log = output_root / "job_logs" / source_group / dataset / f"{filename}.log"
    job_id = (
        f"{source_group}/{dataset}/seed{policy_seed}_{condition}/steps{target_steps}/"
        f"{spec_sha256[:16]}"
    )
    return Job(
        job_id=job_id,
        source_group=source_group,
        source_manifest=source_manifest,
        dataset=dataset,
        condition=condition,
        policy_seed=policy_seed,
        simulation_seed=simulation_seed,
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        source_steps=source_steps,
        selection_min_steps=selection_min_steps,
        target_steps=target_steps,
        render_every=render_every,
        width=width,
        height=height,
        camera=camera,
        generator=generator,
        generator_sha256=generator_sha256,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        spec_sha256=spec_sha256,
        source_video=source_video,
        video=video.resolve(),
        receipt=receipt.resolve(),
        job_log=job_log.resolve(),
    )


def select_jobs(args: argparse.Namespace) -> list[Job]:
    output_root = resolve_path(args.output_root)
    experiment_manifest = resolve_path(args.experiment_manifest)
    variants_manifest = resolve_path(args.variants_manifest)
    legacy_checkpoint_root = resolve_path(args.legacy_checkpoint_root)
    generator = resolve_path(args.generator)
    jobs: list[Job] = []

    for row in read_csv(experiment_manifest):
        if (row.get("status") or "").strip() != "ok":
            continue
        try:
            source_steps = int((row.get("steps") or "").strip())
        except ValueError:
            continue
        if source_steps < args.source_min_steps:
            continue
        checkpoint = resolve_path(required(row, "checkpoint", experiment_manifest))
        jobs.append(
            make_job(
                row=row,
                source_group="go2-full8-20260824-r1",
                source_manifest=experiment_manifest,
                dataset=required(row, "dataset", experiment_manifest),
                generator=generator,
                checkpoint=checkpoint,
                output_root=output_root,
                selection_min_steps=args.source_min_steps,
                target_steps=args.target_steps,
                render_every=args.render_every,
                width=args.width,
                height=args.height,
                camera=args.camera,
            )
        )

    legacy_dataset = "Go2JoystickFlatTerrain-direction-expert-v1"
    for row in read_csv(variants_manifest):
        if (row.get("status") or "").strip() != "ok":
            continue
        try:
            source_steps = int((row.get("steps") or "").strip())
        except ValueError:
            continue
        if source_steps < args.source_min_steps:
            continue
        policy_seed = parse_int(row, "policy_seed", variants_manifest)
        checkpoint = legacy_checkpoint_root / f"agent_seed{policy_seed}.eqx"
        jobs.append(
            make_job(
                row=row,
                source_group="legacy-direction-expert-v1",
                source_manifest=variants_manifest,
                dataset=legacy_dataset,
                generator=generator,
                checkpoint=checkpoint,
                output_root=output_root,
                selection_min_steps=args.source_min_steps,
                target_steps=args.target_steps,
                render_every=args.render_every,
                width=args.width,
                height=args.height,
                camera=args.camera,
            )
        )

    jobs.sort(
        key=lambda job: (job.source_group, job.dataset, job.policy_seed, job.condition)
    )
    return jobs


def record_for_job(job: Job) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "source_group": job.source_group,
        "source_manifest": display_path(job.source_manifest),
        "dataset": job.dataset,
        "condition": job.condition,
        "policy_seed": job.policy_seed,
        "simulation_seed": job.simulation_seed,
        "vx": job.vx,
        "vy": job.vy,
        "yaw_rate": job.yaw_rate,
        "source_steps": job.source_steps,
        "selection_min_steps": job.selection_min_steps,
        "target_steps": job.target_steps,
        "render_every": job.render_every,
        "width": job.width,
        "height": job.height,
        "camera": job.camera,
        "steps": "",
        "reached_target": "",
        "reproduced_source_steps": "",
        "termination": "",
        "duration_seconds": "",
        "frame_count": "",
        "status": "pending",
        "attempts": 0,
        "return_code": "",
        "started_at": "",
        "finished_at": "",
        "elapsed_seconds": "",
        "generator": display_path(job.generator),
        "generator_sha256": job.generator_sha256,
        "checkpoint": display_path(job.checkpoint),
        "checkpoint_sha256": job.checkpoint_sha256,
        "spec_sha256": job.spec_sha256,
        "source_video": display_path(job.source_video),
        "video": display_path(job.video),
        "receipt": display_path(job.receipt),
        "job_log": display_path(job.job_log),
        "video_bytes": "",
        "video_sha256": "",
        "error": "",
    }


def job_from_record(row: dict[str, str], output_root: Path) -> Job:
    source = output_root / "manifest.csv"
    schema = parse_int(row, "schema_version", source)
    if schema != SCHEMA_VERSION:
        raise RunnerError(f"Unsupported schema_version={schema} in {source}")
    video = resolve_path(required(row, "video", source))
    receipt = resolve_path(required(row, "receipt", source))
    job_log = resolve_path(required(row, "job_log", source))
    for candidate, label in (
        (video, "video"),
        (receipt, "receipt"),
        (job_log, "job_log"),
    ):
        if not candidate.is_relative_to(output_root):
            raise RunnerError(f"Snapshot {label} escapes output root: {candidate}")
    generator = resolve_path(required(row, "generator", source))
    checkpoint = resolve_path(required(row, "checkpoint", source))
    job = Job(
        job_id=required(row, "job_id", source),
        source_group=safe_component(
            required(row, "source_group", source), "source_group"
        ),
        source_manifest=resolve_path(required(row, "source_manifest", source)),
        dataset=safe_component(required(row, "dataset", source), "dataset"),
        condition=safe_component(required(row, "condition", source), "condition"),
        policy_seed=parse_int(row, "policy_seed", source),
        simulation_seed=parse_int(row, "simulation_seed", source),
        vx=parse_float(row, "vx", source),
        vy=parse_float(row, "vy", source),
        yaw_rate=parse_float(row, "yaw_rate", source),
        source_steps=parse_int(row, "source_steps", source),
        selection_min_steps=parse_int(row, "selection_min_steps", source),
        target_steps=parse_int(row, "target_steps", source),
        render_every=parse_int(row, "render_every", source),
        width=parse_int(row, "width", source),
        height=parse_int(row, "height", source),
        camera=required(row, "camera", source),
        generator=generator,
        generator_sha256=required(row, "generator_sha256", source),
        checkpoint=checkpoint,
        checkpoint_sha256=required(row, "checkpoint_sha256", source),
        spec_sha256=required(row, "spec_sha256", source),
        source_video=resolve_path(required(row, "source_video", source)),
        video=video,
        receipt=receipt,
        job_log=job_log,
    )
    calculated = calculate_spec_sha256(
        source_group=job.source_group,
        dataset=job.dataset,
        condition=job.condition,
        policy_seed=job.policy_seed,
        simulation_seed=job.simulation_seed,
        vx=job.vx,
        vy=job.vy,
        yaw_rate=job.yaw_rate,
        source_steps=job.source_steps,
        selection_min_steps=job.selection_min_steps,
        target_steps=job.target_steps,
        render_every=job.render_every,
        width=job.width,
        height=job.height,
        camera=job.camera,
        generator=job.generator,
        generator_sha256=job.generator_sha256,
        checkpoint=job.checkpoint,
        checkpoint_sha256=job.checkpoint_sha256,
    )
    if calculated != job.spec_sha256 or not job.job_id.endswith(job.spec_sha256[:16]):
        raise RunnerError(f"Scientific input fingerprint mismatch for {job.job_id}")
    return job


def validate_jobs(jobs: list[Job], args: argparse.Namespace, *, initial: bool) -> None:
    if not jobs:
        raise RunnerError("The selection is empty")
    if not args.allow_selection_count_mismatch and len(jobs) != args.expected_count:
        raise RunnerError(
            f"Expected {args.expected_count} selected jobs, found {len(jobs)}. "
            "Use --allow-selection-count-mismatch only after reviewing the source manifests."
        )
    if initial and not args.allow_selection_count_mismatch:
        counts = Counter(job.source_group for job in jobs)
        if counts != Counter(EXPECTED_SOURCE_COUNTS):
            raise RunnerError(
                f"Unexpected source split: {dict(counts)}; expected {EXPECTED_SOURCE_COUNTS}"
            )
    ids = [job.job_id for job in jobs]
    outputs = [job.video for job in jobs]
    if len(set(ids)) != len(ids):
        raise RunnerError("Duplicate job_id in the selection")
    if len(set(outputs)) != len(outputs):
        raise RunnerError("Two selected jobs point to the same output video")
    for job in jobs:
        if job.source_steps < job.selection_min_steps:
            raise RunnerError(f"Snapshot contains an ineligible job: {job.job_id}")
        if job.target_steps <= 0 or job.render_every <= 0:
            raise RunnerError(f"Invalid rollout settings for {job.job_id}")
        if not job.generator.is_file():
            raise RunnerError(f"Generator not found for {job.job_id}: {job.generator}")
        if not job.checkpoint.is_file():
            raise RunnerError(
                f"Checkpoint not found for {job.job_id}: {job.checkpoint}"
            )
        if cached_file_sha256(job.generator) != job.generator_sha256:
            raise RunnerError(
                f"Generator changed after the selection was frozen: {job.generator}"
            )
        if cached_file_sha256(job.checkpoint) != job.checkpoint_sha256:
            raise RunnerError(
                f"Checkpoint changed after the selection was frozen: {job.checkpoint}"
            )
        if initial and not job.source_video.is_file():
            raise RunnerError(
                f"Source video not found for {job.job_id}: {job.source_video}"
            )


def validate_runtime(args: argparse.Namespace, output_root: Path) -> None:
    python = resolve_executable(args.python)
    generator = resolve_path(args.generator)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RunnerError(f"Python executable is unavailable: {python}")
    if not generator.is_file():
        raise RunnerError(f"Video generator is unavailable: {generator}")
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RunnerError(f"Required executable is unavailable: {executable}")
    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cpu",
        MUJOCO_GL="egl",
        MESA_SHADER_CACHE_DIR="/tmp/neubay_mesa_cache",
    )
    import_check = subprocess.run(
        [str(python), str(generator), "--help"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if import_check.returncode != 0:
        detail = concise_error(
            import_check.stdout + import_check.stderr,
            f"generator preflight exited with {import_check.returncode}",
        )
        raise RunnerError(f"The Go2 video environment failed its import check: {detail}")
    forbidden = {
        ROOT.resolve(),
        resolve_path(args.experiment_manifest).parent,
        resolve_path(args.variants_manifest).parent,
    }
    if output_root in forbidden:
        raise RunnerError(f"Refusing unsafe output root: {output_root}")
    existing_parent = output_root
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    free_bytes = shutil.disk_usage(existing_parent).free
    required_bytes = int(args.min_free_gb * 1024**3)
    if free_bytes < required_bytes:
        raise RunnerError(
            f"Only {free_bytes / 1024**3:.1f} GiB free; "
            f"--min-free-gb requires {args.min_free_gb:.1f} GiB"
        )


def load_or_select_jobs(
    args: argparse.Namespace, *, write_snapshot: bool
) -> tuple[list[Job], list[dict[str, Any]], bool]:
    output_root = resolve_path(args.output_root)
    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists():
        rows: list[dict[str, Any]] = read_csv(manifest_path)
        jobs = [job_from_record(row, output_root) for row in rows]
        validate_jobs(jobs, args, initial=False)
        settings = {
            (
                job.selection_min_steps,
                job.target_steps,
                job.render_every,
                job.width,
                job.height,
                job.camera,
            )
            for job in jobs
        }
        requested = {
            (
                args.source_min_steps,
                args.target_steps,
                args.render_every,
                args.width,
                args.height,
                args.camera,
            )
        }
        if settings != requested:
            raise RunnerError(
                "The existing snapshot uses different rollout settings. "
                "Resume with the original arguments or choose another --output-root."
            )
        requested_generator = resolve_path(args.generator)
        if {job.generator for job in jobs} != {requested_generator}:
            raise RunnerError(
                "The existing snapshot uses a different --generator. "
                "Resume with the original path or choose another --output-root."
            )
        return jobs, rows, False

    jobs = select_jobs(args)
    validate_jobs(jobs, args, initial=True)
    rows = [record_for_job(job) for job in jobs]
    if write_snapshot:
        atomic_write_csv(manifest_path, rows)
    return jobs, rows, True


def run_config(
    args: argparse.Namespace, jobs: list[Job], created_at: str
) -> dict[str, Any]:
    source_manifests = sorted({job.source_manifest for job in jobs})
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "selection": {
            "criterion": f"status == ok and steps >= {args.source_min_steps}",
            "count": len(jobs),
            "source_counts": dict(
                sorted(Counter(job.source_group for job in jobs).items())
            ),
            "source_manifest_sha256": {
                display_path(path): file_sha256(path)
                for path in source_manifests
                if path.is_file()
            },
        },
        "rollout": {
            "target_steps": args.target_steps,
            "render_every": args.render_every,
            "width": args.width,
            "height": args.height,
            "camera": args.camera,
            "sequential": True,
        },
        "runtime": {
            "python": display_path(resolve_executable(args.python)),
            "generator": display_path(resolve_path(args.generator)),
            "timeout_seconds": args.timeout_seconds,
            "retries_per_job": args.retries,
        },
        "manifest": display_path(resolve_path(args.output_root) / "manifest.csv"),
        "output_root": display_path(resolve_path(args.output_root)),
    }


def probe_video(path: Path, *, expected_steps: int, job: Job) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,width,height,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RunnerError(f"ffprobe rejected {path}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        frame_count = int(stream["nb_read_frames"])
        duration = float(payload["format"]["duration"])
        width = int(stream["width"])
        height = int(stream["height"])
        codec = str(stream["codec_name"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerError(f"Incomplete ffprobe metadata for {path}") from error
    expected_frames = expected_steps // job.render_every + 1
    if frame_count != expected_frames:
        raise RunnerError(
            f"Frame mismatch for {path}: got {frame_count}, expected {expected_frames} "
            f"for {expected_steps} steps"
        )
    if width != job.width or height != job.height:
        raise RunnerError(
            f"Resolution mismatch for {path}: got {width}x{height}, "
            f"expected {job.width}x{job.height}"
        )
    if codec != "h264" or duration <= 0:
        raise RunnerError(
            f"Unexpected video metadata for {path}: codec={codec}, duration={duration}"
        )
    return VideoInfo(
        frame_count=frame_count,
        duration_seconds=duration,
        width=width,
        height=height,
        codec=codec,
        video_bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def clear_result(row: dict[str, Any]) -> None:
    row.update(
        steps="",
        reached_target="",
        reproduced_source_steps="",
        termination="",
        duration_seconds="",
        frame_count="",
        video_bytes="",
        video_sha256="",
    )


def validate_reported_steps(job: Job, steps: int) -> None:
    if steps <= 0 or steps > job.target_steps:
        raise RunnerError(
            f"Reported steps {steps} is outside [1, {job.target_steps}]"
        )


def apply_success(
    row: dict[str, Any], *, steps: int, info: VideoInfo, receipt: dict[str, Any]
) -> None:
    row.update(
        steps=steps,
        reached_target="true" if steps >= int(row["target_steps"]) else "false",
        reproduced_source_steps=(
            "true" if steps >= int(row["source_steps"]) else "false"
        ),
        termination=(
            "target_reached"
            if steps >= int(row["target_steps"])
            else "terminated_early"
        ),
        duration_seconds=f"{info.duration_seconds:.6f}",
        frame_count=info.frame_count,
        status="ok",
        return_code=receipt.get("return_code", 0),
        started_at=receipt.get("started_at", row.get("started_at", "")),
        finished_at=receipt.get("finished_at", now_iso()),
        elapsed_seconds=receipt.get("elapsed_seconds", row.get("elapsed_seconds", "")),
        video_bytes=info.video_bytes,
        video_sha256=info.sha256,
        error="",
    )


def reconcile_or_validate_success(job: Job, row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "")
    if status == "ok":
        try:
            steps = int(str(row.get("steps") or ""))
            validate_reported_steps(job, steps)
            info = probe_video(job.video, expected_steps=steps, job=job)
            recorded_hash = str(row.get("video_sha256") or "")
            if recorded_hash and recorded_hash != info.sha256:
                raise RunnerError(f"SHA256 mismatch for {job.video}")
            apply_success(row, steps=steps, info=info, receipt=row)
            return True
        except (OSError, ValueError, RunnerError, subprocess.TimeoutExpired) as error:
            clear_result(row)
            row.update(status="pending", error=f"Existing success is invalid: {error}")
            return False

    if not job.receipt.is_file() or not job.video.is_file():
        if status == "running":
            row.update(status="pending", error="Recovered an interrupted running job")
        return False
    try:
        with job.receipt.open(encoding="utf-8") as file:
            receipt = json.load(file)
        if receipt.get("job_id") != job.job_id:
            raise RunnerError("receipt job_id does not match")
        steps = int(receipt["steps"])
        validate_reported_steps(job, steps)
        info = probe_video(job.video, expected_steps=steps, job=job)
        if receipt.get("video_sha256") != info.sha256:
            raise RunnerError("receipt SHA256 does not match")
        apply_success(row, steps=steps, info=info, receipt=receipt)
        return True
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RunnerError,
        json.JSONDecodeError,
    ) as error:
        clear_result(row)
        row.update(
            status="pending", error=f"Could not reconcile completion receipt: {error}"
        )
        return False


def status_payload(
    rows: list[dict[str, Any]],
    *,
    state: str,
    output_root: Path,
    current_job: str = "",
    started_at: str = "",
) -> dict[str, Any]:
    statuses = Counter(str(row.get("status") or "pending") for row in rows)
    reached = sum(
        1
        for row in rows
        if str(row.get("status")) == "ok"
        and str(row.get("reached_target")).lower() == "true"
    )
    completed = statuses.get("ok", 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "pid": os.getpid(),
        "process_alive": state == "running",
        "started_at": started_at,
        "updated_at": now_iso(),
        "current_job": current_job,
        "total": len(rows),
        "completed": completed,
        "remaining": len(rows) - completed,
        "reached_target": reached,
        "terminated_early": completed - reached,
        "statuses": dict(sorted(statuses.items())),
        "manifest": display_path(output_root / "manifest.csv"),
        "output_root": display_path(output_root),
    }


def write_status(
    rows: list[dict[str, Any]],
    *,
    state: str,
    output_root: Path,
    current_job: str = "",
    started_at: str = "",
) -> None:
    atomic_write_json(
        output_root / "status.json",
        status_payload(
            rows,
            state=state,
            output_root=output_root,
            current_job=current_job,
            started_at=started_at,
        ),
    )


def terminate_process_group(
    process: subprocess.Popen[str], grace_seconds: float = 10.0
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def handle_signal(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    process = _ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    log(f"Received signal {signum}; stopping after persisting the current job")


def child_process_setup(parent_pid: int) -> None:
    """Ask Linux to terminate a generator if the runner disappears abruptly."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        # PR_SET_PDEATHSIG = 1.
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        # This is an extra orphan guard; process-group cleanup remains primary.
        pass


def run_subprocess(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_seconds: int,
    job_log: Path,
    attempt: int,
    started_at: str,
    heartbeat: Callable[[], None] | None = None,
) -> tuple[int, str, bool, bool]:
    global _ACTIVE_PROCESS
    job_log.parent.mkdir(parents=True, exist_ok=True)
    parent_pid = os.getpid()
    with job_log.open("a+", encoding="utf-8") as logfile:
        logfile.write(f"\n=== attempt {attempt} ===\n")
        logfile.write(f"started_at: {started_at}\ncommand: {shlex.join(command)}\n")
        logfile.write("--- output ---\n")
        logfile.flush()
        os.fsync(logfile.fileno())
        output_start = logfile.tell()
        if _STOP_REQUESTED:
            return 130, "Runner stopped before child spawn", False, True
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                preexec_fn=lambda: child_process_setup(parent_pid),
            )
        except OSError as error:
            output = f"Could not start generator: {error}"
            logfile.write(output + "\n")
            logfile.flush()
            os.fsync(logfile.fileno())
            return 127, output, False, False
        _ACTIVE_PROCESS = process
        timed_out = False
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        )
        last_heartbeat = time.monotonic()
        try:
            if _STOP_REQUESTED:
                terminate_process_group(process)
            while process.poll() is None:
                if _STOP_REQUESTED:
                    terminate_process_group(process)
                    break
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    timed_out = True
                    terminate_process_group(process)
                    break
                if heartbeat is not None and now - last_heartbeat >= 30:
                    try:
                        heartbeat()
                    except Exception as error:
                        log(f"Could not update heartbeat: {error}")
                    last_heartbeat = now
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                terminate_process_group(process)
            process.wait()
            interrupted = _STOP_REQUESTED
        finally:
            _ACTIVE_PROCESS = None
        logfile.flush()
        os.fsync(logfile.fileno())
        logfile.seek(output_start)
        output = logfile.read()
        logfile.seek(0, os.SEEK_END)
        if output and not output.endswith("\n"):
            logfile.write("\n")
        logfile.write(
            f"--- attempt result ---\nfinished_at: {now_iso()}\n"
            f"return_code: {process.returncode}\n"
            f"timed_out: {timed_out}\ninterrupted: {interrupted}\n"
        )
        logfile.flush()
        os.fsync(logfile.fileno())
        return process.returncode, output, timed_out, interrupted


def concise_error(output: str, fallback: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    detail = " | ".join(lines[-8:]) if lines else fallback
    return detail[-2000:]


def command_for_job(args: argparse.Namespace, job: Job, partial: Path) -> list[str]:
    return [
        str(resolve_executable(args.python)),
        str(job.generator),
        str(job.checkpoint),
        "--output",
        str(partial),
        "--steps",
        str(job.target_steps),
        "--seed",
        str(job.simulation_seed),
        "--vx",
        str(job.vx),
        "--vy",
        str(job.vy),
        "--yaw-rate",
        str(job.yaw_rate),
        "--render-every",
        str(job.render_every),
        "--width",
        str(job.width),
        "--height",
        str(job.height),
        "--camera",
        job.camera,
    ]


def run_attempt(
    args: argparse.Namespace,
    job: Job,
    row: dict[str, Any],
    *,
    manifest_path: Path,
    all_rows: list[dict[str, Any]],
    events_path: Path,
) -> str:
    output_root = resolve_path(args.output_root)
    partial = job.video.with_name(f".{job.video.stem}.partial.mp4")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    attempt = int(str(row.get("attempts") or 0)) + 1
    started_at = now_iso()
    clear_result(row)
    row.update(
        status="running",
        attempts=attempt,
        started_at=started_at,
        finished_at="",
        elapsed_seconds="",
        return_code="",
        error="",
    )
    atomic_write_csv(manifest_path, all_rows)
    write_status(
        all_rows,
        state="running",
        output_root=output_root,
        current_job=job.job_id,
        started_at=args.runner_started_at,
    )
    append_json_line(
        events_path,
        {
            "event": "attempt_started",
            "at": started_at,
            "attempt": attempt,
            "job_id": job.job_id,
        },
    )

    command = command_for_job(args, job, partial)
    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cpu",
        MUJOCO_GL="egl",
        MESA_SHADER_CACHE_DIR="/tmp/neubay_mesa_cache",
        PYTHONUNBUFFERED="1",
    )
    log(f"Attempt {attempt}: {job.job_id}")
    began = time.monotonic()
    try:
        return_code, output, timed_out, interrupted = run_subprocess(
            command,
            env=environment,
            timeout_seconds=args.timeout_seconds,
            job_log=job.job_log,
            attempt=attempt,
            started_at=started_at,
            heartbeat=lambda: write_status(
                all_rows,
                state="running",
                output_root=output_root,
                current_job=job.job_id,
                started_at=args.runner_started_at,
            ),
        )
    except (OSError, subprocess.SubprocessError) as error:
        return_code = 127
        output = f"Could not execute generator: {error}"
        timed_out = False
        interrupted = _STOP_REQUESTED
    elapsed = time.monotonic() - began
    finished_at = now_iso()

    common = {
        "finished_at": finished_at,
        "elapsed_seconds": f"{elapsed:.3f}",
        "return_code": return_code,
    }
    if interrupted:
        row.update(status="interrupted", error="Runner interrupted by signal", **common)
        outcome = "interrupted"
    elif timed_out:
        row.update(
            status="timeout",
            error=f"Exceeded timeout of {args.timeout_seconds} seconds",
            **common,
        )
        outcome = "timeout"
    elif return_code != 0:
        row.update(
            status="failed",
            error=concise_error(output, f"Generator exited with {return_code}"),
            **common,
        )
        outcome = "failed"
    else:
        step_match = re.search(r"(?m)^Steps:\s*(\d+);", output)
        if not step_match:
            row.update(
                status="failed",
                error="Generator succeeded but did not report a Steps value",
                **common,
            )
            outcome = "failed"
        else:
            steps = int(step_match.group(1))
            try:
                validate_reported_steps(job, steps)
                info = probe_video(partial, expected_steps=steps, job=job)
                job.video.parent.mkdir(parents=True, exist_ok=True)
                os.replace(partial, job.video)
                fsync_directory(job.video.parent)
                receipt = {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": job.job_id,
                    "steps": steps,
                    "reached_target": steps >= job.target_steps,
                    "reproduced_source_steps": steps >= job.source_steps,
                    "return_code": return_code,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "frame_count": info.frame_count,
                    "duration_seconds": info.duration_seconds,
                    "video_bytes": info.video_bytes,
                    "video_sha256": info.sha256,
                    "video": display_path(job.video),
                }
                atomic_write_json(job.receipt, receipt)
                apply_success(row, steps=steps, info=info, receipt=receipt)
                outcome = "ok"
            except (OSError, RunnerError, subprocess.TimeoutExpired) as error:
                row.update(
                    status="failed", error=f"Video validation failed: {error}", **common
                )
                outcome = "failed"

    atomic_write_csv(manifest_path, all_rows)
    append_json_line(
        events_path,
        {
            "event": "attempt_finished",
            "at": finished_at,
            "attempt": attempt,
            "job_id": job.job_id,
            "outcome": outcome,
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 3),
            "error": row.get("error", ""),
        },
    )
    write_status(
        all_rows,
        state="running",
        output_root=output_root,
        current_job="",
        started_at=args.runner_started_at,
    )
    if outcome == "ok":
        log(
            f"Completed {job.job_id}: steps={row['steps']}, "
            f"reached_target={row['reached_target']}, elapsed={elapsed:.1f}s"
        )
    else:
        log(f"{outcome.upper()} {job.job_id}: {row.get('error', '')}")
    return outcome


def acquire_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".runner.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.seek(0)
        owner = lock.read().strip() or "unknown"
        lock.close()
        raise RunnerError(f"Another runner holds {lock_path} (PID {owner})") from error
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    os.fsync(lock.fileno())
    return lock


def write_pid(output_root: Path) -> Path:
    pid_path = output_root / "runner.pid"
    atomic_write_json(
        pid_path,
        {"pid": os.getpid(), "started_at": now_iso(), "command": sys.argv},
    )
    return pid_path


def remove_own_pid_file(pid_path: Path) -> None:
    try:
        with pid_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if int(payload.get("pid", -1)) == os.getpid():
            pid_path.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass


def run(args: argparse.Namespace) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    output_root = resolve_path(args.output_root)
    validate_runtime(args, output_root)
    lock = acquire_lock(output_root)
    pid_path = write_pid(output_root)
    args.runner_started_at = now_iso()
    manifest_path = output_root / "manifest.csv"
    events_path = output_root / "attempts.jsonl"
    try:
        jobs, rows, created = load_or_select_jobs(args, write_snapshot=True)
        if created or not (output_root / "run_config.json").is_file():
            atomic_write_json(
                output_root / "run_config.json",
                run_config(args, jobs, args.runner_started_at),
            )
        rows_by_id = {str(row["job_id"]): row for row in rows}
        changed = False
        for job in jobs:
            row = rows_by_id[job.job_id]
            before = dict(row)
            reconcile_or_validate_success(job, row)
            if row != before:
                changed = True
        if changed:
            atomic_write_csv(manifest_path, rows)

        completed = sum(1 for row in rows if row.get("status") == "ok")
        pending = len(rows) - completed
        log(
            f"Plan ready: total={len(rows)}, completed={completed}, pending={pending}, "
            f"target_steps={args.target_steps}"
        )
        write_status(
            rows,
            state="running",
            output_root=output_root,
            started_at=args.runner_started_at,
        )

        processed = 0
        stopped = False
        for index, job in enumerate(jobs, 1):
            row = rows_by_id[job.job_id]
            if row.get("status") == "ok":
                continue
            if _STOP_REQUESTED:
                stopped = True
                break
            if args.max_jobs and processed >= args.max_jobs:
                break
            processed += 1
            log(f"Job {index}/{len(jobs)} (run {processed} this invocation)")
            outcome = "failed"
            for retry in range(args.retries + 1):
                outcome = run_attempt(
                    args,
                    job,
                    row,
                    manifest_path=manifest_path,
                    all_rows=rows,
                    events_path=events_path,
                )
                if outcome in {"ok", "interrupted"}:
                    break
                if retry < args.retries and not _STOP_REQUESTED:
                    delay = args.retry_delay_seconds * (retry + 1)
                    log(f"Retrying {job.job_id} in {delay:.1f}s")
                    time.sleep(delay)
            if outcome == "interrupted" or _STOP_REQUESTED:
                stopped = True
                break

        statuses = Counter(str(row.get("status") or "pending") for row in rows)
        ok = statuses.get("ok", 0)
        failures = sum(
            statuses.get(value, 0) for value in ("failed", "timeout", "interrupted")
        )
        if stopped:
            state = "interrupted"
            exit_code = 130
        elif args.max_jobs and ok < len(rows) and processed >= args.max_jobs:
            state = "paused"
            exit_code = 0
        elif ok == len(rows):
            state = "completed"
            exit_code = 0
        elif failures:
            state = "completed_with_failures"
            exit_code = 1
        else:
            state = "paused"
            exit_code = 0
        write_status(
            rows,
            state=state,
            output_root=output_root,
            started_at=args.runner_started_at,
        )
        reached = sum(
            1
            for row in rows
            if row.get("status") == "ok" and row.get("reached_target") == "true"
        )
        log(
            f"Runner {state}: ok={ok}/{len(rows)}, reached_target={reached}, "
            f"failures={failures}, manifest={manifest_path}"
        )
        return exit_code
    finally:
        remove_own_pid_file(pid_path)
        lock.close()


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def stored_runner_pid(output_root: Path) -> int:
    pid_path = output_root / "runner.pid"
    if not pid_path.is_file():
        return -1
    try:
        with pid_path.open(encoding="utf-8") as file:
            return int(json.load(file)["pid"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def print_status(args: argparse.Namespace) -> int:
    output_root = resolve_path(args.output_root)
    manifest_path = output_root / "manifest.csv"
    if not manifest_path.is_file():
        print(f"Not initialized: {manifest_path}")
        return 1
    rows = read_csv(manifest_path)
    statuses = Counter((row.get("status") or "pending") for row in rows)
    reached = sum(
        1
        for row in rows
        if row.get("status") == "ok"
        and (row.get("reached_target") or "").lower() == "true"
    )
    pid = stored_runner_pid(output_root)
    state = "unknown"
    current = ""
    updated = ""
    status_path = output_root / "status.json"
    if status_path.is_file():
        try:
            with status_path.open(encoding="utf-8") as file:
                payload = json.load(file)
            state = str(payload.get("state", state))
            current = str(payload.get("current_job", ""))
            updated = str(payload.get("updated_at", ""))
        except (OSError, json.JSONDecodeError):
            pass
    completed = statuses.get("ok", 0)
    print(
        f"Runner: {state}; PID: {pid if pid > 0 else '-'}; alive: {process_is_alive(pid)}"
    )
    targets = {(row.get("target_steps") or "?") for row in rows}
    target_label = next(iter(targets)) if len(targets) == 1 else "target"
    print(
        f"Progress: {completed}/{len(rows)} valid videos; "
        f"reached {target_label}: {reached}; terminated early: {completed - reached}"
    )
    print(
        "Statuses: "
        + ", ".join(f"{key}={value}" for key, value in sorted(statuses.items()))
    )
    if current:
        print(f"Current job: {current}")
    if updated:
        print(f"Last update: {updated}")
    print(f"Manifest: {manifest_path}")
    print(f"Log: {resolve_log_path(args, output_root)}")
    return 0


def stop_runner(args: argparse.Namespace) -> int:
    output_root = resolve_path(args.output_root)
    pid = stored_runner_pid(output_root)
    if pid <= 0:
        raise RunnerError(f"No runner PID is recorded under {output_root}")
    if not process_is_alive(pid):
        raise RunnerError(f"Recorded runner PID {pid} is not alive")
    proc_command = Path(f"/proc/{pid}/cmdline")
    try:
        command_line = proc_command.read_bytes().split(b"\0")
    except OSError as error:
        raise RunnerError(f"Cannot inspect PID {pid}: {error}") from error
    script = str(Path(__file__).resolve()).encode()
    if script not in command_line:
        raise RunnerError(
            f"PID {pid} does not appear to be this runner; refusing to signal it"
        )
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to runner PID {pid}.")
    print("It will terminate the active child and persist an interrupted status.")
    return 0


def preview(args: argparse.Namespace) -> int:
    output_root = resolve_path(args.output_root)
    validate_runtime(args, output_root)
    jobs, rows, created = load_or_select_jobs(args, write_snapshot=False)
    counts = Counter(job.source_group for job in jobs)
    datasets = Counter(job.dataset for job in jobs)
    statuses = Counter(str(row.get("status") or "pending") for row in rows)
    print(
        f"Selection source: {'source manifests' if created else 'frozen output manifest'}"
    )
    print(f"Selected jobs: {len(jobs)}")
    print(
        "Source groups: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )
    print(
        "Datasets: "
        + ", ".join(f"{key}={value}" for key, value in sorted(datasets.items()))
    )
    print(
        "Current statuses: "
        + ", ".join(f"{key}={value}" for key, value in sorted(statuses.items()))
    )
    print(f"Target: {args.target_steps} steps; sequential execution")
    print(f"Output root: {output_root}")
    print("Dry run only: no files were written and no rollout was started.")
    return 0


def resolve_log_path(args: argparse.Namespace, output_root: Path) -> Path:
    if args.log_file:
        return resolve_path(args.log_file)
    return output_root / "runner.log"


def detach(args: argparse.Namespace) -> int:
    if args.dry_run or args.status:
        raise RunnerError("--detach cannot be combined with --dry-run or --status")
    # Fail before detaching if inputs or the frozen snapshot are invalid.
    output_root = resolve_path(args.output_root)
    validate_runtime(args, output_root)
    jobs, _, _ = load_or_select_jobs(args, write_snapshot=False)
    log_path = resolve_log_path(args, output_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_arguments = [argument for argument in sys.argv[1:] if argument != "--detach"]
    command = [sys.executable, str(Path(__file__).resolve()), *child_arguments]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("ab", buffering=0) as logfile:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    time.sleep(0.5)
    if process.poll() is not None:
        raise RunnerError(
            f"Detached runner exited immediately with code {process.returncode}; inspect {log_path}"
        )
    print(f"Started {len(jobs)}-job runner in the background.")
    print(f"PID: {process.pid}")
    print(f"Log: {log_path}")
    print(f"Manifest: {output_root / 'manifest.csv'}")
    print(
        "Status command: "
        f"{shlex.join([sys.executable, str(Path(__file__).resolve()), '--status', '--output-root', str(output_root)])}"
    )
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the 154 Go2 videos that reached at least 100 steps and rerun "
            "their exact commands with a 1000-step horizon."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--experiment-manifest", type=Path, default=DEFAULT_EXPERIMENT_MANIFEST
    )
    parser.add_argument(
        "--variants-manifest", type=Path, default=DEFAULT_VARIANTS_MANIFEST
    )
    parser.add_argument(
        "--legacy-checkpoint-root", type=Path, default=DEFAULT_LEGACY_CHECKPOINT_ROOT
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument(
        "--source-min-steps", type=positive_int, default=SOURCE_MIN_STEPS
    )
    parser.add_argument("--target-steps", type=positive_int, default=TARGET_STEPS)
    parser.add_argument("--expected-count", type=positive_int, default=EXPECTED_TOTAL)
    parser.add_argument(
        "--allow-selection-count-mismatch",
        action="store_true",
        help="Allow a selection other than the reviewed 149+5 jobs.",
    )
    parser.add_argument("--render-every", type=positive_int, default=2)
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--camera", default="track")
    parser.add_argument(
        "--timeout-seconds",
        type=nonnegative_int,
        default=1800,
        help="Per-job timeout; zero disables it (default: 1800).",
    )
    parser.add_argument(
        "--retries",
        type=nonnegative_int,
        default=1,
        help="Retries after a failed or timed-out attempt (default: 1).",
    )
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-jobs",
        type=nonnegative_int,
        default=0,
        help="Stop cleanly after this many pending jobs; zero means all jobs.",
    )
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--stop", action="store_true")
    action.add_argument(
        "--detach",
        action="store_true",
        help="Start a new session in the background and return immediately.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Combined runner log used by --detach (default: OUTPUT_ROOT/runner.log).",
    )
    args = parser.parse_args()
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be zero or greater")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be zero or greater")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.status:
            return print_status(args)
        if args.stop:
            return stop_runner(args)
        if args.dry_run:
            return preview(args)
        if args.detach:
            return detach(args)
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        return run(args)
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
