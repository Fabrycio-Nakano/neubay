# RoboWM 2026 Experiment Instructions

## Goal

We are preparing a short 2–4 page workshop paper evaluating the reliability
of NEUBAY world-model uncertainty under controlled dynamics shifts in Unitree Go2 locomotion.

The project is time-constrained. Prefer minimal, reproducible experiments over
large refactors or new training pipelines.

## Existing assets

This repository already contains, or is expected to contain:

- a NEUBAY world-model ensemble trained for Unitree Go2;
- a trained NEUBAY agent/policy;
- the offline training dataset;
- environment configuration;
- training seeds/configurations;
- nominal Go2 evaluation code.

Do not assume file locations. Inspect the repository first.

## Scientific question

Main question:

Does NEUBAY ensemble uncertainty remain informative when the physical dynamics
of the Go2 simulator differ from the dynamics represented in the training data?

We are NOT trying to prove whether NEUBAY works in general.

We are testing whether its uncertainty signal detects world-model prediction
failure under controlled dynamics mismatch.

## Experimental logic

The trained world model and policy must initially remain frozen.

Baseline:
- evaluate using the nominal simulator configuration.

Dynamics shifts:
1. ground/contact friction;
2. robot payload/body mass.

Optional only if time permits:
3. actuator strength;
4. joint damping.

For every condition, collect:

- real simulator state s[t+1];
- world-model prediction s_hat[t+1];
- ensemble uncertainty;
- action;
- episode return;
- fall/success status;
- relevant locomotion metrics.

Primary relationship:

dynamics shift
    -> world-model prediction error
    -> ensemble uncertainty

Secondary relationship:

world-model reliability
    -> locomotion performance

## Important methodological constraints

- Do not retrain the world model unless explicitly requested.
- Do not retrain the policy unless explicitly requested.
- Do not modify the original training pipeline unnecessarily.
- Keep RoboWM-specific evaluation code isolated.
- Prefer adding code under something like:
  experiments/robowm2026/
- Preserve backward compatibility with existing Go2 experiments.
- Never silently change the nominal environment configuration.
- Every dynamics perturbation must be explicitly logged.
- Use deterministic seeds whenever possible.
- Store raw numerical results before generating plots.
- Prefer CSV/JSON/NPZ outputs that can later be analyzed independently.
- Do not delete existing checkpoints or experimental results.

## Engineering priorities

Priority order:

1. reproduce one nominal Go2 episode;
2. load the existing world-model ensemble;
3. expose world-model prediction and uncertainty;
4. compute prediction error against simulator transitions;
5. run a friction sweep;
6. run a payload sweep;
7. aggregate results;
8. create publication-ready plots.

Do not work on later priorities while an earlier one is broken.

## First-pass dynamics sweep

Prefer a small sweep first.

Friction multiplier:
- 1.0
- 0.8
- 0.6
- 0.4

Payload/body-mass multiplier:
- 1.0
- 1.1
- 1.2
- 1.3

These are provisional experimental values, not immutable scientific choices.

## Metrics

At minimum calculate:

### World-model metrics
- one-step prediction error;
- rollout prediction error when applicable;
- ensemble uncertainty;
- correlation between uncertainty and prediction error.

If enough samples exist:
- AUROC for using uncertainty to detect high prediction error;
- false-safe rate:
  high prediction error + low uncertainty.

### Robot metrics
- episode return;
- fall rate / episode termination;
- velocity tracking error if available.

Do not invent thresholds for "high error" without reporting how they were chosen.

## Working style

Before changing code:

1. inspect the relevant implementation;
2. explain what already exists;
3. identify the smallest modification required;
4. state which files will change.

After changing code:

1. run the smallest meaningful test;
2. report exact commands;
3. report produced output files;
4. report failures explicitly;
5. do not claim success without evidence.

Avoid broad refactoring during the workshop deadline period.
