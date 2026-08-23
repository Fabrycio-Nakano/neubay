# RoboWM 2026 — Go2 World-Model Reliability Study

## Working title

When the Ground Changes:
Stress-Testing World-Model Uncertainty under Dynamics Shifts
in Quadruped Locomotion

## Motivation

NEUBAY uses an ensemble of learned world models and ensemble disagreement
as an epistemic uncertainty signal.

The question in this work is not whether NEUBAY can solve Go2 locomotion.

Instead, we evaluate whether ensemble uncertainty remains a useful indicator
of world-model prediction failure when the simulator dynamics differ from the
training dynamics.

## Core hypothesis

When dynamics shift away from training conditions, prediction error should
increase.

If ensemble uncertainty is a reliable failure signal, uncertainty should also
increase as prediction error increases.

Potential failure mode:

prediction error = high
ensemble uncertainty = low

This corresponds to a "false-safe" or "confidently wrong" prediction.

## RQs

### RQ1
How accurately does the NEUBAY world model predict Go2 transitions under
nominal dynamics?

### RQ2
Does ensemble uncertainty remain informative as physical dynamics progressively
deviate from nominal conditions?

### RQ3
Does uncertainty reliability depend on the type of dynamics shift?

Primary comparison:
- ground friction;
- payload/body mass.

### RQ4
Are failures in world-model reliability associated with degradation in
locomotion performance?

## Experimental pipeline

Offline Go2 dataset
    ->
trained NEUBAY world-model ensemble
    ->
trained NEUBAY policy
    ->
freeze models
    ->
evaluate in simulator

For each simulator step:

state s_t
action a_t

Simulator:
(s_t, a_t) -> s_{t+1}

World model:
(s_t, a_t) -> predicted state s_hat_{t+1}

Calculate:

prediction_error_t = distance(s_{t+1}, s_hat_{t+1})

and obtain:

uncertainty_t = NEUBAY ensemble disagreement

Then analyze:

shift magnitude
vs
prediction error
vs
uncertainty
vs
robot performance.

## Critical distinction

Policy robustness and world-model reliability are not the same thing.

A robot may continue walking even while its world model becomes inaccurate.

Likewise, locomotion may degrade while ensemble uncertainty fails to increase.

The primary paper contribution concerns world-model reliability.
Policy performance is supporting evidence.

## Initial conditions

First establish a nominal baseline.

Do not apply dynamics shifts until:

- existing checkpoints load;
- nominal policy evaluation works;
- world-model prediction can be extracted;
- uncertainty can be extracted;
- prediction error can be computed.

## Friction experiment

Keep the learned model frozen.

Change only simulator dynamics.

Initial sweep:

friction multiplier:
1.0, 0.8, 0.6, 0.4

For each condition:
- multiple episodes/seeds;
- save every transition;
- save prediction;
- save uncertainty;
- save performance metrics.

## Payload experiment

Body/payload mass multiplier:

1.0, 1.1, 1.2, 1.3

Same measurements as friction.

## Potential outcomes

### Outcome A
error increases
uncertainty does not

Interpretation:
ensemble becomes overconfident / false-safe.

### Outcome B
error increases
uncertainty increases

Interpretation:
ensemble disagreement is robust to this dynamics shift.

### Outcome C
uncertainty tracks error for mild shifts but fails for severe shifts

Interpretation:
there is a measurable reliability boundary.

### Outcome D
friction and payload produce different behavior

Interpretation:
uncertainty reliability depends on the nature of the dynamics mismatch.

All four can be scientifically useful if reproducible.

## Minimum viable paper

The minimum viable experiment is:

- Go2;
- nominal dynamics;
- friction sweep;
- at least 3 seeds/episodes per condition;
- prediction error;
- uncertainty;
- robot performance;
- one clear reproducible trend.

Payload is the first extension.

## Deadline principle

Do not expand the scope until the minimum friction experiment produces
interpretable results.
