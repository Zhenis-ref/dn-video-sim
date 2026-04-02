# DN-Video-Sim: ΔN–ΔD-Guided Branch Compression for Quality and Stability Preservation

**DN-Video-Sim** is a simulation benchmark for testing **ΔN–ΔD-guided branch compression** in a task conceptually related to video-generation planning.

The project studies a practical question: can a bounded-budget trajectory-selection architecture reduce the combinatorial cost of branching **without losing quality and stability**, while remaining effective in an environment where locally attractive branches may later collapse?

## Core idea

In video-generation-like planning, the challenge is not only to choose a locally attractive next step, but also to preserve a stable trajectory over time:

- not to lose character identity;
- not to break temporal consistency;
- not to waste computation on weak or misleading branches;
- not to end up in a “falsely good” path that collapses later.

This project models such a situation in a simplified but conflict-rich environment.

Instead of treating all branches equally, the DN-based controllers act as a structured selection layer:
they estimate branch quality, stability, and risk, and compress the search by rejecting less promising trajectories earlier.

## What is implemented in the current prototype

### 1. Latent video-planning environment

The environment models a simplified trajectory-selection problem conceptually related to video generation planning.

Each branch evolves through quality, stability, and risk-related signals, allowing the benchmark to test whether pruning decisions preserve useful trajectories or destroy them too early.

### 2. Baseline and DN-guided modes

The project includes:

- **baseline** mode — weak pruning without DN-guided structural selection;
- **dn_light** mode — lighter ΔN–ΔD-based trajectory scoring;
- **dn_prune** mode — stronger DN-oriented pruning logic.

These modes are tested under different pruning strengths (`weak`, `medium`, `strong`) in order to show where compression helps, where it becomes destructive, and where it begins to preserve both stability and efficiency.

### 3. Multi-seed benchmark metrics

The benchmark runs multiple seeds and reports aggregate metrics, including:

- success rate;
- quality (`Q`);
- stability (`S`);
- quality drop (`Qdrop`);
- structural cost (`Scost`);
- compression ratio (`CR`);
- predicted bad trajectories (`predBad`);
- actual bad trajectories (`actBad`);
- rejection statistics and gate activity.

## What the benchmark is meant to demonstrate

The project is not a real video generator.

It is a simulation benchmark designed to test whether ΔN–ΔD-guided pruning can:

- reject weak trajectories earlier;
- preserve quality and stability;
- reduce search size;
- suppress unnecessary branching growth;
- expose the boundary between useful compression and destructive over-pruning.

## Main practical result

The central question of the benchmark is:

> Can ΔN–ΔD-guided branch compression preserve trajectory quality and stability while significantly reducing branching cost?

The current results are important because they show that, in the stronger configurations, the DN-based approach can approach baseline quality and stability while substantially reducing the size of the explored search tree.

This is the main claim of the project:

**branching cost can be reduced without giving up the trajectory properties that matter most.**

## Why this matters

In branching generation problems, compute is often wasted not because the model is weak, but because too many low-value trajectories remain alive for too long.

This project therefore serves as a proof-of-concept for the idea that trajectory quality depends not only on raw compute volume, but also on how the branching structure is filtered before deeper rollout begins.

## Repository structure

```text
config/
controllers/
core/
env/
experiments/
artifacts/
requirements.txt
