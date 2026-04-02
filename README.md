# DN-Video-Sim: Diagnostic Simulation for ΔN–ΔD-Guided Trajectory Selection

**DN-Video-Sim** is a diagnostic simulation project for testing **ΔN–ΔD-guided trajectory selection and pruning** in a task conceptually related to video-generation planning.

The project studies a practical question: can a bounded-budget trajectory-selection architecture reduce the combinatorial cost of branching **without losing quality and stability**, while remaining effective in a simplified environment where locally attractive branches may later collapse.

## Core idea

In real video generation, the problem is often not only to obtain a locally attractive next step, but also to maintain a stable trajectory over time:

- not to lose character identity;
- not to break temporal consistency;
- not to waste computation on weak or misleading branches;
- not to end up in a “falsely good” path that collapses later.

This project models such a situation in a simplified but conflict-rich environment.

Instead of evaluating all possible branches equally, the DN-based controllers act as a structured selection layer:
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

These modes are tested under different pruning strengths (`weak`, `medium`, `strong`) in order to reveal where compression helps and where it becomes destructive.

### 3. Multi-seed diagnostics

The benchmark runs multiple seeds and reports aggregate diagnostics, including:

- success rate;
- quality (`Q`);
- stability (`S`);
- quality drop (`Qdrop`);
- structural cost (`Scost`);
- compression ratio (`CR`);
- predicted bad trajectories (`predBad`);
- actual bad trajectories (`actBad`);
- rejection statistics and gate activity.

## What the benchmark is designed to test

The project is not a real video generator.
It is a **diagnostic simulation** designed to test whether ΔN–ΔD-guided pruning can:

- reject weak trajectories earlier;
- preserve quality and stability;
- reduce search size;
- expose the boundary between useful compression and destructive over-pruning.

## Example interpretation of results

A typical diagnostic summary compares the baseline against DN-guided modes across multiple seeds.

The main practical question is:

> Can ΔN–ΔD-guided pruning preserve trajectory quality and stability while significantly reducing branching cost?

In this benchmark, the strongest DN configurations are especially important because they show whether the method becomes genuinely useful only under disciplined pruning pressure, or whether softer modes can also remain safe.

## Why this matters

In branching generation problems, compute is often wasted not because the model is weak, but because too many low-value trajectories remain alive for too long.

This project therefore serves as a proof-of-concept for the idea that trajectory quality depends not only on raw compute volume, but also on how the branching structure is filtered before deeper rollout.

## Repository structure

```text
config/
controllers/
core/
env/
artifacts/
main.py