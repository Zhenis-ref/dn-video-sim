from __future__ import annotations

import copy
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# PROJECT ROOT FIX
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# REAL IMPORTS
# ============================================================

from env.latent_video_env import LatentVideoEnv
from core.dn_metrics import compute_dn_metrics


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [7, 13, 21, 42, 84]
DEFAULT_MODES = ["baseline", "dn_light", "dn_prune"]
DEFAULT_PRUNE_LEVELS = ["weak", "medium", "strong"]

MAX_STEPS = 18

EVAL_BUDGET_BASELINE = 4
EVAL_BUDGET_DN_LIGHT = 4
EVAL_BUDGET_DN_PRUNE = 6

# ---- early risk gate ----
EARLY_PHASE_LAST_STEP = 5

# Главное изменение: работаем не с raw pred_risk, а с effective_risk = pred_risk * 5.0
RISK_SCALE = 5.0

RISK_GATE_THRESHOLDS = {
    "baseline": 9.99,   # baseline не режем по риску
    "dn_light": {
        "weak": 0.55,
        "medium": 0.48,
        "strong": 0.40,
    },
    "dn_prune": {
        "weak": 0.50,
        "medium": 0.44,
        "strong": 0.36,
    },
}

RISK_HARD_PENALTY = {
    "dn_light": 0.30,
    "dn_prune": 0.42,
}

MIN_SAFE_CANDIDATES = 1

PRED_BAD_TRIGGER = 0.50


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class StepLog:
    step_idx: int
    mode: str
    prune_level: str
    seed: int

    frontier_size: int
    expanded_nodes: int
    kept_nodes: int
    risk_rejected_nodes: int

    delta_n: float
    delta_d: float
    ds_dt: float

    quality_now: float
    stability_now: float
    identity_now: float
    temporal_now: float
    style_now: float
    motion_now: float

    cumulative_stability_cost: float
    cumulative_quality_drop: float

    predicted_risk_raw: float
    predicted_risk_effective: float
    predicted_final_quality: float
    max_candidate_risk_this_step: float
    risk_gate_triggered: bool
    notes: str = ""


@dataclass
class EpisodeResult:
    mode: str
    prune_level: str
    seed: int

    success: bool
    steps: int

    quality_final: float
    quality_best_seen: float
    quality_drop_from_best: float

    stability_final: float
    identity_final: float
    temporal_final: float
    style_final: float
    motion_final: float

    compression_ratio_nodes: float
    total_expanded_nodes: int
    total_kept_nodes: int
    total_stability_cost: float

    predicted_bad_early: bool
    actual_bad_final: bool

    max_early_risk_seen: float
    total_risk_rejected_nodes: int
    risk_gate_triggered_episode: bool

    failure_reason: str = ""


# ============================================================
# HELPERS
# ============================================================

def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def classify_failure(ep: EpisodeResult) -> str:
    if ep.quality_final < 0.55 and ep.stability_final >= 0.68:
        return "quality_collapsed"
    if ep.stability_final < 0.60 and ep.quality_final >= 0.60:
        return "stability_collapsed"
    if ep.quality_final < 0.55 and ep.stability_final < 0.60:
        return "both_collapsed"
    if ep.compression_ratio_nodes < 1.15 and ep.mode != "baseline":
        return "compression_useless"
    return "ok"


def get_prune_strength(level: str) -> Dict[str, float]:
    table = {
        "weak": {
            "keep_frac": 0.75,
            "temperature": 0.20,
        },
        "medium": {
            "keep_frac": 0.50,
            "temperature": 0.10,
        },
        "strong": {
            "keep_frac": 0.34,
            "temperature": 0.00,
        },
    }
    return table[level]


def get_eval_budget(mode: str) -> int:
    if mode == "baseline":
        return EVAL_BUDGET_BASELINE
    if mode == "dn_light":
        return EVAL_BUDGET_DN_LIGHT
    return EVAL_BUDGET_DN_PRUNE


def get_risk_threshold(mode: str, prune_level: str) -> float:
    if mode == "baseline":
        return float(RISK_GATE_THRESHOLDS["baseline"])
    return float(RISK_GATE_THRESHOLDS[mode][prune_level])


def softmax_sample(scored: List[Tuple[float, Any]], temperature: float, rng: random.Random) -> Any:
    if not scored:
        raise RuntimeError("softmax_sample() received empty list")

    if temperature <= 1e-9:
        return scored[0][1]

    values = [x[0] / max(temperature, 1e-6) for x in scored]
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    z = sum(exps)

    r = rng.random()
    acc = 0.0
    for w, (_, action) in zip(exps, scored):
        acc += w / z
        if r <= acc:
            return action

    return scored[-1][1]


# ============================================================
# ENV ADAPTER
# ============================================================

class EnvAdapter:
    def __init__(self, seed: int):
        self.env = LatentVideoEnv(seed=seed)

    def reset(self) -> Dict[str, float]:
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        return self.env.step(action)

    def sample_action_candidates(self) -> List[Any]:
        return self.env.sample_action_candidates()


# ============================================================
# OBSERVABLE SCORING
# ============================================================

def observable_state_score(info: Dict[str, Any]) -> float:
    quality = float(info.get("quality", 0.0))
    stability = float(info.get("stability", 0.0))
    identity = float(info.get("identity", 0.0))
    temporal = float(info.get("temporal", 0.0))
    style = float(info.get("style", 0.0))
    motion = float(info.get("motion", 0.0))
    drift = float(info.get("drift", 0.0))
    deg = float(info.get("cumulative_degradation", 0.0))

    agreement = 1.0 - (
        abs(identity - temporal)
        + abs(style - motion)
        + 0.5 * abs(identity - motion)
    ) / 2.5
    agreement = max(0.0, min(1.0, agreement))

    return (
        0.42 * quality
        + 0.18 * stability
        + 0.14 * agreement
        - 0.14 * deg
        - 0.12 * drift
    )


def compute_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    result = compute_dn_metrics(info)
    raw_risk = float(result["predicted_final_risk"])
    effective_risk = raw_risk * RISK_SCALE

    return {
        "delta_n": float(result["delta_n"]),
        "delta_d": float(result["delta_d"]),
        "ds_dt": float(result["ds_dt"]),
        "predicted_final_risk_raw": raw_risk,
        "predicted_final_risk_effective": effective_risk,
        "predicted_final_quality": float(result["predicted_final_quality"]),
    }


# ============================================================
# REAL ONE-STEP ROLLOUT
# ============================================================

def evaluate_action_by_real_rollout(
    env_obj: Any,
    action: Any,
) -> Dict[str, float]:
    env_copy = copy.deepcopy(env_obj)
    _, reward, done, info = env_copy.step(action)
    del reward, done

    dn = compute_metrics(info)
    obs_score = observable_state_score(info)

    return {
        "obs_score": obs_score,
        "quality": float(info["quality"]),
        "stability": float(info["stability"]),
        "delta_n": dn["delta_n"],
        "delta_d": dn["delta_d"],
        "pred_risk_raw": dn["predicted_final_risk_raw"],
        "pred_risk_effective": dn["predicted_final_risk_effective"],
        "pred_q": dn["predicted_final_quality"],
    }


def shortlist_candidates(
    candidates: List[Any],
    budget: int,
    rng: random.Random,
) -> List[Any]:
    if len(candidates) <= budget:
        return list(candidates)

    sampled = list(candidates)
    rng.shuffle(sampled)
    return sampled[:budget]


# ============================================================
# CONTROLLERS
# ============================================================

def baseline_pick_action(
    env_obj: Any,
    candidates: List[Any],
    rng: random.Random,
) -> Tuple[Any, int, int, int, float, bool]:
    budget = get_eval_budget("baseline")
    sampled = shortlist_candidates(candidates, budget, rng)

    scored: List[Tuple[float, Any]] = []
    max_risk = 0.0

    for action in sampled:
        ev = evaluate_action_by_real_rollout(env_obj, action)
        max_risk = max(max_risk, ev["pred_risk_effective"])
        scored.append((ev["obs_score"], action))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[0][1]

    expanded_nodes = len(candidates)
    kept_nodes = len(sampled)
    risk_rejected_nodes = 0
    risk_gate_triggered = False
    return chosen, expanded_nodes, kept_nodes, risk_rejected_nodes, max_risk, risk_gate_triggered


def dn_pick_action(
    env_obj: Any,
    candidates: List[Any],
    mode: str,
    prune_level: str,
    rng: random.Random,
    step_idx: int,
) -> Tuple[Any, int, int, int, float, bool]:
    budget = get_eval_budget(mode)
    sampled = shortlist_candidates(candidates, budget, rng)

    threshold = get_risk_threshold(mode, prune_level)
    cfg = get_prune_strength(prune_level)

    all_scored: List[Tuple[float, Any, float]] = []
    safe_scored: List[Tuple[float, Any, float]] = []

    max_risk = 0.0
    risk_rejected_nodes = 0
    risk_gate_triggered = False

    for action in sampled:
        ev = evaluate_action_by_real_rollout(env_obj, action)
        risk = ev["pred_risk_effective"]
        max_risk = max(max_risk, risk)

        dn_score = (
            0.30 * ev["quality"]
            + 0.18 * ev["stability"]
            + 0.20 * ev["pred_q"]
            - 0.14 * ev["delta_d"]
            - 0.10 * ev["delta_n"]
            - 0.08 * risk
        )

        if mode == "dn_prune":
            dn_score -= 0.08 * risk
            dn_score -= 0.05 * ev["delta_n"]

        if step_idx <= EARLY_PHASE_LAST_STEP and risk > threshold:
            risk_gate_triggered = True
            risk_rejected_nodes += 1
            dn_score -= RISK_HARD_PENALTY[mode]
            all_scored.append((dn_score, action, risk))
            continue

        safe_scored.append((dn_score, action, risk))
        all_scored.append((dn_score, action, risk))

    working = safe_scored if len(safe_scored) >= MIN_SAFE_CANDIDATES else all_scored
    working.sort(key=lambda x: x[0], reverse=True)

    keep_n = max(1, math.ceil(len(working) * cfg["keep_frac"]))
    kept = working[:keep_n]

    picked_pool = [(score, action) for score, action, _ in kept]
    chosen = softmax_sample(picked_pool, cfg["temperature"], rng)

    expanded_nodes = len(candidates)
    kept_nodes = len(kept)

    return chosen, expanded_nodes, kept_nodes, risk_rejected_nodes, max_risk, risk_gate_triggered


# ============================================================
# EPISODE
# ============================================================

def run_episode(
    mode: str,
    prune_level: str,
    seed: int,
) -> Tuple[EpisodeResult, List[StepLog]]:
    rng = random.Random(seed)
    adapter = EnvAdapter(seed)
    info = adapter.reset()

    step_logs: List[StepLog] = []

    total_expanded = 0
    total_kept = 0
    total_stability_cost = 0.0
    best_quality_seen = float(info["quality"])

    predicted_bad_early = False
    max_early_risk_seen = 0.0
    total_risk_rejected_nodes = 0
    risk_gate_triggered_episode = False

    for step_idx in range(MAX_STEPS):
        candidates = adapter.sample_action_candidates()

        if mode == "baseline":
            action, expanded_nodes, kept_nodes, risk_rejected_nodes, max_candidate_risk, risk_gate_triggered = baseline_pick_action(
                env_obj=adapter.env,
                candidates=candidates,
                rng=rng,
            )
        else:
            action, expanded_nodes, kept_nodes, risk_rejected_nodes, max_candidate_risk, risk_gate_triggered = dn_pick_action(
                env_obj=adapter.env,
                candidates=candidates,
                mode=mode,
                prune_level=prune_level,
                rng=rng,
                step_idx=step_idx,
            )

        info, reward, done, info = adapter.step(action)
        del reward

        dn = compute_metrics(info)

        q = float(info["quality"])
        s = float(info["stability"])

        best_quality_seen = max(best_quality_seen, q)
        total_expanded += expanded_nodes
        total_kept += kept_nodes

        stability_cost = max(0.0, 0.78 - s)
        total_stability_cost += stability_cost

        quality_drop = max(0.0, best_quality_seen - q)

        if step_idx <= EARLY_PHASE_LAST_STEP:
            max_early_risk_seen = max(max_early_risk_seen, max_candidate_risk)
            if max_candidate_risk > PRED_BAD_TRIGGER:
                predicted_bad_early = True

        total_risk_rejected_nodes += risk_rejected_nodes
        if risk_gate_triggered:
            risk_gate_triggered_episode = True

        step_logs.append(
            StepLog(
                step_idx=step_idx,
                mode=mode,
                prune_level=prune_level,
                seed=seed,
                frontier_size=len(candidates),
                expanded_nodes=expanded_nodes,
                kept_nodes=kept_nodes,
                risk_rejected_nodes=risk_rejected_nodes,
                delta_n=dn["delta_n"],
                delta_d=dn["delta_d"],
                ds_dt=dn["ds_dt"],
                quality_now=q,
                stability_now=s,
                identity_now=float(info["identity"]),
                temporal_now=float(info["temporal"]),
                style_now=float(info["style"]),
                motion_now=float(info["motion"]),
                cumulative_stability_cost=total_stability_cost,
                cumulative_quality_drop=quality_drop,
                predicted_risk_raw=dn["predicted_final_risk_raw"],
                predicted_risk_effective=dn["predicted_final_risk_effective"],
                predicted_final_quality=dn["predicted_final_quality"],
                max_candidate_risk_this_step=max_candidate_risk,
                risk_gate_triggered=risk_gate_triggered,
                notes="",
            )
        )

        if done:
            break

    quality_final = float(info["quality"])
    stability_final = float(info["stability"])
    identity_final = float(info["identity"])
    temporal_final = float(info["temporal"])
    style_final = float(info["style"])
    motion_final = float(info["motion"])

    quality_drop_from_best = max(0.0, best_quality_seen - quality_final)

    success = (quality_final >= 0.72 and stability_final >= 0.70)

    cr_nodes = 1.0 if mode == "baseline" else (float(total_expanded) / float(total_kept))

    episode = EpisodeResult(
        mode=mode,
        prune_level=prune_level,
        seed=seed,
        success=success,
        steps=len(step_logs),
        quality_final=quality_final,
        quality_best_seen=best_quality_seen,
        quality_drop_from_best=quality_drop_from_best,
        stability_final=stability_final,
        identity_final=identity_final,
        temporal_final=temporal_final,
        style_final=style_final,
        motion_final=motion_final,
        compression_ratio_nodes=cr_nodes,
        total_expanded_nodes=total_expanded,
        total_kept_nodes=total_kept,
        total_stability_cost=total_stability_cost,
        predicted_bad_early=predicted_bad_early,
        actual_bad_final=(quality_final < 0.60 or stability_final < 0.60),
        max_early_risk_seen=max_early_risk_seen,
        total_risk_rejected_nodes=total_risk_rejected_nodes,
        risk_gate_triggered_episode=risk_gate_triggered_episode,
        failure_reason="",
    )
    episode.failure_reason = classify_failure(episode)
    return episode, step_logs


# ============================================================
# SUMMARY
# ============================================================

def summarize(results: List[EpisodeResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[EpisodeResult]] = {}
    for r in results:
        grouped.setdefault((r.mode, r.prune_level), []).append(r)

    rows: List[Dict[str, Any]] = []
    for (mode, prune_level), group in grouped.items():
        rows.append({
            "mode": mode,
            "prune_level": prune_level,
            "runs": len(group),
            "success_rate": round(safe_mean([1.0 if x.success else 0.0 for x in group]), 4),
            "quality_final_mean": round(safe_mean([x.quality_final for x in group]), 4),
            "stability_final_mean": round(safe_mean([x.stability_final for x in group]), 4),
            "quality_drop_mean": round(safe_mean([x.quality_drop_from_best for x in group]), 4),
            "stability_cost_mean": round(safe_mean([x.total_stability_cost for x in group]), 4),
            "cr_nodes_mean": round(safe_mean([x.compression_ratio_nodes for x in group]), 4),
            "predicted_bad_rate": round(safe_mean([1.0 if x.predicted_bad_early else 0.0 for x in group]), 4),
            "actual_bad_rate": round(safe_mean([1.0 if x.actual_bad_final else 0.0 for x in group]), 4),
            "max_early_risk_mean": round(safe_mean([x.max_early_risk_seen for x in group]), 4),
            "risk_rejected_nodes_mean": round(safe_mean([float(x.total_risk_rejected_nodes) for x in group]), 4),
            "risk_gate_triggered_rate": round(safe_mean([1.0 if x.risk_gate_triggered_episode else 0.0 for x in group]), 4),
            "quality_collapsed": sum(1 for x in group if x.failure_reason == "quality_collapsed"),
            "stability_collapsed": sum(1 for x in group if x.failure_reason == "stability_collapsed"),
            "both_collapsed": sum(1 for x in group if x.failure_reason == "both_collapsed"),
            "compression_useless": sum(1 for x in group if x.failure_reason == "compression_useless"),
        })

    rows.sort(key=lambda x: (x["mode"], x["prune_level"]))
    return rows


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n=== DIAGNOSTIC SUMMARY ===")
    for r in rows:
        print(
            f"{r['mode']:>9} | {r['prune_level']:<6} | "
            f"runs={r['runs']:<2} | "
            f"succ={r['success_rate']:.2f} | "
            f"Q={r['quality_final_mean']:.3f} | "
            f"S={r['stability_final_mean']:.3f} | "
            f"Qdrop={r['quality_drop_mean']:.3f} | "
            f"Scost={r['stability_cost_mean']:.3f} | "
            f"CR={r['cr_nodes_mean']:.3f} | "
            f"predBad={r['predicted_bad_rate']:.2f} | "
            f"actBad={r['actual_bad_rate']:.2f} | "
            f"maxERisk={r['max_early_risk_mean']:.2f} | "
            f"rej={r['risk_rejected_nodes_mean']:.2f} | "
            f"gate={r['risk_gate_triggered_rate']:.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    all_results: List[EpisodeResult] = []
    all_steps: List[StepLog] = []

    for seed in DEFAULT_SEEDS:
        for mode in DEFAULT_MODES:
            prune_levels = ["weak"] if mode == "baseline" else DEFAULT_PRUNE_LEVELS

            for prune_level in prune_levels:
                print(f"[RUN] seed={seed} mode={mode} prune={prune_level}")
                episode, steps = run_episode(
                    mode=mode,
                    prune_level=prune_level,
                    seed=seed,
                )
                all_results.append(episode)
                all_steps.extend(steps)

    summary_rows = summarize(all_results)
    print_summary_table(summary_rows)

    save_json(OUTPUT_DIR / "episode_results.json", [asdict(x) for x in all_results])
    save_json(OUTPUT_DIR / "step_logs.json", [asdict(x) for x in all_steps])
    save_json(OUTPUT_DIR / "summary.json", summary_rows)

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()