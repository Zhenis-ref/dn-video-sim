from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import math


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DNMetricSnapshot:
    quality_now: float
    quality_prev: float
    quality_best: float
    quality_target: float

    identity: float
    temporal: float
    style: float
    motion: float
    stability: float

    cumulative_degradation: float
    drift: float

    delta_n: float
    delta_d: float
    ds_dt: float

    aspect_variance: float
    agreement_score: float

    quality_vs_trajectory_conflict: float
    temporal_identity_conflict: float
    style_motion_conflict: float
    stability_quality_gap: float

    false_beauty_score: float
    degradation_pressure: float
    structural_fragility: float

    predicted_final_risk: float
    predicted_final_quality: float


# ============================================================
# BASIC HELPERS
# ============================================================

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _variance(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


# ============================================================
# STATE EXTRACTION
# ============================================================

def extract_signal_bundle(state_or_info: Dict[str, Any]) -> Dict[str, float]:
    quality = _clip(_safe_float(state_or_info.get("quality", 0.0)))
    quality_prev = _clip(_safe_float(state_or_info.get("quality_prev", quality)))
    quality_best = _clip(_safe_float(state_or_info.get("quality_best", quality)))
    quality_target = _clip(_safe_float(state_or_info.get("quality_target", 1.0)))

    identity = _clip(_safe_float(state_or_info.get("identity", 0.0)))
    temporal = _clip(_safe_float(state_or_info.get("temporal", 0.0)))
    style = _clip(_safe_float(state_or_info.get("style", 0.0)))
    motion = _clip(_safe_float(state_or_info.get("motion", 0.0)))
    stability = _clip(_safe_float(state_or_info.get("stability", 0.0)))

    cumulative_degradation = _clip(_safe_float(state_or_info.get("cumulative_degradation", 0.0)))
    drift = _clip(_safe_float(state_or_info.get("drift", 0.0)))

    return {
        "quality": quality,
        "quality_prev": quality_prev,
        "quality_best": quality_best,
        "quality_target": quality_target,
        "identity": identity,
        "temporal": temporal,
        "style": style,
        "motion": motion,
        "stability": stability,
        "cumulative_degradation": cumulative_degradation,
        "drift": drift,
    }


# ============================================================
# STRUCTURAL FEATURES
# ============================================================

def compute_structural_features(bundle: Dict[str, float]) -> Dict[str, float]:
    quality = bundle["quality"]
    identity = bundle["identity"]
    temporal = bundle["temporal"]
    style = bundle["style"]
    motion = bundle["motion"]
    stability = bundle["stability"]
    cumulative_degradation = bundle["cumulative_degradation"]
    drift = bundle["drift"]

    aspects = [identity, temporal, style, motion, stability]
    aspect_variance = _variance(aspects)

    conflict_sum = (
        abs(identity - temporal)
        + abs(style - motion)
        + 0.5 * abs(identity - motion)
    ) / 2.5
    agreement_score = _clip(1.0 - conflict_sum)

    trajectory_proxy = 0.45 * temporal + 0.30 * motion + 0.25 * stability
    quality_vs_trajectory_conflict = abs(quality - trajectory_proxy)

    temporal_identity_conflict = abs(temporal - identity)
    style_motion_conflict = abs(style - motion)

    # Если quality заметно выше stability, это часто ложная красота.
    stability_quality_gap = max(0.0, quality - stability)

    # Ложная красота:
    # - quality выглядит хорошо,
    # - но stability/trajectory/consistency не подтверждают.
    false_beauty_score = _clip(
        0.45 * stability_quality_gap
        + 0.25 * quality_vs_trajectory_conflict
        + 0.15 * temporal_identity_conflict
        + 0.15 * style_motion_conflict
    )

    # Давление накопленного будущего развала
    degradation_pressure = _clip(
        0.58 * cumulative_degradation
        + 0.42 * drift
    )

    # Хрупкость структуры:
    # если конфликт высокий и при этом stability не держит систему.
    structural_fragility = _clip(
        0.35 * min(1.0, aspect_variance * 4.0)
        + 0.25 * max(0.0, 0.72 - stability)
        + 0.20 * temporal_identity_conflict
        + 0.20 * style_motion_conflict
    )

    return {
        "aspect_variance": aspect_variance,
        "agreement_score": agreement_score,
        "quality_vs_trajectory_conflict": quality_vs_trajectory_conflict,
        "temporal_identity_conflict": temporal_identity_conflict,
        "style_motion_conflict": style_motion_conflict,
        "stability_quality_gap": stability_quality_gap,
        "false_beauty_score": false_beauty_score,
        "degradation_pressure": degradation_pressure,
        "structural_fragility": structural_fragility,
    }


# ============================================================
# DELTA N
# ============================================================

def compute_delta_n(bundle: Dict[str, float], feats: Dict[str, float]) -> float:
    """
    ΔN = внешний рабочий градиент.
    Здесь важно не только "далеки ли мы от цели",
    но и наблюдается ли откат / напряжение незавершённости.
    """
    quality = bundle["quality"]
    quality_prev = bundle["quality_prev"]
    quality_best = bundle["quality_best"]
    quality_target = bundle["quality_target"]

    target_gap = max(0.0, quality_target - quality)
    best_gap = max(0.0, quality_best - quality)
    rollback = max(0.0, quality_prev - quality)

    delta_n = (
        0.58 * target_gap
        + 0.24 * best_gap
        + 0.18 * rollback
    )
    return _clip(delta_n)


# ============================================================
# DELTA D
# ============================================================

def compute_delta_d(bundle: Dict[str, float], feats: Dict[str, float]) -> float:
    """
    ΔD = внутренняя конфликтность и несогласованность.
    Усиливаем вклад "ложной красоты" и деградационного давления.
    """
    delta_d = (
        0.20 * min(1.0, feats["aspect_variance"] * 4.0)
        + 0.18 * feats["quality_vs_trajectory_conflict"]
        + 0.12 * feats["temporal_identity_conflict"]
        + 0.10 * feats["style_motion_conflict"]
        + 0.16 * feats["false_beauty_score"]
        + 0.14 * feats["degradation_pressure"]
        + 0.10 * feats["structural_fragility"]
    )
    return _clip(delta_d)


# ============================================================
# dS/dt
# ============================================================

def compute_ds_dt(delta_n: float, delta_d: float, feats: Dict[str, float]) -> float:
    """
    Для диагностики dS/dt трактуем как интенсивность проблемной динамики:
    - высокий внешний градиент
    - высокая внутренняя конфликтность
    - усиление при ложной красоте / хрупкости
    """
    ds_dt = (
        1.05 * delta_n
        + 0.90 * delta_d
        + 0.55 * delta_n * delta_d
        + 0.22 * feats["false_beauty_score"]
        + 0.18 * feats["structural_fragility"]
    )
    return max(0.0, ds_dt)


# ============================================================
# PREDICTORS
# ============================================================

def predict_final_risk(bundle: Dict[str, float], feats: Dict[str, float], delta_n: float, delta_d: float) -> float:
    quality = bundle["quality"]
    stability = bundle["stability"]
    cumulative_degradation = bundle["cumulative_degradation"]
    drift = bundle["drift"]

    # Главная идея:
    # риск должен срабатывать даже когда текущее quality ещё "нормальное",
    # если при этом:
    # - stability отстаёт,
    # - есть ложная красота,
    # - копятся drift/degradation,
    # - структура конфликтная.
    risk = (
        0.20 * delta_n
        + 0.18 * delta_d
        + 0.16 * feats["false_beauty_score"]
        + 0.15 * feats["degradation_pressure"]
        + 0.13 * feats["structural_fragility"]
        + 0.10 * max(0.0, 0.75 - stability)
        + 0.08 * feats["quality_vs_trajectory_conflict"]
    )

    # Дополнительный boost риска, если quality уже красивое, а stability нет.
    if quality >= 0.72 and stability < 0.68:
        risk += 0.12

    # Ещё один boost, если накопление уже заметное.
    if cumulative_degradation > 0.24 or drift > 0.24:
        risk += 0.08

    return _clip(risk)


def predict_final_quality(bundle: Dict[str, float], feats: Dict[str, float], delta_n: float, delta_d: float) -> float:
    quality = bundle["quality"]
    stability = bundle["stability"]
    agreement = feats["agreement_score"]

    predicted = (
        0.46 * quality
        + 0.20 * stability
        + 0.12 * agreement
        + 0.10 * (1.0 - delta_n)
        + 0.12 * (1.0 - feats["degradation_pressure"])
        - 0.16 * feats["false_beauty_score"]
        - 0.10 * delta_d
    )
    return _clip(predicted)


# ============================================================
# PUBLIC API
# ============================================================

def compute_dn_metrics(state_or_info: Dict[str, Any]) -> Dict[str, float]:
    bundle = extract_signal_bundle(state_or_info)
    feats = compute_structural_features(bundle)

    delta_n = compute_delta_n(bundle, feats)
    delta_d = compute_delta_d(bundle, feats)
    ds_dt = compute_ds_dt(delta_n, delta_d, feats)

    predicted_final_risk = predict_final_risk(bundle, feats, delta_n, delta_d)
    predicted_final_quality = predict_final_quality(bundle, feats, delta_n, delta_d)

    snapshot = DNMetricSnapshot(
        quality_now=bundle["quality"],
        quality_prev=bundle["quality_prev"],
        quality_best=bundle["quality_best"],
        quality_target=bundle["quality_target"],
        identity=bundle["identity"],
        temporal=bundle["temporal"],
        style=bundle["style"],
        motion=bundle["motion"],
        stability=bundle["stability"],
        cumulative_degradation=bundle["cumulative_degradation"],
        drift=bundle["drift"],
        delta_n=delta_n,
        delta_d=delta_d,
        ds_dt=ds_dt,
        aspect_variance=feats["aspect_variance"],
        agreement_score=feats["agreement_score"],
        quality_vs_trajectory_conflict=feats["quality_vs_trajectory_conflict"],
        temporal_identity_conflict=feats["temporal_identity_conflict"],
        style_motion_conflict=feats["style_motion_conflict"],
        stability_quality_gap=feats["stability_quality_gap"],
        false_beauty_score=feats["false_beauty_score"],
        degradation_pressure=feats["degradation_pressure"],
        structural_fragility=feats["structural_fragility"],
        predicted_final_risk=predicted_final_risk,
        predicted_final_quality=predicted_final_quality,
    )

    return asdict(snapshot)


def batch_compute_dn_metrics(items: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    return [compute_dn_metrics(x) for x in items]


def summarize_dn_metrics(items: List[Dict[str, Any]]) -> Dict[str, float]:
    rows = batch_compute_dn_metrics(items)
    if not rows:
        return {
            "count": 0,
            "delta_n_mean": 0.0,
            "delta_d_mean": 0.0,
            "ds_dt_mean": 0.0,
            "predicted_final_risk_mean": 0.0,
            "predicted_final_quality_mean": 0.0,
            "false_beauty_score_mean": 0.0,
            "degradation_pressure_mean": 0.0,
            "structural_fragility_mean": 0.0,
        }

    return {
        "count": len(rows),
        "delta_n_mean": _mean([r["delta_n"] for r in rows]),
        "delta_d_mean": _mean([r["delta_d"] for r in rows]),
        "ds_dt_mean": _mean([r["ds_dt"] for r in rows]),
        "predicted_final_risk_mean": _mean([r["predicted_final_risk"] for r in rows]),
        "predicted_final_quality_mean": _mean([r["predicted_final_quality"] for r in rows]),
        "false_beauty_score_mean": _mean([r["false_beauty_score"] for r in rows]),
        "degradation_pressure_mean": _mean([r["degradation_pressure"] for r in rows]),
        "structural_fragility_mean": _mean([r["structural_fragility"] for r in rows]),
    }