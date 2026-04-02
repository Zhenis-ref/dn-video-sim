from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _clone_meta(meta: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            cloned[k] = v.copy()
        elif isinstance(v, list):
            cloned[k] = list(v)
        elif isinstance(v, dict):
            cloned[k] = dict(v)
        else:
            cloned[k] = v
    return cloned


@dataclass
class CandidateState:
    candidate_id: int
    z: np.ndarray
    v: np.ndarray

    prompt_align: float
    identity_consistency: float
    temporal_consistency: float
    motion_coherence: float
    style_stability: float

    age: int = 0
    parent_id: int | None = None
    alive: bool = True

    # --- агрегированные метрики текущего состояния ---
    quality: float = 0.0
    degradation_score: float = 0.0
    instability_score: float = 0.0

    # --- история по траектории ---
    quality_trace: list[float] = field(default_factory=list)
    identity_trace: list[float] = field(default_factory=list)
    temporal_trace: list[float] = field(default_factory=list)
    motion_trace: list[float] = field(default_factory=list)
    style_trace: list[float] = field(default_factory=list)

    # --- вычислительная цена и происхождение ---
    generation_depth: int = 0
    branch_cost: int = 0
    eval_count: int = 0
    expanded_count: int = 0
    selected_count: int = 0

    # --- произвольные дополнительные данные ---
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_degraded(self) -> bool:
        return self.quality < 0.55

    @property
    def is_unstable(self) -> bool:
        return (
            self.temporal_consistency < 0.50
            or self.identity_consistency < 0.50
            or self.style_stability < 0.45
        )

    @property
    def best_quality_seen(self) -> float:
        if not self.quality_trace:
            return float(self.quality)
        return float(max(self.quality_trace))

    @property
    def worst_quality_seen(self) -> float:
        if not self.quality_trace:
            return float(self.quality)
        return float(min(self.quality_trace))

    @property
    def quality_drop_from_best(self) -> float:
        return float(max(0.0, self.best_quality_seen - self.quality))

    @property
    def temporal_jitter(self) -> float:
        if len(self.temporal_trace) < 2:
            return 0.0
        diffs = np.diff(np.asarray(self.temporal_trace, dtype=float))
        return float(np.std(diffs))

    @property
    def identity_jitter(self) -> float:
        if len(self.identity_trace) < 2:
            return 0.0
        diffs = np.diff(np.asarray(self.identity_trace, dtype=float))
        return float(np.std(diffs))

    def update_derived_metrics(self) -> None:
        """
        Пересчёт агрегированных метрик из текущих прокси.
        Вызывается после обновления quality/proxy-полей.
        """
        self.degradation_score = float(
            np.clip(
                0.50 * (1.0 - self.identity_consistency)
                + 0.30 * (1.0 - self.style_stability)
                + 0.20 * max(0.0, self.quality_drop_from_best),
                0.0,
                1.0,
            )
        )

        self.instability_score = float(
            np.clip(
                0.45 * (1.0 - self.temporal_consistency)
                + 0.25 * (1.0 - self.motion_coherence)
                + 0.15 * self.temporal_jitter
                + 0.15 * self.identity_jitter,
                0.0,
                1.0,
            )
        )

    def push_trace(self) -> None:
        """
        Сохраняет текущие прокси в историю.
        Вызывается после пересчёта качества.
        """
        self.quality_trace.append(float(self.quality))
        self.identity_trace.append(float(self.identity_consistency))
        self.temporal_trace.append(float(self.temporal_consistency))
        self.motion_trace.append(float(self.motion_coherence))
        self.style_trace.append(float(self.style_stability))

        # защитимся от бесконечного разрастания истории
        max_len = 64
        if len(self.quality_trace) > max_len:
            self.quality_trace = self.quality_trace[-max_len:]
            self.identity_trace = self.identity_trace[-max_len:]
            self.temporal_trace = self.temporal_trace[-max_len:]
            self.motion_trace = self.motion_trace[-max_len:]
            self.style_trace = self.style_trace[-max_len:]

    def register_evaluation(self, count: int = 1) -> None:
        self.eval_count += int(count)

    def register_expansion(self, count: int = 1) -> None:
        self.expanded_count += int(count)

    def register_selection(self, count: int = 1) -> None:
        self.selected_count += int(count)

    def add_branch_cost(self, amount: int = 1) -> None:
        self.branch_cost += int(amount)

    def copy(self) -> "CandidateState":
        return CandidateState(
            candidate_id=int(self.candidate_id),
            z=self.z.copy(),
            v=self.v.copy(),
            prompt_align=float(self.prompt_align),
            identity_consistency=float(self.identity_consistency),
            temporal_consistency=float(self.temporal_consistency),
            motion_coherence=float(self.motion_coherence),
            style_stability=float(self.style_stability),
            age=int(self.age),
            parent_id=self.parent_id,
            alive=bool(self.alive),
            quality=float(self.quality),
            degradation_score=float(self.degradation_score),
            instability_score=float(self.instability_score),
            quality_trace=list(self.quality_trace),
            identity_trace=list(self.identity_trace),
            temporal_trace=list(self.temporal_trace),
            motion_trace=list(self.motion_trace),
            style_trace=list(self.style_trace),
            generation_depth=int(self.generation_depth),
            branch_cost=int(self.branch_cost),
            eval_count=int(self.eval_count),
            expanded_count=int(self.expanded_count),
            selected_count=int(self.selected_count),
            meta=_clone_meta(self.meta),
        )

    def spawn_child(self, new_candidate_id: int | None = None) -> "CandidateState":
        """
        Удобный способ породить дочернее состояние без потери трассировки.
        """
        child = self.copy()
        if new_candidate_id is not None:
            child.candidate_id = int(new_candidate_id)
        child.parent_id = int(self.candidate_id)
        child.age = int(self.age)
        child.generation_depth = int(self.generation_depth) + 1
        return child

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "parent_id": self.parent_id,
            "age": int(self.age),
            "alive": bool(self.alive),
            "quality": float(self.quality),
            "degradation_score": float(self.degradation_score),
            "instability_score": float(self.instability_score),
            "prompt_align": float(self.prompt_align),
            "identity_consistency": float(self.identity_consistency),
            "temporal_consistency": float(self.temporal_consistency),
            "motion_coherence": float(self.motion_coherence),
            "style_stability": float(self.style_stability),
            "generation_depth": int(self.generation_depth),
            "branch_cost": int(self.branch_cost),
            "eval_count": int(self.eval_count),
            "expanded_count": int(self.expanded_count),
            "selected_count": int(self.selected_count),
        }