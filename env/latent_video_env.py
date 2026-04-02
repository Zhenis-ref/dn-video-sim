from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
import math
import random


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class VideoState:
    step_idx: int

    quality: float
    stability: float

    identity: float
    temporal: float
    style: float
    motion: float

    quality_prev: float
    quality_best: float
    quality_target: float

    cumulative_degradation: float
    drift: float
    regime: str


@dataclass
class Action:
    name: str

    prompt_gain: float
    identity_gain: float
    temporal_gain: float
    style_gain: float
    motion_gain: float

    instability: float
    late_penalty_bias: float
    lock_in_bias: float


# ============================================================
# HELPERS
# ============================================================

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


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
# MAIN ENVIRONMENT
# ============================================================

class LatentVideoEnv:
    """
    Диагностическая среда для DN-video-sim.

    Идея:
    - качество не задаётся напрямую одной кнопкой;
    - итог зависит от траектории;
    - есть ранние ложные улучшения;
    - есть поздние penalty за накопленный drift/degradation;
    - все 5 аспектов реально участвуют.
    """

    def __init__(self, seed: int = 42, horizon: int = 18):
        self.seed = seed
        self.rng = random.Random(seed)
        self.horizon = horizon

        self.state: VideoState | None = None

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> Dict[str, float]:
        identity0 = self.rng.uniform(0.46, 0.62)
        temporal0 = self.rng.uniform(0.42, 0.60)
        style0 = self.rng.uniform(0.50, 0.68)
        motion0 = self.rng.uniform(0.40, 0.60)
        stability0 = self.rng.uniform(0.48, 0.64)

        quality0 = self._compose_quality(
            identity=identity0,
            temporal=temporal0,
            style=style0,
            motion=motion0,
            stability=stability0,
            cumulative_degradation=0.0,
            drift=0.0,
            step_idx=0,
        )

        self.state = VideoState(
            step_idx=0,
            quality=quality0,
            stability=stability0,
            identity=identity0,
            temporal=temporal0,
            style=style0,
            motion=motion0,
            quality_prev=quality0,
            quality_best=quality0,
            quality_target=1.0,
            cumulative_degradation=0.0,
            drift=0.0,
            regime="warmup",
        )
        return self._get_info()

    # ========================================================
    # ACTION SPACE
    # ========================================================

    def sample_action_candidates(self) -> List[Action]:
        s = self._require_state()

        phase = s.step_idx / max(1, self.horizon - 1)

        # В начале больше "соблазнительных" prompt/style действий.
        # Позже больше выигрывают temporal/identity/stability действия.
        candidates = [
            Action(
                name="prompt_push",
                prompt_gain=0.11 + 0.02 * (1.0 - phase),
                identity_gain=0.01,
                temporal_gain=-0.01,
                style_gain=0.03,
                motion_gain=0.00,
                instability=0.07,
                late_penalty_bias=0.08,
                lock_in_bias=0.06,
            ),
            Action(
                name="identity_lock",
                prompt_gain=0.01,
                identity_gain=0.08,
                temporal_gain=0.01,
                style_gain=-0.01,
                motion_gain=0.00,
                instability=0.02,
                late_penalty_bias=0.01,
                lock_in_bias=0.00,
            ),
            Action(
                name="temporal_align",
                prompt_gain=0.00,
                identity_gain=0.01,
                temporal_gain=0.09,
                style_gain=0.00,
                motion_gain=0.02,
                instability=0.02,
                late_penalty_bias=0.00,
                lock_in_bias=0.00,
            ),
            Action(
                name="style_boost",
                prompt_gain=0.03,
                identity_gain=-0.01,
                temporal_gain=-0.01,
                style_gain=0.10,
                motion_gain=0.00,
                instability=0.05,
                late_penalty_bias=0.05,
                lock_in_bias=0.03,
            ),
            Action(
                name="motion_boost",
                prompt_gain=0.01,
                identity_gain=0.00,
                temporal_gain=0.02,
                style_gain=-0.01,
                motion_gain=0.09,
                instability=0.05,
                late_penalty_bias=0.04,
                lock_in_bias=0.02,
            ),
            Action(
                name="consistency_repair",
                prompt_gain=-0.01,
                identity_gain=0.03,
                temporal_gain=0.04,
                style_gain=0.02,
                motion_gain=0.03,
                instability=-0.03,
                late_penalty_bias=-0.02,
                lock_in_bias=-0.01,
            ),
            Action(
                name="aggressive_refine",
                prompt_gain=0.08,
                identity_gain=0.02,
                temporal_gain=0.01,
                style_gain=0.03,
                motion_gain=0.01,
                instability=0.09,
                late_penalty_bias=0.10,
                lock_in_bias=0.07,
            ),
            Action(
                name="trajectory_repair",
                prompt_gain=-0.02,
                identity_gain=0.02,
                temporal_gain=0.07,
                style_gain=0.00,
                motion_gain=0.05,
                instability=-0.01,
                late_penalty_bias=-0.01,
                lock_in_bias=0.00,
            ),
        ]

        # Небольшая вариативность внутри одного seed.
        jittered: List[Action] = []
        for a in candidates:
            jittered.append(
                Action(
                    name=a.name,
                    prompt_gain=a.prompt_gain + self.rng.uniform(-0.01, 0.01),
                    identity_gain=a.identity_gain + self.rng.uniform(-0.01, 0.01),
                    temporal_gain=a.temporal_gain + self.rng.uniform(-0.01, 0.01),
                    style_gain=a.style_gain + self.rng.uniform(-0.01, 0.01),
                    motion_gain=a.motion_gain + self.rng.uniform(-0.01, 0.01),
                    instability=a.instability + self.rng.uniform(-0.01, 0.01),
                    late_penalty_bias=a.late_penalty_bias + self.rng.uniform(-0.01, 0.01),
                    lock_in_bias=a.lock_in_bias + self.rng.uniform(-0.01, 0.01),
                )
            )

        return jittered

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action: Action) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        s = self._require_state()

        prev_quality = s.quality
        t = s.step_idx + 1
        phase = t / max(1, self.horizon - 1)

        # ---- обновление аспектов ----
        identity = _clip(
            s.identity
            + action.identity_gain
            - 0.020 * max(0.0, s.drift - 0.35)
            + self.rng.uniform(-0.01, 0.01)
        )

        temporal = _clip(
            s.temporal
            + action.temporal_gain
            - 0.030 * max(0.0, s.cumulative_degradation - 0.30)
            + self.rng.uniform(-0.01, 0.01)
        )

        style = _clip(
            s.style
            + action.style_gain
            - 0.015 * max(0.0, s.drift - 0.40)
            + self.rng.uniform(-0.01, 0.01)
        )

        motion = _clip(
            s.motion
            + action.motion_gain
            - 0.025 * max(0.0, s.cumulative_degradation - 0.25)
            + self.rng.uniform(-0.01, 0.01)
        )

        # stability не просто растёт/падает линейно
        aspect_conflict = (
            abs(identity - temporal)
            + abs(style - motion)
            + abs(identity - motion) * 0.5
        ) / 2.5

        stability = _clip(
            s.stability
            + 0.045
            - 0.55 * max(0.0, action.instability)
            - 0.25 * aspect_conflict
            - 0.10 * max(0.0, phase - 0.55) * action.late_penalty_bias
            + self.rng.uniform(-0.01, 0.01)
        )

        # ---- накопительная деградация ----
        cumulative_degradation = _clip(
            s.cumulative_degradation
            + 0.040 * max(0.0, action.instability)
            + 0.025 * aspect_conflict
            + 0.040 * max(0.0, phase - 0.50) * max(0.0, action.late_penalty_bias)
            - 0.035 * (1.0 if action.name in {"consistency_repair", "trajectory_repair"} else 0.0),
            0.0,
            1.0,
        )

        # ---- drift / lock-in ----
        drift = _clip(
            s.drift
            + 0.030 * action.prompt_gain
            + 0.035 * max(0.0, action.lock_in_bias)
            + 0.020 * max(0.0, phase - 0.45)
            - 0.030 * (1.0 if action.name in {"identity_lock", "temporal_align", "trajectory_repair"} else 0.0),
            0.0,
            1.0,
        )

        quality = self._compose_quality(
            identity=identity,
            temporal=temporal,
            style=style,
            motion=motion,
            stability=stability,
            cumulative_degradation=cumulative_degradation,
            drift=drift,
            step_idx=t,
        )

        # ---- важная нелинейность ----
        # Ложный ранний буст: prompt/style действия вначале выглядят хорошо
        if t <= 4 and action.name in {"prompt_push", "style_boost", "aggressive_refine"}:
            quality = _clip(quality + 0.035)

        # Поздний распад: если долго кормили "красоту", потом бьёт по качеству
        if t >= 8:
            quality = _clip(
                quality
                - 0.060 * max(0.0, cumulative_degradation - 0.28)
                - 0.050 * max(0.0, drift - 0.30)
            )

        quality_best = max(s.quality_best, quality)

        regime = self._classify_regime(
            quality=quality,
            stability=stability,
            drift=drift,
            cumulative_degradation=cumulative_degradation,
            identity=identity,
            temporal=temporal,
            style=style,
            motion=motion,
        )

        self.state = VideoState(
            step_idx=t,
            quality=quality,
            stability=stability,
            identity=identity,
            temporal=temporal,
            style=style,
            motion=motion,
            quality_prev=prev_quality,
            quality_best=quality_best,
            quality_target=1.0,
            cumulative_degradation=cumulative_degradation,
            drift=drift,
            regime=regime,
        )

        done = (t >= self.horizon - 1)
        reward = quality - prev_quality

        info = self._get_info()
        return info, reward, done, info

    # ========================================================
    # INTERNAL QUALITY MODEL
    # ========================================================

    def _compose_quality(
        self,
        identity: float,
        temporal: float,
        style: float,
        motion: float,
        stability: float,
        cumulative_degradation: float,
        drift: float,
        step_idx: int,
    ) -> float:
        """
        Здесь качество реально траекторное.
        Оно зависит:
        - от всех 5 аспектов,
        - от их согласованности,
        - от поздних штрафов за накопленный drift/degradation.
        """

        base = (
            0.24 * identity
            + 0.22 * temporal
            + 0.16 * style
            + 0.16 * motion
            + 0.22 * stability
        )

        agreement = 1.0 - (
            abs(identity - temporal)
            + abs(style - motion)
            + abs(identity - motion) * 0.5
        ) / 2.5
        agreement = _clip(agreement)

        # ранняя "красота" допустима, но не должна гарантировать победу
        early_bonus = 0.02 * math.exp(-0.35 * step_idx)

        # поздний штраф — то, чего не хватало в игрушечной среде
        late_penalty = 0.0
        if step_idx >= 7:
            late_penalty = (
                0.18 * cumulative_degradation
                + 0.15 * drift
            )

        quality = (
            0.72 * base
            + 0.22 * agreement
            + early_bonus
            - late_penalty
        )
        return _clip(quality)

    # ========================================================
    # REGIME LABELS
    # ========================================================

    def _classify_regime(
        self,
        quality: float,
        stability: float,
        drift: float,
        cumulative_degradation: float,
        identity: float,
        temporal: float,
        style: float,
        motion: float,
    ) -> str:
        conflict = _variance([identity, temporal, style, motion, stability])

        if quality < 0.50 and stability < 0.55:
            return "collapse"

        if drift > 0.45 and cumulative_degradation > 0.40:
            return "delayed_breakdown"

        if quality > 0.72 and conflict > 0.020 and stability < 0.68:
            return "false_beauty"

        if quality > 0.74 and stability > 0.74 and conflict < 0.010:
            return "working_corridor"

        if stability > 0.75 and quality < 0.64:
            return "overconstrained"

        return "transition"

    # ========================================================
    # INFO EXPORT
    # ========================================================

    def _get_info(self) -> Dict[str, float]:
        s = self._require_state()
        return {
            "step_idx": float(s.step_idx),
            "quality": float(s.quality),
            "stability": float(s.stability),
            "identity": float(s.identity),
            "temporal": float(s.temporal),
            "style": float(s.style),
            "motion": float(s.motion),
            "quality_prev": float(s.quality_prev),
            "quality_best": float(s.quality_best),
            "quality_target": float(s.quality_target),
            "cumulative_degradation": float(s.cumulative_degradation),
            "drift": float(s.drift),
            "regime_code": float(self._regime_to_code(s.regime)),
        }

    @staticmethod
    def _regime_to_code(name: str) -> int:
        table = {
            "warmup": 0,
            "transition": 1,
            "false_beauty": 2,
            "working_corridor": 3,
            "overconstrained": 4,
            "delayed_breakdown": 5,
            "collapse": 6,
        }
        return table.get(name, 1)

    def get_state_dict(self) -> Dict[str, Any]:
        return asdict(self._require_state())

    def _require_state(self) -> VideoState:
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self.state