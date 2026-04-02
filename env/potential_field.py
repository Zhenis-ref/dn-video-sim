from __future__ import annotations

import numpy as np


class PotentialField:
    def __init__(
        self,
        latent_dim: int,
        num_wells: int,
        well_depth_min: float,
        well_depth_max: float,
        well_width_min: float,
        well_width_max: float,
        ruggedness: float,
        drift_scale: float,
        velocity_scale: float,
        rng: np.random.Generator,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.num_wells = int(num_wells)
        self.ruggedness = float(ruggedness)
        self.drift_scale = float(drift_scale)
        self.velocity_scale = float(velocity_scale)
        self.rng = rng

        # -----------------------------
        # Основные "хорошие" колодцы
        # -----------------------------
        self.centers = rng.normal(0.0, 1.0, size=(self.num_wells, self.latent_dim))
        self.depths = rng.uniform(well_depth_min, well_depth_max, size=self.num_wells)
        self.widths = rng.uniform(well_width_min, well_width_max, size=self.num_wells)

        # -----------------------------
        # Ложные колодцы / обманки
        # -----------------------------
        false_wells = max(1, self.num_wells // 2)
        self.false_centers = rng.normal(0.0, 1.3, size=(false_wells, self.latent_dim))
        self.false_depths = rng.uniform(
            0.25 * well_depth_min,
            0.70 * well_depth_max,
            size=false_wells,
        )
        self.false_widths = rng.uniform(
            0.6 * well_width_min,
            1.2 * well_width_max,
            size=false_wells,
        )

        # -----------------------------
        # Рыхлая "шероховатость" рельефа
        # -----------------------------
        num_rugged = 4
        self.rugged_vectors = rng.normal(0.0, 1.0, size=(num_rugged, self.latent_dim))
        self.rugged_phases = rng.uniform(0.0, 2.0 * np.pi, size=num_rugged)
        self.rugged_scales = rng.uniform(0.6, 1.4, size=num_rugged)

        # -----------------------------
        # Барьерные направления
        # -----------------------------
        num_barriers = 2
        self.barrier_vectors = rng.normal(0.0, 1.0, size=(num_barriers, self.latent_dim))
        self.barrier_vectors /= np.linalg.norm(self.barrier_vectors, axis=1, keepdims=True) + 1e-8
        self.barrier_scales = rng.uniform(0.4, 1.0, size=num_barriers)
        self.barrier_widths = rng.uniform(0.7, 1.6, size=num_barriers)

        # -----------------------------
        # Медленный дрейф рельефа
        # -----------------------------
        self.global_shift = np.zeros(self.latent_dim, dtype=float)
        self.shift_velocity = rng.normal(0.0, 0.02, size=self.latent_dim)
        self.deformation_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        self.t = 0

    def _current_shift(self) -> np.ndarray:
        """
        Медленная деформация ландшафта.
        """
        time_term = 0.15 * np.sin(0.01 * self.t + self.deformation_phase)
        return self.global_shift + time_term * self.shift_velocity

    def advance_time(self) -> None:
        """
        Можно вызывать один раз на шаг среды, если понадобится расширять динамику.
        Пока сделано безопасно: поле меняется медленно.
        """
        self.t += 1
        self.global_shift = 0.995 * self.global_shift + 0.005 * self.shift_velocity

    def potential(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        shift = self._current_shift()
        zz = z - shift

        total = 0.0

        # -----------------------------
        # Главные устойчивые attractor wells
        # Чем ниже potential, тем лучше "область"
        # Поэтому хорошие колодцы идут со знаком минус.
        # -----------------------------
        for c, a, w in zip(self.centers, self.depths, self.widths):
            delta = zz - c
            dist2 = np.sum(delta**2)
            total += -a * np.exp(-dist2 / (w**2 + 1e-8))

        # -----------------------------
        # Ложные колодцы
        # Они тоже выглядят привлекательно, но слабее и уже.
        # -----------------------------
        for c, a, w in zip(self.false_centers, self.false_depths, self.false_widths):
            delta = zz - c
            dist2 = np.sum(delta**2)
            total += -0.65 * a * np.exp(-dist2 / (w**2 + 1e-8))

        # -----------------------------
        # Рыхлая многомасштабная шероховатость
        # -----------------------------
        rugged = 0.0
        for vec, phase, scale in zip(self.rugged_vectors, self.rugged_phases, self.rugged_scales):
            rugged += np.sin(scale * np.dot(vec, zz) + phase)
            rugged += 0.5 * np.cos(0.7 * scale * np.dot(vec, zz) - phase)

        total += self.ruggedness * rugged

        # -----------------------------
        # Барьеры: создают сложность перехода между зонами
        # -----------------------------
        for vec, b_scale, b_width in zip(self.barrier_vectors, self.barrier_scales, self.barrier_widths):
            proj = np.dot(vec, zz)
            total += 0.35 * b_scale * np.exp(-(proj**2) / (b_width**2 + 1e-8))

        # -----------------------------
        # Слабый глобальный "контейнер", чтобы траектории не улетали бесконечно
        # -----------------------------
        radius2 = np.sum(zz**2)
        total += 0.03 * radius2 + 0.002 * (radius2**2)

        return float(total)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        shift = self._current_shift()
        zz = z - shift

        grad = np.zeros_like(zz)

        # -----------------------------
        # Градиент хороших колодцев
        # -----------------------------
        for c, a, w in zip(self.centers, self.depths, self.widths):
            delta = zz - c
            w2 = w**2 + 1e-8
            dist2 = np.sum(delta**2)
            coeff = -a * np.exp(-dist2 / w2)
            # d/dz [ -a exp(-dist2/w2) ] = coeff * (-2/w2) * delta * (-1)? Нет:
            # grad = (-a)*exp(...) * (-2/w2)*delta = (2a/w2)exp(...)*delta
            grad += coeff * (-2.0 / w2) * delta

        # -----------------------------
        # Градиент ложных колодцев
        # -----------------------------
        for c, a, w in zip(self.false_centers, self.false_depths, self.false_widths):
            delta = zz - c
            w2 = w**2 + 1e-8
            dist2 = np.sum(delta**2)
            coeff = -0.65 * a * np.exp(-dist2 / w2)
            grad += coeff * (-2.0 / w2) * delta

        # -----------------------------
        # Градиент шероховатости
        # -----------------------------
        for vec, phase, scale in zip(self.rugged_vectors, self.rugged_phases, self.rugged_scales):
            arg1 = scale * np.dot(vec, zz) + phase
            arg2 = 0.7 * scale * np.dot(vec, zz) - phase

            grad += self.ruggedness * np.cos(arg1) * scale * vec
            grad += self.ruggedness * 0.5 * (-np.sin(arg2)) * 0.7 * scale * vec

        # -----------------------------
        # Градиент барьеров
        # -----------------------------
        for vec, b_scale, b_width in zip(self.barrier_vectors, self.barrier_scales, self.barrier_widths):
            bw2 = b_width**2 + 1e-8
            proj = np.dot(vec, zz)
            coeff = 0.35 * b_scale * np.exp(-(proj**2) / bw2)
            grad += coeff * (-2.0 * proj / bw2) * vec

        # -----------------------------
        # Глобальный контейнер
        # d/dz [0.03 r^2 + 0.002 r^4] = 0.06 z + 0.008 r^2 z
        # -----------------------------
        radius2 = np.sum(zz**2)
        grad += 0.06 * zz + 0.008 * radius2 * zz

        return grad

    def drift(self, z: np.ndarray) -> np.ndarray:
        """
        Drift — более мягкий компонент движения, чем velocity_update.
        Он должен тянуть систему, но не превращать всё в тупой градиентный спуск.
        """
        g = self.gradient(z)
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-8:
            g_unit = g / g_norm
        else:
            g_unit = g

        # Мягкий градиентный дрейф + слабая поперечная компонента
        # чтобы траектории не были слишком прямолинейными.
        if self.latent_dim >= 2:
            perp = np.roll(g_unit, 1)
        else:
            perp = g_unit.copy()

        drift = -self.drift_scale * (0.82 * g + 0.18 * perp)
        return drift

    def velocity_update(self, z: np.ndarray) -> np.ndarray:
        """
        Более "инерционная" часть.
        """
        g = self.gradient(z)
        g_norm = np.linalg.norm(g)

        if g_norm > 1e-8:
            g_unit = g / g_norm
        else:
            g_unit = g

        # На сильном градиенте движение должно быть более жёстким,
        # но всё же ограниченным, чтобы не было дикого разлёта.
        scale = np.tanh(0.6 * g_norm)
        return -self.velocity_scale * scale * (0.75 * g + 0.25 * g_unit)