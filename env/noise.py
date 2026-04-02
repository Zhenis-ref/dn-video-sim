from __future__ import annotations

import numpy as np


class AR1Noise:
    def __init__(
        self,
        dim: int,
        sigma: float,
        ar_lambda: float,
        rng: np.random.Generator,
        spike_prob: float = 0.0,
        spike_scale: float = 3.0,
        clip_scale: float = 6.0,
    ) -> None:
        self.dim = int(dim)
        self.sigma = float(sigma)
        self.base_sigma = float(sigma)
        self.ar_lambda = float(ar_lambda)
        self.rng = rng

        self.spike_prob = float(spike_prob)
        self.spike_scale = float(spike_scale)
        self.clip_scale = float(clip_scale)

        self.state = np.zeros(self.dim, dtype=float)

    def reset(self) -> None:
        self.state = np.zeros(self.dim, dtype=float)

    def set_sigma(self, sigma: float) -> None:
        self.sigma = float(max(0.0, sigma))

    def sample(self) -> np.ndarray:
        """
        Цветной шум с памятью + редкие всплески.
        """
        sigma = max(self.sigma, 0.0)

        if sigma <= 0.0:
            self.state = self.ar_lambda * self.state
            return self.state.copy()

        # Белый базовый шум
        white = self.rng.normal(0.0, sigma, size=self.dim)

        # Редкие импульсные всплески
        if self.spike_prob > 0.0 and self.rng.random() < self.spike_prob:
            spike = self.rng.normal(0.0, sigma * self.spike_scale, size=self.dim)
        else:
            spike = 0.0

        innovation = white + spike

        # AR(1)-обновление с сохранением разумной дисперсии
        damp = np.sqrt(max(1.0 - self.ar_lambda**2, 0.0))
        self.state = self.ar_lambda * self.state + damp * innovation

        # Ограничение на экстремальные выбросы
        clip_bound = max(self.clip_scale * max(self.base_sigma, sigma), 1e-8)
        self.state = np.clip(self.state, -clip_bound, clip_bound)

        return self.state.copy()


class MixedNoise:
    """
    Запасной генератор для будущих стресс-тестов:
    смесь AR(1)-шума и редкого burst-компонента.
    """

    def __init__(
        self,
        dim: int,
        sigma_base: float,
        sigma_burst: float,
        ar_lambda: float,
        burst_prob: float,
        rng: np.random.Generator,
    ) -> None:
        self.dim = int(dim)
        self.sigma_base = float(sigma_base)
        self.sigma_burst = float(sigma_burst)
        self.ar_lambda = float(ar_lambda)
        self.burst_prob = float(burst_prob)
        self.rng = rng

        self.state = np.zeros(self.dim, dtype=float)

    def reset(self) -> None:
        self.state = np.zeros(self.dim, dtype=float)

    def sample(self) -> np.ndarray:
        white = self.rng.normal(0.0, self.sigma_base, size=self.dim)

        if self.rng.random() < self.burst_prob:
            burst = self.rng.normal(0.0, self.sigma_burst, size=self.dim)
        else:
            burst = 0.0

        innovation = white + burst
        damp = np.sqrt(max(1.0 - self.ar_lambda**2, 0.0))
        self.state = self.ar_lambda * self.state + damp * innovation
        return self.state.copy()