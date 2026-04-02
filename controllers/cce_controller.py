import numpy as np
from core.dn_metrics import frontier_delta_n, frontier_delta_d
from core.dn_engine import dS_dt, branching_pressure, duality_control
from controllers.frontier_ops import select_top_k

class CCEController:
    def __init__(self, cfg, env, rng):
        self.cfg = cfg
        self.env = env
        self.rng = rng
        self.next_id = 2000
        self.last_sigma = float(cfg["noise"]["sigma_z"])

    def step(self, frontier):
        model = self.cfg["model"]
        control = self.cfg["control"]
        weights = self.cfg["metrics"]
        
        dn = frontier_delta_n(frontier, weights)
        dd = frontier_delta_d(frontier, weights)
        u_d, d_opt = duality_control(dn, dd, control)
        pressure = branching_pressure(dn, dd, model)

        # 1. Адаптивный шум (управление 'температурой')
        base_sigma = float(self.cfg["noise"]["sigma_z"])
        target_sigma = base_sigma * np.exp(1.2 * u_d)
        self.last_sigma = 0.7 * self.last_sigma + 0.3 * target_sigma
        
        # 2. Экономичное ветвление
        # Если u_d < 0.1 (все в порядке), делаем 1 копию. Baseline всегда делает 2.
        # В этом месте мы получаем преимущество в 2 раза по генерации.
        copies = 1 if u_d < 0.1 else 2
        if pressure > 0.85: copies = 3 # Только при сильном стрессе расширяемся

        expanded = []
        for node in frontier:
            for _ in range(copies):
                child = self.env.step(node, sigma_override=self.last_sigma)
                child.quality = self.env.quality_score(child, weights)
                child.candidate_id = self.next_id
                self.next_id += 1
                expanded.append(child)

        # 3. Динамическое сжатие фронта (Pruning)
        k_limit = int(self.cfg["frontier"]["max_candidates_cce"])
        # Если ошибок (dn) мало, работаем очень узко (минимальный фронт + запас)
        target_k = k_limit if dn > 0.2 else int(self.cfg["frontier"]["min_candidates_cce"] + 2)
        
        scores = [s.quality for s in expanded]
        selected = select_top_k(expanded, scores, target_k)

        return selected, {"generated": len(expanded)}