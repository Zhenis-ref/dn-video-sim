from __future__ import annotations

from controllers.frontier_ops import select_top_k


class BaselineController:
    def __init__(self, cfg, env, rng):
        self.cfg = cfg
        self.env = env
        self.rng = rng
        self.next_id = 1000

    def step(self, frontier):
        """
        Честный baseline:
        - одинаковое число копий на узел,
        - без ΔN/ΔD-регуляции,
        - без адаптивного шумоподавления,
        - отбор только по качеству.

        Это важно: baseline должен быть вменяемым, а не заведомо слабым.
        """
        weights = self.cfg["metrics"]
        frontier_cfg = self.cfg["frontier"]

        k_max = int(frontier_cfg["max_candidates_baseline"])

        # Базовый поиск должен быть шире и тупее, чем CCE,
        # но не бессмысленно взрывным.
        # Если в конфиге нет отдельного параметра, используем 2 копии как safe default.
        copies = int(frontier_cfg.get("baseline_copies", 2))
        copies = max(1, copies)

        expanded = []
        generated = 0

        for node in frontier:
            for _ in range(copies):
                new_state = self.env.step(node)
                new_state.parent_id = node.candidate_id
                new_state.candidate_id = self.next_id
                self.next_id += 1

                new_state.register_expansion(1)
                new_state.register_selection(0)

                new_state.quality = float(self.env.quality_score(new_state, weights))
                new_state.update_derived_metrics()

                expanded.append(new_state)
                generated += 1

        if not expanded:
            return frontier, {
                "generated": 0,
                "selected": len(frontier),
                "copies": copies,
                "target_k": len(frontier),
            }

        scores = []
        for s in expanded:
            q = float(self.env.quality_score(s, weights))
            s.quality = q
            s.update_derived_metrics()

            # Baseline intentionally simpler than CCE:
            # он смотрит в первую очередь на качество,
            # и только слегка штрафует уже явную деградацию/нестабильность.
            score = (
                1.00 * q
                - 0.15 * float(getattr(s, "degradation_score", 0.0))
                - 0.10 * float(getattr(s, "instability_score", 0.0))
            )
            scores.append(float(score))

        selected = select_top_k(expanded, scores, k_max)

        if len(selected) == 0:
            fallback_k = min(max(1, k_max), len(expanded))
            selected = select_top_k(expanded, scores, fallback_k)

        for s in selected:
            s.register_selection(1)
            s.quality = float(self.env.quality_score(s, weights))
            s.update_derived_metrics()

        return selected, {
            "generated": int(generated),
            "selected": int(len(selected)),
            "copies": int(copies),
            "target_k": int(k_max),
        }