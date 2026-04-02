from __future__ import annotations

import numpy as np

from env.state import CandidateState


def expand_frontier(
    frontier: list[CandidateState],
    env,
    next_id_start: int,
    copies_per_node: int = 2,
    sigma_override: float | None = None,
) -> tuple[list[CandidateState], int]:
    """
    Универсальное расширение frontier через текущий интерфейс env.step(...).

    Оставлено как вспомогательная функция совместимости.
    """
    new_frontier: list[CandidateState] = []
    next_id = int(next_id_start)

    if not frontier or copies_per_node <= 0:
        return new_frontier, next_id

    for state in frontier:
        for _ in range(int(copies_per_node)):
            child = env.step(state, sigma_override=sigma_override)
            child.candidate_id = next_id
            child.parent_id = state.candidate_id
            child.register_expansion(1)
            next_id += 1
            new_frontier.append(child)

    return new_frontier, next_id


def select_top_k(frontier: list[CandidateState], scores, k: int) -> list[CandidateState]:
    """
    Возвращает top-k по scores в порядке убывания score.
    """
    if not frontier or k <= 0:
        return []

    n = len(frontier)
    if n <= k:
        return list(frontier)

    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.shape[0] != n:
        raise ValueError("Length of scores must match length of frontier.")

    idx = np.argsort(scores_arr)[::-1][:k]
    return [frontier[int(i)] for i in idx]


def random_subset(frontier: list[CandidateState], k: int, rng) -> list[CandidateState]:
    if not frontier or k <= 0:
        return []

    if len(frontier) <= k:
        return list(frontier)

    idx = rng.choice(len(frontier), size=int(k), replace=False)
    return [frontier[int(i)] for i in idx]


def _latent_distance(a: CandidateState, b: CandidateState) -> float:
    return float(np.linalg.norm(np.asarray(a.z, dtype=float) - np.asarray(b.z, dtype=float)))


def diversity_prune(
    frontier: list[CandidateState],
    k: int,
    scores: list[float] | np.ndarray | None = None,
) -> list[CandidateState]:
    """
    Farthest-first pruning без дубликатов.

    Идея:
    - первый элемент можно взять либо лучший по score,
      либо просто первый, если score не передан;
    - дальше каждый раз добавляем объект, который
      максимально далёк от уже выбранного множества.
    """
    if not frontier or k <= 0:
        return []

    n = len(frontier)
    if n <= k:
        return list(frontier)

    # 1. стартовая точка
    if scores is not None:
        scores_arr = np.asarray(scores, dtype=float)
        if scores_arr.shape[0] != n:
            raise ValueError("Length of scores must match length of frontier.")
        first_idx = int(np.argmax(scores_arr))
    else:
        first_idx = 0

    selected_indices = [first_idx]
    remaining = set(range(n))
    remaining.remove(first_idx)

    # 2. farthest-first traversal
    while len(selected_indices) < k and remaining:
        best_idx = None
        best_min_dist = -1.0

        for idx in remaining:
            candidate = frontier[idx]
            min_dist_to_selected = min(
                _latent_distance(candidate, frontier[s_idx])
                for s_idx in selected_indices
            )

            if min_dist_to_selected > best_min_dist:
                best_min_dist = min_dist_to_selected
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [frontier[i] for i in selected_indices]