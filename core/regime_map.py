def classify_regime(delta_n: float, delta_d: float, ncrit: float) -> str:
    if delta_d < 0.15 and delta_n < ncrit:
        return "stasis"

    if delta_d > 1.25 and delta_n >= ncrit:
        return "collapse"

    if delta_d > 1.0 and delta_n < ncrit:
        return "false_diversity"

    if delta_d < 0.20 and delta_n >= ncrit:
        return "fragile_lock"

    return "working_corridor"