import numpy as np

def alpha(delta_n: float, delta_d: float, params: dict) -> float:
    """Каноническая чувствительность: реакция на давление и подавление хаоса."""
    A = float(params["A"])
    k = float(params["k"])
    ncrit = float(params["Ncrit"])
    p = float(params["p"])
    eps = float(params["eps"])

    # Логистический порог активации при росте ошибок (delta_n)
    logistic = 1.0 + np.exp(-k * (delta_n - ncrit))
    # Подавление активности при избыточном разбросе (delta_d)
    suppression = (delta_d ** p) + eps
    return float(A / (logistic * suppression))

def dS_dt(delta_n: float, delta_d: float, params: dict) -> float:
    """Темп изменений: баланс между исправлением ошибок и поиском нового."""
    a = alpha(delta_n, delta_d, params)
    beta = float(params["beta"])
    # Основное уравнение DN-динамики
    return float(a * delta_n + beta * delta_d)

def branching_pressure(delta_n: float, delta_d: float, params: dict) -> float:
    """Сигнал на расширение дерева (от 0 до 1)."""
    raw = dS_dt(delta_n, delta_d, params)
    return float(np.tanh(raw)) if raw > 0 else 0.0

def delta_d_opt(delta_n: float, control_params: dict) -> float:
    """Целевая сложность: чем выше риск (delta_n), тем уже должен быть поиск."""
    d_min = float(control_params["D_min"])
    d_max = float(control_params["D_max"])
    k_dual = float(control_params["k_dual"])
    return float(d_min + (d_max / (1.0 + k_dual * delta_n)))

def duality_control(delta_n: float, delta_d: float, control_params: dict) -> tuple[float, float]:
    """Ошибка управления: разница между тем, что есть, и тем, что нужно."""
    gamma = float(control_params["gamma"])
    d_opt = delta_d_opt(delta_n, control_params)
    u_d = gamma * (d_opt - delta_d)
    return float(u_d), float(d_opt)