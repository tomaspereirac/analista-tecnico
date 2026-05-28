"""
analysis.py — Detecciones de análisis técnico price-action.

Funciones puras que reciben un DataFrame OHLCV y devuelven estructuras
con la info necesaria para dibujar en cualquier librería gráfica.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ─────────────────────────── Estructuras ───────────────────────────
@dataclass
class SwingPoints:
    high_idx: np.ndarray
    high_prices: np.ndarray
    low_idx: np.ndarray
    low_prices: np.ndarray


@dataclass
class Level:
    price: float
    touches: int
    bar_indices: list


@dataclass
class Trendline:
    slope: float
    intercept: float
    start_idx: int
    point_indices: list


@dataclass
class Channel:
    upper: Trendline
    lower: Trendline


@dataclass
class FibLevels:
    swing_high_idx: int
    swing_low_idx: int
    high_price: float
    low_price: float
    direction: str  # "up" si el último move fue alcista, "down" si bajista
    levels: dict = field(default_factory=dict)  # {ratio: price}


@dataclass
class Pattern:
    name: str
    bar_indices: list
    prices: list
    description: str


# ─────────────────────────── Detecciones ───────────────────────────
def detect_swings(df: pd.DataFrame, distance: int = 5,
                  prominence_pct: float = 0.015) -> SwingPoints:
    """Detecta swing highs y lows con find_peaks."""
    highs = df['High'].values.astype(float)
    lows = df['Low'].values.astype(float)
    mean_price = float(df['Close'].mean())
    prom = mean_price * prominence_pct

    high_idx, _ = find_peaks(highs, distance=distance, prominence=prom)
    low_idx, _ = find_peaks(-lows, distance=distance, prominence=prom)

    return SwingPoints(
        high_idx=high_idx,
        high_prices=highs[high_idx] if len(high_idx) else np.array([]),
        low_idx=low_idx,
        low_prices=lows[low_idx] if len(low_idx) else np.array([]),
    )


def cluster_levels(swings: SwingPoints, tolerance_pct: float = 0.02,
                   min_touches: int = 2) -> list[Level]:
    """Agrupa swings cercanos en precio para encontrar S/R."""
    all_prices = np.concatenate([swings.high_prices, swings.low_prices])
    all_idx = np.concatenate([swings.high_idx, swings.low_idx])
    if len(all_prices) == 0:
        return []

    order = np.argsort(all_prices)
    sp = all_prices[order]
    si = all_idx[order]

    clusters = [[(sp[0], si[0])]]
    for p, i in zip(sp[1:], si[1:]):
        last = clusters[-1][-1][0]
        if abs(p - last) / last < tolerance_pct:
            clusters[-1].append((p, i))
        else:
            clusters.append([(p, i)])

    levels = []
    for c in clusters:
        if len(c) >= min_touches:
            prices = [x[0] for x in c]
            idxs = [int(x[1]) for x in c]
            levels.append(Level(
                price=float(np.mean(prices)),
                touches=len(c),
                bar_indices=idxs,
            ))
    return sorted(levels, key=lambda l: -l.touches)


def fit_trendline(indices: np.ndarray, prices: np.ndarray,
                  max_points: int = 3) -> Optional[Trendline]:
    """Ajusta una recta a los últimos N puntos."""
    if len(indices) < 2:
        return None
    n = min(max_points, len(indices))
    xs = indices[-n:].astype(float)
    ys = prices[-n:]
    slope, intercept = np.polyfit(xs, ys, 1)
    return Trendline(
        slope=float(slope),
        intercept=float(intercept),
        start_idx=int(xs[0]),
        point_indices=[int(x) for x in xs],
    )


def detect_channel(swings: SwingPoints, n_points: int = 3,
                   slope_tolerance: float = 0.5) -> Optional[Channel]:
    """Canal: dos trendlines paralelas (pendientes similares)."""
    if len(swings.low_idx) < 2 or len(swings.high_idx) < 2:
        return None
    lower = fit_trendline(swings.low_idx, swings.low_prices, n_points)
    upper = fit_trendline(swings.high_idx, swings.high_prices, n_points)
    if lower is None or upper is None:
        return None
    avg = (abs(lower.slope) + abs(upper.slope)) / 2 + 1e-9
    if abs(lower.slope - upper.slope) / avg > slope_tolerance:
        return None
    return Channel(upper=upper, lower=lower)


def fibonacci_retracements(swings: SwingPoints) -> Optional[FibLevels]:
    """Fibo entre el swing high y swing low más recientes."""
    if len(swings.high_idx) == 0 or len(swings.low_idx) == 0:
        return None
    last_h_i = int(swings.high_idx[-1])
    last_l_i = int(swings.low_idx[-1])
    last_h_p = float(swings.high_prices[-1])
    last_l_p = float(swings.low_prices[-1])

    direction = "up" if last_h_i > last_l_i else "down"
    diff = last_h_p - last_l_p
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    # Si el move fue alcista, los retrocesos cuentan desde el high hacia abajo.
    # Si fue bajista, desde el low hacia arriba.
    if direction == "up":
        levels = {r: last_h_p - diff * r for r in ratios}
    else:
        levels = {r: last_l_p + diff * r for r in ratios}

    return FibLevels(
        swing_high_idx=last_h_i,
        swing_low_idx=last_l_i,
        high_price=last_h_p,
        low_price=last_l_p,
        direction=direction,
        levels=levels,
    )


def detect_double_top(swings: SwingPoints,
                      tolerance_pct: float = 0.025) -> list[Pattern]:
    out = []
    for i in range(len(swings.high_idx) - 1):
        p1, p2 = swings.high_prices[i], swings.high_prices[i + 1]
        if abs(p1 - p2) / p1 < tolerance_pct:
            out.append(Pattern(
                name="Doble Techo",
                bar_indices=[int(swings.high_idx[i]), int(swings.high_idx[i + 1])],
                prices=[float(p1), float(p2)],
                description=f"Dos máximos en ~${(p1+p2)/2:.2f}. "
                           f"Sesgo bajista si rompe el mínimo entre ambos.",
            ))
    return out


def detect_double_bottom(swings: SwingPoints,
                         tolerance_pct: float = 0.025) -> list[Pattern]:
    out = []
    for i in range(len(swings.low_idx) - 1):
        p1, p2 = swings.low_prices[i], swings.low_prices[i + 1]
        if abs(p1 - p2) / p1 < tolerance_pct:
            out.append(Pattern(
                name="Doble Piso",
                bar_indices=[int(swings.low_idx[i]), int(swings.low_idx[i + 1])],
                prices=[float(p1), float(p2)],
                description=f"Dos mínimos en ~${(p1+p2)/2:.2f}. "
                           f"Sesgo alcista si rompe el máximo entre ambos.",
            ))
    return out


def detect_triangle(swings: SwingPoints) -> Optional[Pattern]:
    """Triángulo según convergencia de las trendlines."""
    if len(swings.high_idx) < 2 or len(swings.low_idx) < 2:
        return None
    upper = fit_trendline(swings.high_idx, swings.high_prices)
    lower = fit_trendline(swings.low_idx, swings.low_prices)
    if upper is None or lower is None:
        return None

    # Normalizamos slope respecto al precio promedio para tener un % por barra
    avg_price = (upper.intercept + lower.intercept) / 2
    up_norm = upper.slope / avg_price if avg_price else 0
    lo_norm = lower.slope / avg_price if avg_price else 0
    FLAT = 5e-4   # ~0.05% por barra: lo consideramos horizontal

    if up_norm < -FLAT and lo_norm > FLAT:
        name = "Triángulo Simétrico"
        desc = "Máximos descendentes + mínimos ascendentes. Ruptura define dirección."
    elif up_norm < -FLAT and abs(lo_norm) <= FLAT:
        name = "Triángulo Descendente"
        desc = "Máximos descendentes con piso horizontal. Sesgo bajista."
    elif abs(up_norm) <= FLAT and lo_norm > FLAT:
        name = "Triángulo Ascendente"
        desc = "Mínimos ascendentes con techo horizontal. Sesgo alcista."
    else:
        return None

    return Pattern(
        name=name,
        bar_indices=upper.point_indices + lower.point_indices,
        prices=[],
        description=desc,
    )


def find_patterns(swings: SwingPoints) -> list[Pattern]:
    patterns: list[Pattern] = []
    patterns.extend(detect_double_top(swings))
    patterns.extend(detect_double_bottom(swings))
    tri = detect_triangle(swings)
    if tri:
        patterns.append(tri)
    return patterns
