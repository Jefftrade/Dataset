"""
kalman.py
=========
Filtro de Kalman escalar para post-procesamiento de secuencias de probabilidad de anomalía.

Modelo de espacio-estado (observación de paseo aleatorio):
    Ecuación de estado :  s_{t+1} = s_t + w_t,   w_t ~ N(0, σ²_w)
    Ecuación de observación:  p_t     = s_t + v_t,   v_t ~ N(0, σ²_v)

donde:
    s_t  – probabilidad real latente de anomalía (estado oculto)
    p_t  – salida de probabilidad ruidosa del modelo neuronal (observado)
    σ²_w – varianza del ruido del proceso   (qué tan rápido puede cambiar el estado real)
    σ²_v – varianza del ruido de observación (ruido en la salida del modelo)

Recursión de Kalman
-------------------
Predicción:
    ŝ_{t|t-1} = ŝ_{t-1|t-1}                        (predicción de estado)
    P_{t|t-1} = P_{t-1|t-1} + σ²_w                  (predicción de varianza)

Ganancia de Kalman:
    K_t = P_{t|t-1} / (P_{t|t-1} + σ²_v)            ∈ (0, 1)

Actualización:
    ŝ_{t|t} = ŝ_{t|t-1} + K_t · (p_t − ŝ_{t|t-1}) (media posterior)
    P_{t|t} = (1 − K_t) · P_{t|t-1}                 (varianza posterior)

Comportamiento limitante
-----------------------
σ²_w → 0  (sistema muy estable):  K_t → 0  (ignorar observaciones, confiar en predicción)
σ²_w → ∞  (sistema muy volátil): K_t → 1 (confiar en observaciones, descartar predicción)
"""

from __future__ import annotations

import numpy as np
from typing import Union


class KalmanSmoother:
    """
    Filtro de Kalman escalar para suavizar secuencias de probabilidad de anomalía.

    Parámetros
    ----------
    sigma2_w : float > 0
        Varianza del ruido del proceso.  Controla qué tan rápido el estado de anomalía
        latente se asume que cambia.  Grande → filtro se adapta rápido (menos
        suavizado).  Pequeño → filtro asume estado casi-constante (más
        suavizado, pero lento para reaccionar a ataques súbitos).

    sigma2_v : float > 0
        Varianza del ruido de observación.  Modela la incertidumbre en la
        probabilidad de salida del modelo neuronal.  Grande → salida del modelo se considera
        ruidosa → más peso dado al paso de predicción.

    Uso
    -----
    >>> ks = KalmanSmoother(sigma2_w=1e-3, sigma2_v=1e-2)
    >>> smoothed = ks.smooth_sequence(model_probs)
    """

    def __init__(self, sigma2_w: float = 1e-3, sigma2_v: float = 1e-2):
        if sigma2_w <= 0 or sigma2_v <= 0:
            raise ValueError(
                f"Las varianzas de ruido deben ser estrictamente positivas; "
                f"obtenidas σ²_w={sigma2_w}, σ²_v={sigma2_v}"
            )
        self.sigma2_w: float = sigma2_w
        self.sigma2_v: float = sigma2_v
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Reiniciar filtro al estado inicial.
        Llamar entre series de tiempo independientes (p.ej. diferentes lotes de prueba)
        para evitar contaminación de estado.
        """
        # Prior: asumir que la probabilidad es 0.5 (máxima incertidumbre)
        self._s_hat: float = 0.5
        self._P:     float = 1.0     # covarianza inicial grande → prior difuso

    # ------------------------------------------------------------------
    def update(self, p_t: float) -> float:
        """
        Ejecutar un paso predicción → actualización del filtro de Kalman.

        Parámetros
        ----------
        p_t : float
            Probabilidad de anomalía observada del modelo neuronal en el tiempo t.

        Retorna
        -------
        s_hat_t : float
            Estimación posterior de la verdadera probabilidad de anomalía en el tiempo t.
        """
        # ── Paso de predicción ─────────────────────────────────────────────
        # Para paseo aleatorio: predicción = última estimación (sin término de deriva)
        s_pred = self._s_hat                             # ŝ_{t|t-1}
        P_pred = self._P + self.sigma2_w                 # P_{t|t-1}

        # ── Ganancia de Kalman ───────────────────────────────────────────────
        # K_t = P_{t|t-1} / (P_{t|t-1} + σ²_v)
        # K_t ∈ (0,1): balancea predicción vs observación
        K: float = P_pred / (P_pred + self.sigma2_v)     # K_t

        # ── Paso de actualización ───────────────────────────────────────────
        innovation      = p_t - s_pred                   # p_t − ŝ_{t|t-1}
        self._s_hat     = s_pred + K * innovation        # ŝ_{t|t}
        self._P         = (1.0 - K) * P_pred            # P_{t|t}

        return float(self._s_hat)

    # ------------------------------------------------------------------
    def smooth_sequence(
        self,
        probs: Union[list[float], np.ndarray],
    ) -> np.ndarray:
        """
        Aplicar el filtro secuencialmente a una secuencia completa de probabilidades.

        Parámetros
        ----------
        probs : similar a array de forma (T,)
            Probabilidades de anomalía en [0, 1] producidas por el modelo neuronal.

        Retorna
        -------
        smoothed : np.ndarray de forma (T,)
            Estimaciones de probabilidades filtradas.
        """
        self.reset()
        smoothed = np.empty(len(probs), dtype=np.float32)
        for t, p in enumerate(probs):
            smoothed[t] = self.update(float(p))
        return smoothed

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"KalmanSmoother("
            f"σ²_w={self.sigma2_w:.2e}, "
            f"σ²_v={self.sigma2_v:.2e})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Conveniencia: analizar la sensibilidad de la ganancia de Kalman
# ─────────────────────────────────────────────────────────────────────────────

def steady_state_gain(sigma2_w: float, sigma2_v: float) -> float:
    """
    Calcular la ganancia de Kalman en estado estacionario K_∞.

    Resolviendo la ecuación algebraica de Riccati para el modelo escalar
    de observación de paseo aleatorio se obtiene:

        P_∞ = (σ²_w/2) + sqrt((σ²_w/2)² + σ²_w · σ²_v)

    y luego K_∞ = P_∞ / (P_∞ + σ²_v).

    Esto es útil para entender cuán agresivo será el suavizado
    en estado estacionario, antes de ejecutarse sobre datos reales.
    """
    half_w  = sigma2_w / 2.0
    P_inf   = half_w + np.sqrt(half_w**2 + sigma2_w * sigma2_v)
    K_inf   = P_inf  / (P_inf + sigma2_v)
    return float(K_inf)
