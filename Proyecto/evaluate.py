"""
evaluate.py
===========
Pipeline de evaluación:收集predicciones, calcular métricas, comparar
baseline vs variantes con filtro Kalman.

Métricas
--------
- Accuracy  : fracción de clasificaciones binarias correctas
- Recall    : recall macro-promediado (importante para clase de ataque minoritaria)
- F1-Score  : media armónica macro-promediada de precisión y recall
- MAE       : error absoluto medio entre probabilidad y etiqueta binaria
- MSE       : error cuadrático medio entre probabilidad y etiqueta binaria
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from model   import AnomalyDetector
from kalman  import KalmanSmoother


# ─────────────────────────────────────────────────────────────────────────────
# Recolectar probabilidades crudas de un modelo entrenado
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(
    model:  AnomalyDetector,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ejecutar el modelo en todos los lotes en `loader` y recolectar
    probabilidades de anomalía y etiquetas de verdad terreno.

    Parámetros
    ----------
    model  : AnomalyDetector entrenado
    loader : DataLoader (val o test)
    device : dispositivo torch

    Retorna
    -------
    probs  : (N,) float32  P(class=anomaly) para cada ventana
    labels : (N,) int64    etiquetas binarias de verdad terreno
    """
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[int]   = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)                        # (B, 2)
        probs   = torch.softmax(logits, dim=-1)[:, 1]  # P(anomaly)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(y_batch.numpy().tolist())

    return (
        np.array(all_probs,  dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Calcular métricas a partir de probabilidades
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    probs:     np.ndarray,
    labels:    np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcular las cinco métricas requeridas.

    Parámetros
    ----------
    probs     : probabilidades predichas de anomalía (N,)
    labels    : etiquetas binarias de verdad terreno      (N,)
    threshold : umbral de decisión para clasificación binaria

    Retorna
    -------
    dict con claves: accuracy, recall, f1, mae, mse
    """
    preds = (probs >= threshold).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "recall":   float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1":       float(f1_score(labels, preds,     average="macro", zero_division=0)),
        "mae":      float(mean_absolute_error(labels, probs)),
        "mse":      float(mean_squared_error(labels,  probs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación completa: baseline + variantes Kalman
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    model:          AnomalyDetector,
    test_loader:    DataLoader,
    kalman_configs: list[tuple[float, float]],
    device:         torch.device,
    threshold:      float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Ejecutar evaluaciones baseline + con filtro Kalman.

    Parámetros
    ----------
    model          : AnomalyDetector entrenado
    test_loader    : DataLoader para conjunto de prueba
    kalman_configs : lista de tuplas (σ²_w, σ²_v) a evaluar
    device         : dispositivo torch
    threshold      : umbral de clasificación (default 0.5)

    Retorna
    -------
    results : {variant_name: metrics_dict}
    """
    print("Recolectando probabilidades crudas del modelo …")
    probs, labels = collect_probs(model, test_loader, device)

    results: dict[str, dict[str, float]] = {}

    # ── Baseline: probabilidades crudas del modelo ────────────────────────────
    results["Baseline (sin Kalman)"] = compute_metrics(probs, labels, threshold)

    # ── Variantes Kalman ───────────────────────────────────────────────
    for sigma2_w, sigma2_v in kalman_configs:
        ks       = KalmanSmoother(sigma2_w, sigma2_v)
        smoothed = ks.smooth_sequence(probs)
        name     = f"Kalman (σ²w={sigma2_w:.0e}, σ²v={sigma2_v:.0e})"
        results[name] = compute_metrics(smoothed, labels, threshold)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Imprimir tabla con formato
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Imprimir una tabla de comparación formateada a stdout."""
    col_w = 45
    header = (
        f"{'Variante':<{col_w}} "
        f"{'Accuracy':>9} {'Recall':>8} {'F1':>8} {'MAE':>8} {'MSE':>8}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, m in results.items():
        print(
            f"{name:<{col_w}} "
            f"{m['accuracy']:>9.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['mae']:>8.4f} "
            f"{m['mse']:>8.4f}"
        )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Helper de análisis de sensibilidad
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    probs:          np.ndarray,
    labels:         np.ndarray,
    sigma2_w_vals:  list[float],
    sigma2_v_vals:  list[float],
    threshold:      float = 0.5,
) -> dict:
    """
    Búsqueda en grilla sobre hiperparámetros de Kalman.

    Retorna un dict indexado por (σ²_w, σ²_v) con las métricas correspondientes.
    """
    grid_results: dict[tuple, dict] = {}

    for sw in sigma2_w_vals:
        for sv in sigma2_v_vals:
            ks       = KalmanSmoother(sw, sv)
            smoothed = ks.smooth_sequence(probs)
            metrics  = compute_metrics(smoothed, labels, threshold)
            grid_results[(sw, sv)] = metrics

    # Encontrar mejor configuración por F1
    best_key = max(grid_results, key=lambda k: grid_results[k]["f1"])
    print(f"\nMejor config Kalman por F1: σ²_w={best_key[0]}, σ²_v={best_key[1]}")
    print(f"  F1 = {grid_results[best_key]['f1']:.4f}")

    return grid_results
