"""
train.py
========
Bucle de entrenamiento para AnomalyDetector.

Características
--------
- Optimizador Adam, lr=1e-3
- Tamaño de batch 32 (configurado en DataLoader)
- CrossEntropyLoss
- Recorte de gradientes (clip_grad_norm_, max_norm=1.0)
- Selección del mejor checkpoint basada en macro-F1 de validación
- Logging por época de pérdida de entrenamiento, pérdida de validación y F1 de validación
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from model import AnomalyDetector


# ─────────────────────────────────────────────────────────────────────────────
# Época de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:         AnomalyDetector,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    criterion:     nn.Module,
    device:        torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Ejecutar una época completa de entrenamiento.

    Parámetros
    ----------
    model         : AnomalyDetector (debe estar en `device`)
    loader        : DataLoader de entrenamiento
    optimizer     : optimizador Adam (u otro)
    criterion     : función de pérdida (CrossEntropyLoss)
    device        : dispositivo objetivo
    max_grad_norm : norma de recorte de gradientes

    Retorna
    -------
    mean_loss : float – pérdida media de entrenamiento sobre todas las muestras
    """
    model.train()
    total_loss = 0.0
    n_samples  = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)        # (B, L, F)
        y_batch = y_batch.to(device)        # (B,)

        optimizer.zero_grad()
        logits = model(X_batch)             # (B, 2)
        loss   = criterion(logits, y_batch)

        loss.backward()

        # Recorte de gradientes: previene explosión de gradientes para secuencias largas
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        n_samples  += len(y_batch)

    return total_loss / max(n_samples, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Época de validación / evaluación
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model:     AnomalyDetector,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Evaluar modelo en un DataLoader.

    Retorna
    -------
    mean_loss : float
    macro_f1  : float  – F1 macro-promediado (bueno para clases desbalanceadas)
    """
    model.eval()
    total_loss  = 0.0
    n_samples   = 0
    all_preds:  list[int] = []
    all_labels: list[int] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        n_samples  += len(y_batch)

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().tolist())

    mean_loss = total_loss / max(n_samples, 1)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return mean_loss, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Procedimiento de entrenamiento completo
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:           AnomalyDetector,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    n_epochs:        int   = 10,
    lr:              float = 1e-3,
    max_grad_norm:   float = 1.0,
    checkpoint_path: str   = "best_model.pt",
    device:          torch.device | None = None,
    model_kwargs:    dict | None         = None,
) -> dict:
    """
    Procedimiento completo de entrenamiento con guardado temprano basado en validación.

    Parámetros
    ----------
    model            : instancia de AnomalyDetector
    train_loader     : DataLoader para conjunto de entrenamiento
    val_loader       : DataLoader para conjunto de validación
    n_epochs         : número máximo de épocas para entrenar
    lr               : tasa de aprendizaje de Adam (default 1e-3)
    max_grad_norm    : norma de recorte de gradientes (default 1.0)
    checkpoint_path  : ruta para guardar el mejor checkpoint
    device           : dispositivo torch; auto-detectado si None
    model_kwargs     : kwargs del constructor necesarios para recargar el modelo

    Retorna
    -------
    history : dict con claves
        "train_loss" : list[float] – pérdida de entrenamiento por época
        "val_loss"   : list[float] – pérdida de validación por época
        "val_f1"     : list[float] – macro-F1 de validación por época
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "val_f1":     [],
    }

    print("=" * 65)
    print(f"  Entrenando en : {device}")
    print(f"  Épocas      : {n_epochs}")
    print(f"  LR          : {lr}")
    print(f"  Recorte grad   : {max_grad_norm}")
    print(f"  Checkpoint  : {checkpoint_path}")
    print("=" * 65)
    print(f"{'Época':>6}  {'Pérdida de Entrenamiento':>12}  {'Pérdida Val':>10}  "
          f"{'F1 Val':>8}  {'Tiempo(s)':>8}")
    print("-" * 65)

    for epoch in range(1, n_epochs + 1):
        t_start = time.time()

        tr_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm
        )
        val_loss, val_f1 = eval_epoch(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t_start

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"{epoch:>6}  {tr_loss:>12.6f}  {val_loss:>10.6f}  "
            f"{val_f1:>8.4f}  {elapsed:>8.1f}"
        )

        # Guardar checkpoint cuando F1 de validación mejora
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if model_kwargs is not None:
                model.save(checkpoint_path, model_kwargs)
            print(f"         ✓ Nuevo mejor checkpoint  (F1 val = {best_val_f1:.4f})")

    print("=" * 65)
    print(f"  Entrenamiento completado.  Mejor F1 val = {best_val_f1:.4f}")
    print("=" * 65)

    return history
