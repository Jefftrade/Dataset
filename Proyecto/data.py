"""
data.py
=======
Carga, preprocesamiento, balanceo y ventaneo del conjunto de datos UNSW-NB15.

Canalización
--------
1. cargar_raw()             – leer CSV, limpiar, descartar columnas irrelevantes
2. codificar_categoricas()  – codificación ordinal proto / servicio / estado
3. preprocesar()           – MinMaxScaler (ajuste solo en ENTRENAMIENTO), SMOTE+under
4. SlidingWindowDataset   – ventana deslizante de longitud L, etiqueta por votación mayoritaria
5. crear_cargadores()         – fábrica de extremo a extremo → DataLoaders

Nota importante sobre fuga de datos
-------------------------------
El MinMaxScaler se ajusta SOLO en la división de entrenamiento.
Aplicarlo a los conjuntos de validación y prueba utiliza transform() (no fit_transform()).
Si el escalador se ajustara a todo el conjunto de datos, las estadísticas de futuros/pruebas
se filtrarían al entrenamiento, inflando artificialmente el rendimiento.
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline       import Pipeline as ImbPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["proto", "service", "state"]
TARGET_COL       = "label"
DROP_COLS        = ["id", "attack_cat"]   # columnas no características


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga sin procesar
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """
    Leer un CSV de UNSW-NB15 y realizar limpieza mínima.

    - Eliminar espacios en blanco de los nombres de columnas
    - Descartar columnas identificador / no características
    - Descartar filas con objetivo faltante
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # Descartar columnas irrelevantes si están presentes
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Asegurar objetivo binario (0 = normal, 1 = ataque)
    if TARGET_COL in df.columns:
        df.dropna(subset=[TARGET_COL], inplace=True)
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Codificación categórica
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codificación ordinal de CATEGORICAL_COLS en su lugar.
    Las categorías desconocidas en val/test se asignarán a un nuevo entero mediante
    fit_transform en la columna completa.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            # Columna ausente (algunas versiones de archivos la omiten): agregar ceros
            df[col] = 0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Canalización de preprocesamiento completa
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Todas las columnas excepto el objetivo."""
    return [c for c in df.columns if c != TARGET_COL]


def preprocess(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    balance:  bool = True,
    random_state: int = 42,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    MinMaxScaler,
]:
    """
    Preprocesamiento completo.

    1. Extraer matriz de características y etiquetas.
    2. Ajustar MinMaxScaler solo en datos de entrenamiento (→ sin fuga).
    3. Aplicar sobremuestreo SMOTE + submuestreo aleatorio en conjunto de entrenamiento.

    Parámetros
    ----------
    df_train, df_val, df_test : DataFrames después de la codificación categórica
    balance    : si aplicar SMOTE + submuestreo en el entrenamiento
    random_state : semilla de reproducibilidad

    Devuelve
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    feat_cols = get_feature_cols(df_train)

    # Convertir a numpy
    X_tr   = df_train[feat_cols].values.astype(np.float32)
    y_tr   = df_train[TARGET_COL].values.astype(np.int64)
    X_val  = df_val[feat_cols].values.astype(np.float32)
    y_val  = df_val[TARGET_COL].values.astype(np.int64)
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = df_test[TARGET_COL].values.astype(np.int64)

    # ── Ajustar escalador SOLO en entrenamiento ── previene fuga ──────────────
    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_val  = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # ── Equilibrar conjunto de entrenamiento ──────────────────────────────────────────
    if balance:
        # Canalización SMOTE + submuestreo de imbalanced-learn
        bal_pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
            ("under", RandomUnderSampler(random_state=random_state)),
        ])
        X_tr, y_tr = bal_pipeline.fit_resample(X_tr, y_tr)
        X_tr = X_tr.astype(np.float32)
        y_tr = y_tr.astype(np.int64)

    return X_tr, y_tr, X_val, y_val, X_test, y_test, scaler


def preprocess_single_file(
    csv_path: str | Path,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    balance:   bool  = True,
    random_state: int = 42,
):
    """
    Envoltura de conveniencia: cargar un CSV único, dividir en entrenamiento/validación/prueba,
    luego ejecutar la canalización de preprocesamiento completa.

    Devuelve la misma tupla que preprocess() más feat_cols.
    """
    df = encode_categoricals(load_raw(csv_path))

    # División estratificada para preservar la proporción de clase
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    val_fraction = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_fraction,
        stratify=y_tmp,
        random_state=random_state,
    )

    # Envolver de vuelta en DataFrames para preprocess()
    df_tr  = pd.DataFrame(X_tr,  columns=feat_cols); df_tr[TARGET_COL]  = y_tr
    df_val = pd.DataFrame(X_val, columns=feat_cols); df_val[TARGET_COL] = y_val
    df_te  = pd.DataFrame(X_test,columns=feat_cols); df_te[TARGET_COL]  = y_test

    result = preprocess(df_tr, df_val, df_te, balance=balance,
                        random_state=random_state)
    return result + (feat_cols,)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conjunto de datos de ventana deslizante
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Conjunto de datos de ventana deslizante.

    Convierte una secuencia plana (N, F) en un conjunto de ventanas (M, L, F)
    donde cada ventana está etiquetada por la votación mayoritaria de etiquetas
    dentro de esa ventana.

    Parámetros
    ----------
    X           : matriz (N, F) float32 de características normalizadas
    y           : etiquetas binarias (N,) int64
    window_size : longitud de ventana L
    stride      : paso entre ventanas consecutivas
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 64,
        stride:      int = 32,
    ):
        assert len(X) == len(y), "X e y deben tener la misma longitud"
        self.window_size = window_size

        windows: list[np.ndarray] = []
        labels:  list[int]        = []

        n = len(X)
        for start in range(0, n - window_size + 1, stride):
            windows.append(X[start : start + window_size])
            chunk  = y[start : start + window_size]
            # Etiqueta de votación mayoritaria: si >50% son anómalos, la ventana es anómala
            labels.append(int(np.bincount(chunk).argmax()))

        if len(windows) == 0:
            raise ValueError(
                f"No se crearon ventanas: longitud del conjunto de datos {n} < window_size {window_size}"
            )

        self.X = np.stack(windows, axis=0).astype(np.float32)  # (M, L, F)
        self.y = np.array(labels, dtype=np.int64)              # (M,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),   # (L, F)
            torch.tensor(self.y[idx]),       # escalar
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Función de fábrica
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(
    csv_path:    str | Path,
    window_size: int   = 64,
    stride:      int   = 32,
    batch_size:  int   = 32,
    balance:     bool  = True,
    val_size:    float = 0.15,
    test_size:   float = 0.15,
    num_workers: int   = 0,
    random_state: int  = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Fábrica de extremo a extremo: CSV único → DataLoaders (entrenamiento, validación, prueba).

    Parámetros
    ----------
    csv_path    : ruta a CSV de UNSW-NB15 (variante de archivo único)
    window_size : longitud de secuencia L
    stride      : paso de ventana deslizante
    batch_size  : tamaño de mini-lote
    balance     : SMOTE + submuestreo
    val_size    : fracción para validación
    test_size   : fracción para prueba
    num_workers : trabajadores de DataLoader
    random_state: reproducibilidad

    Devuelve
    -------
    train_loader, val_loader, test_loader, input_dim (F)
    """
    result = preprocess_single_file(
        csv_path,
        val_size=val_size,
        test_size=test_size,
        balance=balance,
        random_state=random_state,
    )
    X_tr, y_tr, X_val, y_val, X_te, y_te, scaler, feat_cols = result

    ds_tr  = SlidingWindowDataset(X_tr,  y_tr,  window_size, stride)
    ds_val = SlidingWindowDataset(X_val, y_val, window_size, stride)
    ds_te  = SlidingWindowDataset(X_te,  y_te,  window_size, stride)

    print(f"Dataset sizes → train: {len(ds_tr)} | val: {len(ds_val)} | test: {len(ds_te)}")
    print(f"Class balance (train) → {np.bincount(ds_tr.y)}")

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    train_loader = DataLoader(ds_tr,  shuffle=True,  **loader_kw)
    val_loader   = DataLoader(ds_val, shuffle=False, **loader_kw)
    test_loader  = DataLoader(ds_te,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, X_tr.shape[1]
