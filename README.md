# Detección de Anomalías en Tráfico de Red
## Parcial II — Matemáticas Computacionales
### Modelo Híbrido SSM + Espectral con post-procesamiento Kalman

---

## Estructura del proyecto

```
proyecto/
├── model.py               # Arquitectura: SelectiveSSM, TemporalBlock,
│                          #   SpectralBlock, Fusion, AnomalyDetector
├── data.py                # Carga y preprocesamiento de UNSW-NB15
├── kalman.py              # Filtro de Kalman escalar
├── train.py               # Loop de entrenamiento
├── evaluate.py            # Cálculo de métricas y análisis comparativo
├── notebook.ipynb         # Ejecución paso a paso (Bloques 2 y 3)
├── documento_tecnico.tex  # Derivación matemática completa (Bloque 1)
└── README.md
```

---

## Requisitos del sistema (cumplidos)

| # | Requisito | Componente |
|---|-----------|------------|
| R1 | Complejidad lineal O(L) | `SelectiveSSM` recurrencia O(L·D·N) |
| R2 | Estable sin vanishing/exploding gradient | A = -diag(λ), λ>0 eigenvalores en (0,1) |
| R3 | Extracción de componentes periódicos | `SpectralBlock` vía rFFT/iRFFT |
| R4 | Fusión adaptativa con coeficientes aprendibles | `Fusion` α,β ∈ R^D |
| R5 | Post-procesamiento con estimador gaussiano recursivo | `KalmanSmoother` |

---

## Instalación

```bash
git clone <url-del-repo>
cd proyecto
pip install torch numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

---

## Dataset

Descargar **UNSW-NB15** de:
- https://research.unsw.edu.au/projects/unsw-nb15-dataset
- https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Colocar el CSV en la raíz y actualizar `CSV_PATH` en el notebook.
Si el archivo no está disponible, el notebook genera datos sintéticos automáticamente.

---

## Cómo ejecutar

### Opción 1: Notebook (recomendado)
```bash
jupyter notebook notebook.ipynb
# Kernel → Restart & Run All
```

### Opción 2: Scripts
```python
from data import make_loaders
from model import AnomalyDetector
from train import train
from evaluate import evaluate_all, print_results_table
import torch

# Entrenamiento
train_loader, val_loader, test_loader, input_dim = make_loaders('datos.csv')
model_kwargs = dict(input_dim=input_dim, d_model=64, d_state=16,
                    seq_len=64, top_k=16, kernel_size=3, n_layers=2, dropout=0.1)
model = AnomalyDetector(**model_kwargs)
train(model, train_loader, val_loader, n_epochs=20, checkpoint_path='best_model.pt',
      model_kwargs=model_kwargs)

# Evaluación
model = AnomalyDetector.load('best_model.pt')
device = torch.device('cpu')
results = evaluate_all(model, test_loader,
                       kalman_configs=[(1e-4, 1e-2), (1e-3, 1e-2), (1e-2, 1e-2)],
                       device=device)
print_results_table(results)

# Inferencia
prob = model.predict_proba(nueva_ventana)
```

---

## Hiperparámetros principales

| Parámetro | Valor | Categoría |
|-----------|-------|-----------|
| `d_model` | 64 | Capacidad |
| `d_state` | 16 | Capacidad |
| `top_k`   | 16 | Capacidad |
| `n_layers`| 2  | Capacidad |
| `seq_len` | 64 | Costo computacional |
| `batch_size` | 32 | Costo computacional |
| `lr` | 1e-3 | Estabilidad del entrenamiento |
| `clip_grad_norm` | 1.0 | Estabilidad del entrenamiento |
| `dropout` | 0.1 | Estabilidad del entrenamiento |

---

## Entorno recomendado

Google Colab con GPU T4 gratuita. Python >= 3.9, PyTorch >= 2.0.

## Librerías permitidas utilizadas

PyTorch, NumPy, Pandas, scikit-learn, imbalanced-learn, Matplotlib, Seaborn.

> **No se utilizó** mamba-ssm, s4-pytorch, state-spaces ni ningún paquete
> que implemente directamente la capa SSM requerida.

## Entregables generados

- `documento_tecnico.tex` — Derivación matemática completa (Bloque 1)
- `notebook.ipynb` — Implementación ejecutable con celdas ejecutadas (Bloques 2 y 3)
- `best_model.pt` — Checkpoint del mejor modelo según val F1
- `results_comparison.csv` — Tabla comparativa de métricas
