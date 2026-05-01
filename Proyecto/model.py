"""
model.py
========
Detector de Anomalías Híbrido SSM + Espectral para detección de anomalías de tráfico de red.

Arquitectura
------------
características_crudas (B, L, F)
  → proyección lineal       (F → D)
  → [TemporalBlock || SpectralBlock]   (ramas paralelas)
  → Fusión (α, β aprendibles)
  → mean-pool sobre L
  → LayerNorm → Linear(D, 2)   (logits binarios)

Módulos
-------
SelectiveSSM   – SSM discreto con Δ, B, C dependientes de entrada (estilo Mamba)
TemporalBlock  – SelectiveSSM + convolución depthwise + puerta SiLU + residual + LN
SpectralBlock  – rFFT → filtro top-K → iRFFT
Fusión         – z = α ⊙ x_temp + β ⊙ x_spec
AnomalyDetector– Pipeline completo con helpers de carga/guardado
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Modelo de Espacio-Estado Selectivo  (S-SSM)
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    Discrete-time Selective SSM with ZOH discretisation.

    Continuous system (per feature channel d, state index n):
        x'(t) = A·x(t) + B(u)·u(t)
        y(t)  = C(u)·x(t)

    A is restricted to -diag(λ₁,…,λ_N), λᵢ > 0  → guaranteed stability.

    ZOH discretisation at step Δ:
        Ā  = exp(Δ ⊙ A)                  ∈ R^{D×N}
        B̄  = (Ā − I) ⊘ A  ⊙  B(u_t)    ∈ R^{B×L×D×N}

    Recurrence (O(L·D·N)):
        h_t = Ā ⊙ h_{t-1} + B̄_t
        y_t = Σ_n  C_t[n] · h_t[:,:,n]  → shape (B, L, D)

    Parameters
    ----------
    d_model : int   Feature / channel dimension D.
    d_state : int   SSM hidden state size N.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A almacenado como log para garantizar positividad: A = -exp(A_log) < 0
        # Forma (D, N) — un conjunto de autovalores por canal de característica
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.5)

        # Proyecciones dependientes de entrada: u_t → Δ_t, B_t, C_t
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)
        self.B_proj     = nn.Linear(d_model, d_state,  bias=False)
        self.C_proj     = nn.Linear(d_model, d_state,  bias=False)

        # Inicializar sesgo delta para que softplus(sesgo) ≈ 0.018 inicialmente
        nn.init.constant_(self.delta_proj.bias, -4.0)

    # ------------------------------------------------------------------
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        u : Tensor  forma (B, L, D)

        Retorna
        -------
        y : Tensor  forma (B, L, D)
        """
        B, L, D = u.shape
        N = self.d_state

        # ─── Construir A: diagonal negativa, λᵢ > 0 garantiza estabilidad ──────
        A = -torch.exp(self.A_log)                   # (D, N)

        # ─── Parámetros dependientes de entrada ─────────────────────────────────────────
        delta = F.softplus(self.delta_proj(u))        # (B, L, D)
        Bt    = self.B_proj(u)                        # (B, L, N)
        Ct    = self.C_proj(u)                        # (B, L, N)

        # ─── Discretización ZOH ─────────────────────────────────────────────────────
        # Ā = exp(Δ ⊗ A),  broadcast: delta (B,L,D,1) * A (1,1,D,N)
        dA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )                                             # (B, L, D, N)

        # B̄ = (Ā − 1) / A * B_t,  Bt (B,L,N) → (B,L,1,N)
        dB = (dA - 1.0) / A.unsqueeze(0).unsqueeze(0) \
             * Bt.unsqueeze(2)                        # (B, L, D, N)

        # ─── Recurrencia secuencial O(L · D · N) ──────────────────────────────────
        h: torch.Tensor = u.new_zeros(B, D, N)
        ys: list[torch.Tensor] = []

        for t in range(L):
            h  = dA[:, t] * h + dB[:, t]             # (B, D, N)
            # y_t[b,d] = Σ_n C_t[b,n]·h[b,d,n]
            yt = (h * Ct[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(yt)

        return torch.stack(ys, dim=1)                 # (B, L, D)


# =============================================================================
# 2. Bloque Temporal
# =============================================================================

class TemporalBlock(nn.Module):
    """
    Envuelve SelectiveSSM con conv causal local, puerta SiLU y residual.

    Pase hacia adelante:
        1. Conv1d depthwise causal  (contexto de corto alcance, ancho k)
        2. SelectiveSSM             (recurrencia selectiva de largo alcance)
        3. Puerta SiLU                g_t = SiLU(gate_proj(u_t))
        4. Salida puerteada             x_t = ssm(u_t) ⊙ g_t
        5. Residual + LayerNorm     out = LN(x_t + u_t)

    Parámetros
    ----------
    d_model     : dimensión de característica D
    d_state     : dimensión de estado SSM N
    kernel_size : ancho de conv causal k
    """

    def __init__(
        self,
        d_model:     int,
        d_state:     int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv_k = kernel_size

        # Conv depthwise causal (groups=D → por-canal)
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,   # rellenar izquierda; recortamos la cola
            groups=d_model,
            bias=True,
        )

        self.ssm       = SelectiveSSM(d_model, d_state)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        u : (B, L, D)
        Retorna
        -------
        out : (B, L, D)
        """
        # Conv causal: (B, D, L) → recortar cola → (B, L, D)
        u_conv = self.conv(u.transpose(1, 2))
        # Eliminar relleno derecho para mantener propiedad causal
        if self.conv_k > 1:
            u_conv = u_conv[..., :-(self.conv_k - 1)]
        u_conv = u_conv.transpose(1, 2)         # (B, L, D)

        # SSM sobre señal convolucionada localmente
        x   = self.ssm(u_conv)                 # (B, L, D)

        # Puerta SiLU (modulación consciente de entrada)
        g   = F.silu(self.gate_proj(u))        # (B, L, D)
        out = x * g                            # elemento a elemento

        # Conexión residual + normalización
        return self.norm(out + u)              # (B, L, D)


# =============================================================================
# 3. Bloque Espectral
# =============================================================================

class SpectralBlock(nn.Module):
    """
    Extractor de características espectrales vía rFFT → filtro complejo aprendible → iRFFT.

    Pasos (según la formulación del examen):
      (a) rFFT a lo largo del eje temporal     X[k] ∈ C^{(L/2+1)×D}
      (b) mantener los K bins de mayor magnitud media
      (c) aplicar un filtro complejo aprendible W ∈ C^{K×D}  (elemento a elemento)
      (d) iRFFT de vuelta al dominio del tiempo

    La propiedad de simetría conjugada del rFFT significa que para una
    entrada real de longitud L, solo se necesitan L/2+1 coeficientes complejos
    (los restantes son espejos conjugados).

    Parámetros
    ----------
    d_model : dimensión de característica D
    seq_len : longitud esperada de secuencia L  (usada para asignar buffer de filtro)
    top_k   : número de bins de frecuencia a mantener (K)
    """

    def __init__(self, d_model: int, seq_len: int, top_k: int = 16):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.top_k   = min(top_k, seq_len // 2 + 1)
        self.n_freqs = seq_len // 2 + 1

        # Filtro complejo aprendible almacenado como par (real, imaginario)
        # Inicializado cerca de 1+0j para que el bloque comience como casi-identidad
        self.W_real = nn.Parameter(torch.ones(self.top_k, d_model) * 0.02)
        self.W_imag = nn.Parameter(torch.zeros(self.top_k, d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        u : (B, L, D)
        Retorna
        -------
        out : (B, L, D)  señal reconstruida del espectro filtrado
        """
        B, L, D = u.shape

        # (a) rFFT a lo largo de la dimensión temporal → (B, n_freqs, D)  complejo
        X = torch.fft.rfft(u, dim=1, norm="ortho")

        # (b) Seleccionar los K bins principales por potencia media sobre lotes y canales
        magnitudes = X.abs().mean(dim=(0, 2))                    # (n_freqs,)
        top_idx    = torch.topk(magnitudes, self.top_k).indices  # (K,)
        X_k        = X[:, top_idx, :]                           # (B, K, D)

        # (c) Filtro complejo aprendible W ∈ C^{K×D}
        W          = torch.complex(self.W_real, self.W_imag)    # (K, D)
        X_filtered = X_k * W.unsqueeze(0)                       # (B, K, D)

        # Reconstruir espectro completo (poner a cero bins no seleccionados)
        X_out              = torch.zeros_like(X)
        X_out[:, top_idx, :] = X_filtered

        # (d) iRFFT de vuelta al dominio del tiempo → (B, L, D)
        return torch.fft.irfft(X_out, n=L, dim=1, norm="ortho")


# =============================================================================
# 4. Fusion
# =============================================================================

class Fusion(nn.Module):
    """
    Fusión adaptativa elemento a elemento:

        z_t = α ⊙ x_temporal + β ⊙ x_spectral

    α, β ∈ R^D son pesos aprendibles por canal.
    Usar elemento a elemento (⊙) en lugar de un escalar permite al modelo
    ponderar selectivamente cada dimensión de característica independientemente.

    Inicializado como α = β = 0.5 (contribución igual equilibrada).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((d_model,), 0.5))
        self.beta  = nn.Parameter(torch.full((d_model,), 0.5))

    def forward(
        self,
        x_temp: torch.Tensor,
        x_spec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parámetros
        ----------
        x_temp : (B, L, D)  salida de rama temporal
        x_spec : (B, L, D)  salida de rama espectral
        Retorna
        -------
        z : (B, L, D)
        """
        return self.alpha * x_temp + self.beta * x_spec


# =============================================================================
# 5. Full Anomaly Detector
# =============================================================================

class AnomalyDetector(nn.Module):
    """
    Pipeline completo de detección de anomalías híbrido.

    características crudas (B, L, F)
      → input_proj: Linear(F→D) + Dropout
      → por cada capa:
            x_t  = TemporalBlock(z)          ─┐ paralelo
            x_s  = SpectralBlock(z)          ─┤
            z    = Fusion(x_t, x_s)          ←┘
      → pooling promedio sobre L  →  (B, D)
      → LayerNorm → Linear(D, 2)  (logits CE binarios)

    Parámetros
    ----------
    input_dim   : número de características de entrada F
    d_model     : ancho del modelo interno D        (capacidad)
    d_state     : dimensión de estado SSM N         (capacidad)
    seq_len     : ventana / longitud de secuencia L    (costo computacional)
    top_k       : bins espectrales a mantener K       (capacidad + costo)
    kernel_size : ancho de conv depthwise          (capacidad)
    n_layers    : número de capas del codificador      (capacidad + costo)
    dropout     : tasa de dropout de entrada            (regularización)
    """

    def __init__(
        self,
        input_dim:   int,
        d_model:     int   = 64,
        d_state:     int   = 16,
        seq_len:     int   = 64,
        top_k:       int   = 16,
        kernel_size: int   = 3,
        n_layers:    int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.seq_len  = seq_len
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout    = nn.Dropout(dropout)

        # Encoder layers (parallel temporal + spectral + fusion)
        self.temporal_blocks = nn.ModuleList(
            [TemporalBlock(d_model, d_state, kernel_size)
             for _ in range(n_layers)]
        )
        self.spectral_blocks = nn.ModuleList(
            [SpectralBlock(d_model, seq_len, top_k)
             for _ in range(n_layers)]
        )
        self.fusions = nn.ModuleList(
            [Fusion(d_model) for _ in range(n_layers)]
        )

        # Cabeza de clasificación
        self.head_norm = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, 2)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier uniforme para capas lineales; unos/ceros para normalizaciones."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : (B, L, F)  características ventiladas crudas
        Retorna
        -------
        logits : (B, 2)
        """
        z = self.dropout(self.input_proj(x))    # (B, L, D)

        for temp_blk, spec_blk, fuse in zip(
            self.temporal_blocks,
            self.spectral_blocks,
            self.fusions,
        ):
            x_t = temp_blk(z)                   # (B, L, D)
            x_s = spec_blk(z)                   # (B, L, D)
            z   = fuse(x_t, x_s)               # (B, L, D)

        # Pooling promedio global + clasificación
        pooled = z.mean(dim=1)                  # (B, D)
        return self.head(self.head_norm(pooled))# (B, 2)

    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retornar P(anomalía) para una o un lote de ventanas.

        Parámetros
        ----------
        x : (L, F)  o  (B, L, F)
        Retorna
        -------
        prob : (B,) o escalar — probabilidad de anomalía
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
        return probs[:, 1]

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "AnomalyDetector":
        """Cargar modelo desde un checkpoint creado con .save()."""
        ckpt  = torch.load(path, map_location=map_location)
        model = cls(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def save(self, path: str, model_kwargs: dict) -> None:
        """Guardar pesos del modelo + kwargs del constructor."""
        torch.save(
            {"state_dict": self.state_dict(), "model_kwargs": model_kwargs},
            path,
        )
