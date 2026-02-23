import numpy as np
from numpy.linalg import pinv as inversa
from Funcoes_auxiliares import matriz_observacao


def filtro_WLS_heterocedastico(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 2,
    eps: float = 1e-6,
    valor_max_clip: int | None = None,
    valor_min_clip: int = 0,
    normalizar_pesos: bool = True,
):
    """
    WLS com peso inverso da amplitude:
        w(n) = 1 / (|y(n)| + eps)
    """
    readout_shaper = np.clip(readout, valor_min_clip, valor_max_clip)
    X = matriz_observacao(readout_shaper, ordem_filtro=ordem_filter)
    y = np.asarray(sinal_desejado)[delay : delay + X.shape[0]]

    Xb = np.column_stack([X, np.ones(X.shape[0])])  # bias
    w = 1.0 / (np.abs(y) + eps)

    if normalizar_pesos:
        w = w / np.mean(w)

    sw = np.sqrt(w)
    Xw = Xb * sw[:, None]
    yw = y * sw

    beta = inversa(Xw.T @ Xw) @ Xw.T @ yw

    sinal_estimado = np.zeros(len(readout_shaper), dtype=float)
    sinal_estimado[: X.shape[0]] = Xb @ beta
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip, valor_max_clip)
    return sinal_estimado


def filtro_WLS_base_expandida(
    sinal_desejado: list | np.ndarray,
    readout: list | np.ndarray,
    ordem_filter: int = 7,
    delay: int = 2,
    eps: float = 1e-6,
    incluir_x2: bool = True,
    incluir_xn_xn1: bool = True,
    valor_max_clip: int | None = None,
    valor_min_clip: int = 0,
    normalizar_pesos: bool = True,
):
    """
    Combina as duas ideias:
      - WLS (heterocedasticidade)
      - base expandida (não linear leve)
    """
    readout_shaper = np.clip(readout, valor_min_clip, valor_max_clip)
    Xlin = matriz_observacao(readout_shaper, ordem_filtro=ordem_filter)
    y = np.asarray(sinal_desejado)[delay : delay + Xlin.shape[0]]

    feats = [Xlin]
    if incluir_x2:
        feats.append(Xlin**2)
    if incluir_xn_xn1 and ordem_filter >= 2:
        feats.append(Xlin[:, 1:] * Xlin[:, :-1])

    Xexp = np.column_stack(feats)
    Xb = np.column_stack([Xexp, np.ones(Xexp.shape[0])])

    w = 1.0 / (np.abs(y) + eps)
    if normalizar_pesos:
        w = w / np.mean(w)

    sw = np.sqrt(w)
    Xw = Xb * sw[:, None]
    yw = y * sw

    beta = inversa(Xw.T @ Xw) @ Xw.T @ yw

    sinal_estimado = np.zeros(len(readout_shaper), dtype=float)
    sinal_estimado[: Xlin.shape[0]] = Xb @ beta
    sinal_estimado = np.clip(sinal_estimado, valor_min_clip, valor_max_clip)
    return sinal_estimado
