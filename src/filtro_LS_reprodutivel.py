"""Filtro LS (Moore-Penrose) com busca reprodutivel de ordem/delay e avaliação Monte Carlo."""

# Bibliotecas #
import numpy as np
from typing import Callable
from numpy.linalg import pinv as inversa
import matplotlib.pyplot as plt
from src.Gerador_de_Sinais import main as gerador_sinal_entrada_saida


# Funções auxiliares #
def RMSE_e_MAE_por_ordem(A, B, printar: bool = False):
    diff = np.asarray(A) - np.asarray(B)
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    if printar:
        print(f"RMSE = {rmse:.6f}")
        print(f"MAE  = {mae:.6f}")
    return rmse, mae


def matriz_observacao(sinal: np.ndarray, ordem_filtro: int = 2):
    return np.lib.stride_tricks.sliding_window_view(np.asarray(sinal), ordem_filtro)


def filtro_LS_novo(
    sinal_desejado: np.ndarray,
    readout: np.ndarray,
    ordem_filter: int = 7,
    delay: int = 2,
    valor_minimo_clip: float | None = 0.0,
    valor_max_clip: float | None = None,
):
    """Deconvolucao LS via pseudo-inversa com bias + clip opcional."""
    sinal_desejado = np.asarray(sinal_desejado)
    readout = np.asarray(readout)

    # clip correto: np.clip(x, a_min, a_max)
    readout_shaper = np.clip(readout, valor_minimo_clip, valor_max_clip)

    X = matriz_observacao(readout_shaper, ordem_filtro=ordem_filter)
    X_bias = np.column_stack([X, np.ones(X.shape[0])])

    end = X.shape[0] + delay
    y = sinal_desejado[delay:end]

    w_full = inversa(X_bias.T @ X_bias) @ X_bias.T @ y
    peso = np.flipud(w_full[:-1])
    bias = w_full[-1]

    sinal_estimado = np.zeros(len(readout_shaper))
    max_i = len(readout_shaper) - ordem_filter + 1
    for i in range(max_i):
        sinal_estimado[i] = np.sum(readout_shaper[i : i + ordem_filter] * peso) + bias

    sinal_estimado = np.clip(sinal_estimado, valor_minimo_clip, valor_max_clip)
    return sinal_estimado


def busca_ordem_otima_filtro(
    ordem_mais_alta: int,
    signal_original: np.ndarray,
    step: int = 1,
    filtro: Callable = filtro_LS_novo,
    modo_comparacao: str = "seu",  # "seu" | "justo"
    **filtro_kwargs,
):
    """
    Busca melhor ordem por RMSE/MAE.
    modo_comparacao="seu": compara cada ordem na janela valida dela (N-k+1).
    modo_comparacao="justo": compara todas as ordens na mesma janela comum.
    """
    if ordem_mais_alta < 3:
        raise ValueError("ordem_mais_alta deve ser >= 3")
    if step <= 0:
        raise ValueError("step deve ser > 0")

    signal_original = np.asarray(signal_original)
    N = len(signal_original)
    delay = int(filtro_kwargs.get("delay", 0))

    if modo_comparacao not in {"seu", "justo"}:
        raise ValueError("modo_comparacao deve ser 'seu' ou 'justo'")

    common_len = N - ordem_mais_alta + 1
    if modo_comparacao == "justo" and common_len <= 0:
        raise ValueError("ordem_mais_alta muito alta para comparacao justa")

    melhor_rmse = np.inf
    melhor_mae = np.inf
    out = {
        "Ordem_Filtro_RMSE": None,
        "RMSE": np.inf,
        "Ordem_Filtro_MAE": None,
        "MAE": np.inf,
    }

    for ordem_filter in range(3, ordem_mais_alta + 1, step):
        sinal = filtro(
            sinal_desejado=signal_original,
            ordem_filter=ordem_filter,
            **filtro_kwargs,
        )

        if modo_comparacao == "seu":
            max_i = N - ordem_filter + 1
            estimado_eval = sinal[:max_i]
            original_eval = signal_original[delay : delay + max_i]
        else:
            estimado_eval = sinal[:common_len]
            original_eval = signal_original[delay : delay + common_len]

        L = min(len(estimado_eval), len(original_eval))
        rmse, mae = RMSE_e_MAE_por_ordem(estimado_eval[:L], original_eval[:L])

        if rmse < melhor_rmse:
            melhor_rmse = rmse
            out["RMSE"] = float(rmse)
            out["Ordem_Filtro_RMSE"] = ordem_filter

        if mae < melhor_mae:
            melhor_mae = mae
            out["MAE"] = float(mae)
            out["Ordem_Filtro_MAE"] = ordem_filter

    out["RMSE"] = round(out["RMSE"], 6)
    out["MAE"] = round(out["MAE"], 6)
    return out


def busca_delay_otimo(
    delays: int,
    signal_original: np.ndarray,
    filtro: Callable = filtro_LS_novo,
    **filtro_kwargs,
):
    """Busca melhor delay com ordem fixa, comparando apenas a parte valida."""
    signal_original = np.asarray(signal_original)
    N = len(signal_original)
    ordem = int(filtro_kwargs["ordem_filter"])
    max_i = N - ordem + 1

    melhor = {
        "Delay_RMSE": None,
        "RMSE": np.inf,
        "Delay_MAE": None,
        "MAE": np.inf,
    }

    for d in range(delays):
        sinal = filtro(
            sinal_desejado=signal_original,
            delay=d,
            **filtro_kwargs,
        )

        estimado_eval = np.asarray(sinal)[:max_i]
        original_eval = signal_original[d : d + max_i]

        L = min(len(estimado_eval), len(original_eval))
        rmse, mae = RMSE_e_MAE_por_ordem(estimado_eval[:L], original_eval[:L])

        if rmse < melhor["RMSE"]:
            melhor["RMSE"] = rmse
            melhor["Delay_RMSE"] = d
        if mae < melhor["MAE"]:
            melhor["MAE"] = mae
            melhor["Delay_MAE"] = d

    melhor["RMSE"] = round(float(melhor["RMSE"]), 6)
    melhor["MAE"] = round(float(melhor["MAE"]), 6)
    return melhor


def resumo_stats(v):
    v = np.asarray(v, dtype=float)
    return {
        "media": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }


def experimento_monte_carlo(
    n_execucoes: int = 100,
    seed_inicial: int = 42,
    qntd_amostras: int = 50,
    ordem_max: int = 20,
    delays: int = 5,
    modo_comparacao_ordem: str = "seu",  # "seu" ou "justo"
):
    rmses, maes, ordens, delays_best = [], [], [], []

    for k in range(n_execucoes):
        np.random.seed(seed_inicial + k)

        sinal_original, readout = gerador_sinal_entrada_saida(qntd_amostras)

        melhor_ordem = busca_ordem_otima_filtro(
            ordem_mais_alta=ordem_max,
            signal_original=sinal_original,
            step=1,
            filtro=filtro_LS_novo,
            modo_comparacao=modo_comparacao_ordem,
            readout=readout,
        )
        ordem = melhor_ordem["Ordem_Filtro_RMSE"]

        melhor_delay = busca_delay_otimo(
            delays=delays,
            signal_original=sinal_original,
            filtro=filtro_LS_novo,
            readout=readout,
            ordem_filter=ordem,
        )
        d = melhor_delay["Delay_RMSE"]

        est = filtro_LS_novo(
            sinal_desejado=sinal_original,
            readout=readout,
            ordem_filter=ordem,
            delay=d,
        )

        max_i = len(sinal_original) - ordem + 1
        rmse, mae = RMSE_e_MAE_por_ordem(est[:max_i], sinal_original[d : d + max_i])

        rmses.append(rmse)
        maes.append(mae)
        ordens.append(ordem)
        delays_best.append(d)

    return {
        "RMSE": resumo_stats(rmses),
        "MAE": resumo_stats(maes),
        "ordem_media": float(np.mean(ordens)),
        "delay_medio": float(np.mean(delays_best)),
        "ordens": ordens,
        "delays": delays_best,
    }


def plot_estimado_x_original(original, estimado, limite_filtro=None):
    plt.figure(figsize=(10, 5))
    plt.plot(original, label="Original")
    plt.plot(estimado, label="Estimado")
    if limite_filtro is not None:
        plt.axvline(limite_filtro, color="black", linestyle="--", linewidth=1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.title("Original x Estimado")
    plt.xlabel("Amostra")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Exemplo 1: uma execucao
    np.random.seed(123)
    N = 50
    sinal_original, readout = gerador_sinal_entrada_saida(N)

    melhor_ordem = busca_ordem_otima_filtro(
        ordem_mais_alta=20,
        signal_original=sinal_original,
        filtro=filtro_LS_novo,
        modo_comparacao="seu",  # troque para "justo" se quiser
        readout=readout,
    )
    ordem = melhor_ordem["Ordem_Filtro_RMSE"]

    melhor_delay = busca_delay_otimo(
        delays=5,
        signal_original=sinal_original,
        filtro=filtro_LS_novo,
        readout=readout,
        ordem_filter=ordem,
    )
    delay = melhor_delay["Delay_RMSE"]

    sinal_est = filtro_LS_novo(
        sinal_desejado=sinal_original,
        readout=readout,
        ordem_filter=ordem,
        delay=delay,
    )

    max_i = N - ordem + 1
    rmse, mae = RMSE_e_MAE_por_ordem(
        sinal_est[:max_i], sinal_original[delay : delay + max_i]
    )

    print("Melhor ordem:", ordem, "| Melhor delay:", delay)
    print("RMSE final:", round(float(rmse), 6), "| MAE final:", round(float(mae), 6))

    # plot_estimado_x_original(
    #     original=sinal_original,
    #     estimado=sinal_est,
    #     limite_filtro=(N - ordem + 1),
    # )

    # Exemplo 2: Monte Carlo reprodutivel
    resultado = experimento_monte_carlo(
        n_execucoes=100,
        seed_inicial=42,
        qntd_amostras=50,
        ordem_max=20,
        delays=5,
        modo_comparacao_ordem="seu",
    )
    print("\nResumo Monte Carlo:")
    print(resultado)
