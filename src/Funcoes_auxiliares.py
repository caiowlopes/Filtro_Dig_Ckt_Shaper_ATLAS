"""Fuinções auxiliares para os Filtros"""

import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


# Funções auxiliares #
# Plot comparação
def plot_estimado_x_original(
    original: np.ndarray,
    estimado: np.ndarray,
    limite_filtro: int | float,
    xlimite_min: int | float = 0,
    xlimite_max: int | float = 0,
    title: str = "Original x Estimado",
):
    """
    limite_filtro= qntd_amostra - ordem + 1
    """
    # Plot
    plt.figure(figsize=(10, 6))

    if xlimite_max == 0:
        xlimite_max = 1.01 * len(original)

    plt.xlim(xlimite_min, xlimite_max)
    plt.title(title)
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")

    plt.plot(original)
    plt.plot(estimado)
    plt.axvline(limite_filtro, color="black", linestyle="--", linewidth=1.0)

    plt.legend(["Original", "Estimado", "Limite Estimativa"], loc="upper right")

    plt.grid()
    plt.show()


# Comparação numerica
def RMSE_e_MAE_por_ordem(
    A: np.ndarray, B: np.ndarray, limite_filtro: int | float = 0, printar: bool = False
):
    """
    Calcula o RMSEW e o MAE. Ambos podem ser: np.array | int | floats...

    limite = quantidade_de_amostras - ordem_filtro + 1

    RMSE: root mean squared error
    MAE: mean absolute error

    Caso requisitado, os resultados são imprimidos.
    """
    diff = A[:limite_filtro] - B[:limite_filtro] if limite_filtro != 0 else A - B

    rmse = np.sqrt(np.mean(diff**2))
    erro_abs_medio = np.mean(np.abs(diff))

    if printar:
        print(f"{erro_abs_medio = :.4f}")
        print(f"{rmse = :.4f}")

    return rmse, erro_abs_medio


# Função matriz de Observação
def matriz_observacao(sinal: list | np.ndarray, ordem_filtro: int = 2):
    """
    Constrói a matriz de observação a partir do sinal de entrada utilizando
    janelas deslizantes de tamanho igual à ordem do filtro.
    """
    return np.lib.stride_tricks.sliding_window_view(np.array(sinal), ordem_filtro)


# Busca melhor ordem do filtro #
# Busca considerando janelas diferentes, cada ordem com a sua
def busca_ordem_otima_filtro(
    signal_original: np.ndarray,
    filtro: Callable,
    ordem_mais_alta: int,
    step: int = 1,
    tamanho_janela_fixo: bool = False,
    delay: int = 2,
    **filtro_kwargs,
):
    """
    Esse filtro não aceita ordem menor do que 3.

    Parametros:
    -----------
    ordem_mais_alta: int, obrigatorio.
        Valor da ordem mais alta a ser testada.
    """

    melhor_ordem_dict = {
        "Ordem_Filtro_RMSE": 0,
        "RMSE": np.inf,
        "Ordem_Filtro_MAE": 0,
        "MAE": np.inf,
    }

    if ordem_mais_alta < 3:
        ordem_mais_alta = 3

    for ordem_filter in range(3, ordem_mais_alta + 1, step):

        sinal = filtro(
            sinal_desejado=signal_original,
            ordem_filter=ordem_filter,
            **filtro_kwargs,
        )

        if tamanho_janela_fixo:
            # Considerando a mesma janela para todos
            # Janela fixa para todas as ordens candidatas
            janela = len(signal_original) - ordem_mais_alta + 1

        else:
            # Cada ordem tem uma janela proporcional
            janela = len(signal_original) - ordem_filter + 1

        if janela <= 0:
            raise ValueError(
                "Sinal curto demais para a ordem_mais_alta informada. "
                "É necessário len(signal_original) - ordem_mais_alta + 1 > 0."
            )

        estimado_eval = sinal[:janela]
        original_eval = signal_original[delay : delay + janela]

        rmse, mae = RMSE_e_MAE_por_ordem(estimado_eval, original_eval)

        if rmse < melhor_ordem_dict["RMSE"]:
            melhor_ordem_dict["RMSE"] = round(rmse, 10)
            melhor_ordem_dict["Ordem_Filtro_RMSE"] = ordem_filter

        if mae < melhor_ordem_dict["MAE"]:
            melhor_ordem_dict["MAE"] = round(mae, 10)
            melhor_ordem_dict["Ordem_Filtro_MAE"] = ordem_filter

    return melhor_ordem_dict


# Busca melhor delay do filtro #
def busca_delay_otimo(
    signal_original: np.ndarray,
    filtro: Callable,
    delay_maximo: int = 7,
    **filtro_kwargs,
):
    """
    melhor = {
        "Delay_RMSE": Valor do delay pelo critério RMSE,
        "RMSE": Valor da medida com dado delay,
        "Delay_MAE": Valor do delay pelo critério MAE,
        "MAE": Valor da medida com dado delay,
    }
    """
    melhor = {
        "Delay_RMSE": None,
        "RMSE": np.inf,
        "Delay_MAE": None,
        "MAE": np.inf,
    }

    for delay in range(delay_maximo):
        sinal = filtro(
            sinal_desejado=signal_original,
            delay=delay,
            **filtro_kwargs,
        )
        rmse, mae = RMSE_e_MAE_por_ordem(sinal, signal_original)

        if rmse < melhor["RMSE"]:
            melhor["RMSE"] = rmse
            melhor["Delay_RMSE"] = delay

        if mae < melhor["MAE"]:
            melhor["MAE"] = mae
            melhor["Delay_MAE"] = delay

    melhor["RMSE"] = melhor["RMSE"]
    melhor["MAE"] = melhor["MAE"]
    return melhor


# Função para fazer a busca de melhor ordem e melhor delay
def grid_search_ordem_delay_otimos(
    filtro,
    sinal_desejado,
    readout,
    ordem_maxima: int = 30,
    delays=range(0, 13),
    criterio="rmse",  # "rmse" ou "mae"
    tamanho_janela_fixo: bool = True,
    **filtro_kwargs,
):
    sinal_desejado = np.asarray(sinal_desejado)
    resultados = []

    melhor = {
        "ordem": None,
        "delay": None,
        "rmse": np.inf,
        "mae": np.inf,
    }

    for ordem in range(3, ordem_maxima):
        for delay in delays:
            try:
                sinal_estimado = filtro(
                    sinal_desejado=sinal_desejado,
                    readout=readout,
                    ordem_filter=ordem,
                    delay=delay,
                    **filtro_kwargs,
                )
            except ValueError:
                resultados.append(
                    {
                        "ordem": ordem,
                        "delay": delay,
                        "rmse": "ValueError",
                        "mae": "ValueError",
                    }
                )
                continue

            # n_valid = min(len(readout) - ordem + 1, len(sinal_desejado) - delay)

            # if n_valid <= 0:
            #     resultados.append(
            #         {
            #             "ordem": ordem,
            #             "delay": delay,
            #             "rmse": "n_invalid",
            #             "mae": "n_invalid",
            #         }
            #     )
            #     continue

            # est_eval = np.asarray(sinal_estimado)[:n_valid]
            # des_eval = sinal_desejado[delay : delay + n_valid]

            if not tamanho_janela_fixo:
                # Cada ordem tem uma janela proporcional
                janela = len(sinal_desejado) - ordem + 1
            else:
                # Considerando a mesma janela para todos
                # Janela fixa para todas as ordens candidatas
                janela = len(sinal_desejado) - ordem_maxima + 1

            if janela <= 0:
                raise ValueError(
                    "Sinal curto demais para a ordem_mais_alta informada. "
                    "É necessário len(signal_original) - ordem_mais_alta + 1 > 0."
                )

            est_eval = sinal_estimado[:janela]
            des_eval = sinal_desejado[delay : delay + janela]

            rmse, mae = RMSE_e_MAE_por_ordem(est_eval, des_eval)

            resultados.append(
                {"ordem": ordem, "delay": delay, "rmse": rmse, "mae": mae}
            )

            score_atual = rmse if criterio.lower() == "rmse" else mae
            score_melhor = (
                melhor["rmse"] if criterio.lower() == "rmse" else melhor["mae"]
            )

            if score_atual < score_melhor:
                melhor = {"ordem": ordem, "delay": delay, "rmse": rmse, "mae": mae}

    return melhor, resultados
