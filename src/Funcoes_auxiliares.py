"""Fuinções auxiliares para os Filtros"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


# Funções auxiliares #
# Classifica valores por ranges
def classificar_lista_por_ranges(
    valores: list | np.ndarray,
    ranges: dict[str, tuple[float, float]],
    pico_min: float,
    label_parado: str = "parado",
    label_subindo: str = "subindo",
    label_pico: str = "pico",
    label_descendo: str = "descendo",
):
    """
    Substitui cada valor por um rotulo baseado em ranges.

    Regras:
    - Se o valor estiver no range de 'pico' e for >= pico_min, rotula como 'pico'.
    - Se estiver no range de 'pico' mas < pico_min, rotula como 'parado'.
    - Caso contrario, usa o primeiro range que casar (ordem do dict).
    - Se nenhum range casar, rotula como 'parado'.
    """
    if label_pico not in ranges:
        raise ValueError("ranges deve conter a chave 'pico'.")

    resultado = []

    for v in np.asarray(valores):
        rotulo = label_parado

        pico_min_range, pico_max_range = ranges[label_pico]
        if pico_min_range <= v <= pico_max_range:
            rotulo = label_pico if v >= pico_min else label_parado
            resultado.append(rotulo)
            continue

        for nome, (vmin, vmax) in ranges.items():
            if nome == label_pico:
                continue
            if vmin <= v <= vmax:
                if nome == "parado":
                    rotulo = label_parado
                elif nome == "subindo":
                    rotulo = label_subindo
                elif nome == "descendo":
                    rotulo = label_descendo
                else:
                    rotulo = nome
                break

        resultado.append(rotulo)

    return resultado


# Plot comparação
def plot_estimado_x_original(
    original: np.ndarray,
    estimado: np.ndarray,
    ordem: int = 7,
    xlimite_min: int | float = 0,
    xlimite_max: int | float = 0,
    title: str = "Original x Estimado",
    limite_filtro: bool = False,
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
    if limite_filtro:
        lim_filtro = len(original) - ordem + 1
        plt.axvline(lim_filtro, color="black", linestyle="--", linewidth=1.0)

    plt.legend(["Original", "Estimado", "Limite Estimativa"], loc="upper right")

    plt.grid()
    plt.show()


# Comparação numerica
def RMSE_e_MAE_por_ordem(
    A: np.ndarray, B: np.ndarray, ordem_filtro: int | float = 0, printar: bool = False
):
    """
    Calcula o RMSEW e o MAE. Ambos podem ser: np.array | int | floats...

    limite = quantidade_de_amostras - ordem_filtro + 1

    RMSE: root mean squared error
    MAE: mean absolute error

    Caso requisitado, os resultados são imprimidos.
    """
    limite_filtro = len(A) - ordem_filtro + 1

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

    Exemplo:
    sinal = [1,2,3,4,5]
    ordem_filtro = 3

    saida = [[1 2 3]
            [2 3 4]
            [3 4 5]]
    """
    return np.lib.stride_tricks.sliding_window_view(np.array(sinal), ordem_filtro)


# Busca melhor ordem do filtro #
# Busca considerando janelas diferentes, cada ordem com a sua
def busca_ordem_otima_filtro(
    signal_original: np.ndarray,
    filtro: Callable,
    ordem_mais_alta: int,
    step: int = 1,
    tamanho_janela_fixo: bool = True,
    delay: int = 2,
    **filtro_kwargs,
):
    """
    Esse filtro não aceita ordem menor do que 3.

    Parametros:
    -----------
    ordem_mais_alta: int, obrigatorio.
        Valor da ordem mais alta a ser testada.

    Retorno:
    --------
    Retorna um dicionario com as melhores ordens baseado nos metodos RMSE e MAE.
    Retorna um historico com todas as ordens testadas e uma tupla com as metricas RMSE e MAE, nessa ordem.
    """
    historico = {}
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

        rmse, mae = RMSE_e_MAE_por_ordem(
            estimado_eval, original_eval  # , limite_filtro=janela
        )

        if rmse < melhor_ordem_dict["RMSE"]:
            melhor_ordem_dict["RMSE"] = round(rmse, 10)
            melhor_ordem_dict["Ordem_Filtro_RMSE"] = ordem_filter

        if mae < melhor_ordem_dict["MAE"]:
            melhor_ordem_dict["MAE"] = round(mae, 10)
            melhor_ordem_dict["Ordem_Filtro_MAE"] = ordem_filter

        historico[f"{ordem_filter}"] = [rmse, mae]

    return historico, melhor_ordem_dict


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


# Aplicando os pesos e bias já calculados em um sinal novo #
def estimado_com_pesos_ja_calculados(
    pesos,
    bias,
    Readout,
    ordem_filtro: int = 7,
):  # , seed=None
    """
    Aplica pesos e bias previamente calculados em uma entrada nova.

    O obetivo é verificar a capacidade de generalizaçao do filtro.
    """
    # sinal novo
    # sinal_original_2 = original_signal_generator(quantidade_de_amostras, seed=seed)

    # leitura sinal novo
    # Readout_Shaper_2 = leitura_shaper(sinal_original_2, seed=seed)

    # aplicando filtro #
    qntd_leitura = len(Readout)

    # Sinal Estimado/Recuperado
    sinal_estimado_2 = np.zeros(qntd_leitura, dtype=float)

    # Tamanho do sinal estimado
    len_sinal_estimado = qntd_leitura - ordem_filtro + 1

    # Parte adaptativa do filtro
    for i in range(len_sinal_estimado):
        sinal_estimado_2[i] = np.dot(Readout[i : i + ordem_filtro], pesos) + bias

    sinal_estimado_2 = np.clip(sinal_estimado_2, 0, None)
    return sinal_estimado_2
