import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import remez
from scipy.signal import lfilter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def parks_mcclellan_filter(
    fs: float = 20000,  # frequência de amostragem (Hz)
    passband: tuple = (0, 2000),  # banda útil do sinal
    stopband: tuple = (4000, None),  # banda onde o ruído domina
    num_taps: int = 101,  # número de coeficientes (ímpar p/ fase linear)
    weight_pass: float = 1.0,  # peso do erro na banda passante
    weight_stop: float = 10.0,  # peso do erro na banda de rejeição
    plot_response: bool = True,  # mostrar resposta em frequência
):
    """
    Projeto de filtro FIR equiripple usando Parks-McClellan (Remez).
    Todos os parâmetros possuem valores padrão coerentes
    com simulações típicas de circuito shaper.
    """

    # Se stopband superior não for definido, usa Nyquist
    if stopband[1] is None:
        stopband = (stopband[0], fs / 2)

    # ==========================================================
    # 1) Definição das bandas
    # ==========================================================
    # As bandas devem estar em ordem crescente:
    # [stop_low, stop_high, pass_low, pass_high]
    # Neste caso estamos fazendo um passa-baixa.
    bands = [stopband[0], stopband[1], passband[0], passband[1]]

    # ==========================================================
    # 2) Ganho desejado em cada banda
    # ==========================================================
    # 0 → rejeitar
    # 1 → manter
    desired = [0, 1]

    # ==========================================================
    # 3) Peso do erro em cada banda
    # ==========================================================
    # Quanto maior o peso, menor o ripple naquela banda.
    weights = [weight_stop, weight_pass]

    # ==========================================================
    # 4) Cálculo dos coeficientes via Remez (Projeto do filtro)
    # ==========================================================
    h = remez(num_taps, bands, desired, weight=weights, fs=fs)

    # ==========================================================
    # 5) Resposta em frequência
    # ==========================================================
    if plot_response:
        w, H = freqz(h, worN=4096, fs=fs)

        plt.figure()
        plt.plot(w, 20 * np.log10(np.abs(H) + 1e-12))
        plt.title("Resposta em Frequência - Parks–McClellan")
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid()
        plt.show()

    return h


if __name__ == "__main__":

    # 1) Sinal original (puro)
    fs = 20000
    t = np.linspace(0, 0.1, int(0.1 * fs))

    original = np.exp(-40 * t) * np.sin(2 * np.pi * 1200 * t)

    # 2) Modelo simplificado do shaper
    # (exemplo: resposta exponencial)
    shaper_response = np.exp(-200 * t)
    shaped = np.convolve(original, shaper_response, mode="same")

    # 3) Adição de ruído do circuito de leitura
    noise = 0.05 * np.random.randn(len(shaped))
    reading_signal = shaped + noise

    # 4) Projeto e aplicação do filtro
    h = parks_mcclellan_filter()

    filtered = lfilter(h, 1.0, reading_signal)

    # Compensação do atraso linear do FIR
    delay = (len(h) - 1) // 2
    filtered = filtered[delay:]
    original_cut = original[: len(filtered)]

    # 5) Comparação com o sinal puro
    rmse = np.sqrt(mean_squared_error(original_cut, filtered))  # type: ignore
    mae = mean_absolute_error(original_cut, filtered)  # type: ignore

    print("RMSE:", rmse)
    print("MAE :", mae)
