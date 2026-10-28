"""Решение ДЗ-1: модели Дебая, Лоренца, Друде.

Запуск из корня репозитория:
    python dz1/src/solution.py --dz dz1 --variant 0           # вывести результаты
    python dz1/src/solution.py --dz dz1 --variant 0 --save    # + таблица CSV и графики в dz1/results/

Соглашение о знаках: поле ~ exp(j*omega*t), проницаемость eps = eps' - j*eps'',
потери дают eps'' > 0. Источники констант - SOURCES.md.
"""
import argparse
import csv
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTROL = json.loads((ROOT / "tests" / "control_answers.json").read_text(encoding="utf-8"))

# Фундаментальные константы, CODATA 2022 (NIST), см. SOURCES.md [S1]
E_CHARGE = 1.602176634e-19      # Кл, точное значение
M_ELECTRON = 9.1093837139e-31   # кг
EPS0 = 8.8541878188e-12         # Ф/м
C_LIGHT = 299792458.0           # м/с, точное значение

NP_TO_DB = 20 / np.log(10)      # 8,686 дБ/Нп


# ---------------------------------------------------------------- варианты

def get_variant(dz, n):
    """Исходные данные варианта n задания dz (словарь)."""
    if n == 0:
        return dict(CONTROL[dz]["variant"])
    if dz == "dz1":
        return {
            "tau_ps": 5 + 0.2 * n,
            "f0": 55 + n % 11,                 # ГГц
            "dfL": 0.5 + 0.25 * (n % 5),       # ГГц
            "deps": 1e-4,
            "N_F": (0.4 + 0.05 * n) * 1e12,    # м^-3
            "nu_F": 1e3,                       # с^-1
            "N_D": (0.5 + 0.1 * (n % 10)) * 1e9,
            "nu_D": 1e7,
            "f_test_MHz": [1, 3, 5, 10, 30],
        }
    raise NotImplementedError("Формулы вариантов ещё не добавлены")


# ---------------------------------------------------------------- модели

def debye(f_hz, tau_s):
    """Нормированный Дебай (eps_ст - eps_вч = 1): возвращает eps' - eps_вч и eps''."""
    wt = 2 * np.pi * f_hz * tau_s
    return 1 / (1 + wt**2), wt / (1 + wt**2)


def lorentz(f_hz, f0_hz, nu, deps):
    """Лоренц из уравнения m x'' + m nu x' + m w0^2 x = -eE.

    eps = 1 + wp^2 / (w0^2 - w^2 + j w nu), сила линии deps = wp^2 / w0^2.
    Возвращает комплексную eps (eps' - j eps'').
    """
    w, w0 = 2 * np.pi * f_hz, 2 * np.pi * f0_hz
    return 1 + deps * w0**2 / (w0**2 - w**2 + 1j * w * nu)


def plasma_freq(N):
    """Плазменная частота f_p = w_p / 2pi, Гц."""
    return np.sqrt(N * E_CHARGE**2 / (EPS0 * M_ELECTRON)) / (2 * np.pi)


def drude(f_hz, N, nu):
    """Друде из уравнения m x'' + m nu x' = -eE: eps = 1 - wp^2 / (w (w - j nu))."""
    w = 2 * np.pi * f_hz
    wp2 = (2 * np.pi * plasma_freq(N))**2
    return 1 - wp2 / (w * (w - 1j * nu))


def fwhm(f, y):
    """Ширина пика на половине высоты с линейной интерполяцией краёв."""
    i = int(np.argmax(y))
    half = y[i] / 2
    left = np.where(y[:i] < half)[0][-1]
    right = i + np.where(y[i:] < half)[0][0]
    fl = np.interp(half, [y[left], y[left + 1]], [f[left], f[left + 1]])
    fr = np.interp(half, [y[right], y[right - 1]], [f[right], f[right - 1]])
    return fr - fl


# Линии кислорода 54–67 ГГц: f0 (ГГц) и a3 (a4 = 0) из Rec. ITU-R P.676-13, Annex 1, Table 1 [S3]
O2_LINES = [
    (54.130025, 9.960), (54.671180, 10.370), (55.221384, 10.890), (55.783815, 11.340),
    (56.264774, 17.030), (56.363399, 11.890), (56.968211, 12.230), (57.612486, 12.620),
    (58.323877, 12.950), (58.446588, 14.910), (59.164204, 13.530), (59.590983, 14.080),
    (60.306056, 14.150), (60.434778, 13.390), (61.150562, 12.920), (61.800158, 12.630),
    (62.411220, 12.170), (62.486253, 15.130), (62.997984, 11.740), (63.568526, 11.340),
    (64.127775, 10.880), (64.678910, 10.380), (65.224078, 9.960), (65.764779, 9.550),
    (66.302096, 9.060), (66.836834, 8.580),
]

# Атмосфера для ширин линий: стандартная атмосфера Rec. ITU-R P.835 [S4]
GROUND = {"p_tot": 1013.25, "T": 288.15, "rho": 7.5}   # гПа, К, г/м³
H20KM = {"p_tot": 55.29, "T": 216.65, "rho": 0.0}      # на 20 км пар пренебрежимо мал


def o2_halfwidth(a3, p_tot, T, rho, a4=0.0):
    """Полуширина линии O2 по P.676-13, (4), (6a), (6b): ГГц.

    p_tot - полное давление, гПа; давление сухого воздуха p = p_tot - e,
    e = rho*T/216.7 (4); Zeeman-поправка sqrt(df^2 + 2.25e-6).
    """
    theta = 300 / T
    e = rho * T / 216.7
    df = a3 * 1e-4 * ((p_tot - e) * theta**(0.8 - a4) + 1.1 * e * theta)
    return np.sqrt(df**2 + 2.25e-6)


def regime(f_hz, eps1, nu):
    """Режим по разделу 1.5: поглощение (w < nu), отражение (eps' < 0), прозрачность."""
    if 2 * np.pi * f_hz < nu:
        return "поглощение"
    return "отражение" if eps1 < 0 else "прозрачность"


# ---------------------------------------------------------------- решение

def solve_dz1(v):
    r = {}
    tau = v["tau_ps"] * 1e-12

    # 1. Дебай: сетка 0,1–100 ГГц с шагом 1 МГц
    f = np.arange(0.1e9, 100e9 + 1, 1e6)
    _, e2 = debye(f, tau)
    i = int(np.argmax(e2))
    r["debye_fmax_num"] = f[i] / 1e9
    r["debye_fmax_an"] = 1 / (2 * np.pi * tau) / 1e9       # (1.8): w*tau = 1
    r["debye_eppmax"] = float(e2[i])

    # 2. Лоренц: f0 ± 10 ГГц с шагом 0,1 МГц, nu = 2*pi*dfL
    f0, nu = v["f0"] * 1e9, 2 * np.pi * v["dfL"] * 1e9
    f = np.arange(f0 - 10e9, f0 + 10e9 + 1, 0.1e6)
    e2 = -lorentz(f, f0, nu, v["deps"]).imag
    i = int(np.argmax(e2))
    r["lor_fpeak"] = f[i] / 1e9
    r["lor_eppmax"] = float(e2[i])
    r["lor_fwhm"] = fwhm(f, e2) / 1e9
    r["lor_eppmax_est"] = v["deps"] * v["f0"] / v["dfL"]    # оценка deps*f0/dfL
    r["lor_eps2_min"] = float(e2.min())                     # проверка знака: > 0

    # 3. Друде: слои F и D на пробных частотах
    for layer in ("F", "D"):
        N, nu = v[f"N_{layer}"], v[f"nu_{layer}"]
        r[f"fp_{layer}_MHz"] = plasma_freq(N) / 1e6
        for fm in v["f_test_MHz"]:
            fh = fm * 1e6
            eps = drude(fh, N, nu)
            e1, e2 = eps.real, -eps.imag
            s = np.sqrt(eps)                                # главная ветвь: n - j*kappa
            n_, kappa = s.real, -s.imag
            key = f"{layer}_{fm:g}MHz"
            r[f"{key}_eps1"] = float(e1)
            r[f"{key}_eps2"] = float(e2)
            r[f"{key}_tand"] = float(abs(e2 / e1))          # (1.3)
            r[f"{key}_n"] = float(n_)
            r[f"{key}_kappa"] = float(kappa)
            r[f"{key}_att_dBkm"] = float(NP_TO_DB * 2 * np.pi * fh / C_LIGHT * kappa * 1e3)
            r[f"{key}_regime"] = regime(fh, e1, nu)

    # 4б. Три ближайшие к f0 линии кислорода и их ширины (FWHM = 2 * полуширина)
    near = sorted(O2_LINES, key=lambda line: abs(line[0] - v["f0"]))[:3]
    for i, (fl, a3) in enumerate(sorted(near), 1):
        w0 = 2 * o2_halfwidth(a3, **GROUND)
        w20 = 2 * o2_halfwidth(a3, **H20KM)
        r[f"o2_{i}_f0"] = fl
        r[f"o2_{i}_fwhm_0km"] = float(w0)
        r[f"o2_{i}_fwhm_20km"] = float(w20)
        r[f"o2_{i}_dfL_ratio"] = float(v["dfL"] / w0)      # во сколько раз dfL шире реальной
    return r


def solve_dz2(v):
    # TODO: проницаемость воды, спектр капель, удельное затухание
    raise NotImplementedError


def solve_dz3(v):
    # TODO: лёд, снег, град
    raise NotImplementedError


def solve_dz4(v):
    # TODO: геометрия трассы и рефракция
    raise NotImplementedError


def solve_dz5(v):
    # TODO: профили атмосферы и бюджет потерь
    raise NotImplementedError


def solve_dz6(v, ref_inputs=None):
    """v: city, f, el, tau, hs. ref_inputs: эталонные данные карт (R001, P0, R01, R1, hR, lat, lon).

    Если ref_inputs не None, используйте их вместо собственных найденных значений:
    так проверяется сам расчёт, а не выбор источников.
    """
    # TODO: метод P.618, подгонка, разнесённый приём
    raise NotImplementedError


SOLVERS = {"dz1": solve_dz1, "dz2": solve_dz2, "dz3": solve_dz3,
           "dz4": solve_dz4, "dz5": solve_dz5, "dz6": solve_dz6}


def solve(dz, n, ref_inputs=None):
    v = get_variant(dz, n)
    if dz == "dz6":
        return SOLVERS[dz](v, ref_inputs)
    return SOLVERS[dz](v)


# ---------------------------------------------------------------- вывод

def save_dz1(n, r):
    """Таблица результатов и три графика варианта n в dz1/results/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    v = get_variant("dz1", n)

    with open(results / f"dz1_variant{n}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["key", "value"])
        for k, val in r.items():
            w.writerow([k, f"{val:.6g}" if isinstance(val, float) else val])

    # Дебай
    tau = v["tau_ps"] * 1e-12
    f = np.logspace(np.log10(0.1e9), np.log10(100e9), 2000)
    e1, e2 = debye(f, tau)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(f / 1e9, e1, label=r"$\varepsilon' - \varepsilon_{вч}$")
    ax.semilogx(f / 1e9, e2, label=r"$\varepsilon''$")
    ax.axvline(r["debye_fmax_an"], color="gray", ls="--", lw=1,
               label=f"$f_{{max}}$ = {r['debye_fmax_an']:.3f} ГГц")
    ax.set(xlabel="Частота, ГГц", ylabel="Нормированная проницаемость",
           title=f"Модель Дебая, τ = {v['tau_ps']:g} пс (вариант {n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / f"dz1_debye_variant{n}.png", dpi=150)
    plt.close(fig)

    # Лоренц
    f0, nu = v["f0"] * 1e9, 2 * np.pi * v["dfL"] * 1e9
    f = np.arange(f0 - 10e9, f0 + 10e9 + 1, 0.1e6)
    e2 = -lorentz(f, f0, nu, v["deps"]).imag
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(f / 1e9, e2 * 1e3, label=r"$\varepsilon''$")
    ax.axhline(r["lor_eppmax"] / 2 * 1e3, color="gray", ls="--", lw=1,
               label=f"половина высоты, FWHM = {r['lor_fwhm']:.4f} ГГц")
    ax.set(xlabel="Частота, ГГц", ylabel=r"$\varepsilon''\cdot10^{3}$",
           title=f"Модель Лоренца, $f_0$ = {v['f0']:g} ГГц, $\\Delta f_L$ = {v['dfL']:g} ГГц (вариант {n})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig(results / f"dz1_lorentz_variant{n}.png", dpi=150)
    plt.close(fig)

    # Друде, слой F
    f = np.linspace(1e6, 30e6, 3000)
    s = np.sqrt(drude(f, v["N_F"], v["nu_F"]))
    fp = r["fp_F_MHz"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(f / 1e6, s.real, label=r"$n = \mathrm{Re}\sqrt{\hat\varepsilon}$")
    ax.plot(f / 1e6, -s.imag, ls="--", label=r"$\kappa = -\mathrm{Im}\sqrt{\hat\varepsilon}$")
    ax.axvline(fp, color="gray", ls=":", lw=1, label=f"$f_p$ = {fp:.3f} МГц")
    ax.set(xlabel="Частота, МГц", ylabel="Показатель преломления",
           title=f"Модель Друде, слой F, $N_F$ = {v['N_F']:.3g} м⁻³ (вариант {n})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / f"dz1_drude_F_variant{n}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz", default="dz1")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="сохранить таблицу и графики в dz1/results/")
    a = ap.parse_args()
    ref = CONTROL.get(a.dz, {}).get("inputs") if a.variant == 0 else None
    res = solve(a.dz, a.variant, ref)
    for key, value in res.items():
        print(f"{key}: {value}")
    if a.save and a.dz == "dz1":
        save_dz1(a.variant, res)
