"""ДЗ-1, п. 4: ионосфера по модели IRI-2020 и проверка частот соударений nu_D и nu_F.

1. График профилей N_e(h) днём и ночью по выходным файлам IRI-2020 [S7].
2. Частоты соударений электронов:
   слой D - с нейтралами, по Schunk & Nagy [S8] и для сравнения по Davies (1.62a) [S6]:
       nu_en(N2) = 2.33e-11 n(N2) (1 - 1.21e-4 T) T         (n в см^-3, T в К)
       nu_en(O2) = 1.82e-10 n(O2) (1 + 3.6e-2 T^0.5) T^0.5
       nu_en     = 5e-16 n_n T^0.5                           (Davies, n_n в м^-3)
   p, T и n = p/(kT) - по стандартной атмосфере P.835-7 [S4]; в слое D T_e = T
   (электроны в тепловом равновесии с газом), доли N2 и O2 - U.S. Standard Atmosphere 1976 [S9];
   слой F - с ионами, nu_ei = 54.5 n_i / T_e^1.5, где N_e и T_e в пике F2 из IRI [S7].

Запуск из корня репозитория:  python dz1/src/ionosphere.py  ->  dz1/results/collision_check.csv, dz1_iri_profiles.png
"""
import csv
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]      # папка dz1
DATA = ROOT / "data"
RESULTS = ROOT / "results"
IRI_RUNS = (("день, 12:00 мск", "iri2020_moscow_2025-03-15_09UT.txt"),
            ("ночь, 00:00 мск", "iri2020_moscow_2025-03-15_21UT.txt"))

K_BOLTZMANN = 1.380649e-23           # Дж/К, CODATA 2022 (точное значение) [S1]
X_N2, X_O2 = 0.78084, 0.209476       # объёмные доли в сухом воздухе [S9]
R_EARTH_GP = 6356.766                # км, для перевода в геопотенциальную высоту, P.835-7 (1a)

# P.835-7, ф. (2a)–(2g), (3a)–(3g): (H0, T0, градиент K/км', P0 гПа)
P835_LAYERS = [(0, 288.15, -6.5, 1013.25), (11, 216.65, 0.0, 226.3226),
               (20, 216.65, 1.0, 54.74980), (32, 228.65, 2.8, 8.680422),
               (47, 270.65, 0.0, 1.109106), (51, 270.65, -2.8, 0.6694167),
               (71, 214.65, -2.0, 0.03956649)]


def p835(z_km):
    """Температура (К) и давление (гПа) на геометрической высоте z < 86 км по P.835-7."""
    h = R_EARTH_GP * z_km / (R_EARTH_GP + z_km)                       # (1a)
    h0, t0, lapse, p0 = [lay for lay in P835_LAYERS if lay[0] <= h][-1]
    t = t0 + lapse * (h - h0)
    if lapse == 0:
        p = p0 * math.exp(-34.1632 * (h - h0) / t0)
    else:
        p = p0 * (t0 / t) ** (34.1632 / lapse)
    return t, p


def read_iri(path):
    """Высота км -> (Ne см^-3, Te К) из стандартной таблицы IRI; Ne = -1 - нет данных."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) > 6:
            try:
                h, ne, te = float(parts[0]), float(parts[1]), float(parts[5])
            except ValueError:
                continue
            rows[h] = (ne, te)
    return rows


def nu_en(n_cm3, t):
    """Соударения электронов с N2 и O2 по Schunk & Nagy, с^-1."""
    return (2.33e-11 * X_N2 * n_cm3 * (1 - 1.21e-4 * t) * t
            + 1.82e-10 * X_O2 * n_cm3 * (1 + 3.6e-2 * t**0.5) * t**0.5)


def nu_en_davies(n_cm3, t):
    """Davies (1.62a): nu_en = 5e-16 n_n T^1/2, n_n в м^-3 [S6]. Показатель степени
    на скане читается не полностью; -16 - единственное значение, согласное с § 5.4 и рис. 7.5."""
    return 5e-16 * n_cm3 * 1e6 * t**0.5


def nu_ei(ne_cm3, te):
    """Кулоновские соударения электронов с ионами, с^-1 (n_i = Ne)."""
    return 54.5 * ne_cm3 / te**1.5


def collision_table():
    out = []
    # Слой D: только нейтральный газ, днём и ночью одинаково
    for z in (60, 70, 80, 90):
        t, p = p835(z)
        n_cm3 = p * 100 / (K_BOLTZMANN * t) * 1e-6
        out.append(["D", z, f"{n_cm3:.3g}", f"{t:.1f}", f"{nu_en(n_cm3, t):.3g}",
                    f"{nu_en_davies(n_cm3, t):.3g}"])
    # Слой F: пик F2 днём и ночью, кулоновские соударения
    for label, fname in IRI_RUNS:
        iri = read_iri(DATA / fname)
        z_peak = max(iri, key=lambda h: iri[h][0])
        ne, te = iri[z_peak]
        out.append([f"F2 (пик), {label}", z_peak, f"{ne:.3g}", f"{te:.0f}", f"{nu_ei(ne, te):.3g}", ""])
    return out


def plot_profiles():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for (label, fname), style in zip(IRI_RUNS, ("-", "--")):
        iri = read_iri(DATA / fname)
        hs = [h for h in sorted(iri) if iri[h][0] > 0]
        ax.semilogx([iri[h][0] * 1e6 for h in hs], hs, style, marker=".", ms=3, label=label)
    ax.set(xlabel="Концентрация электронов $N_e$, м⁻³", ylabel="Высота, км", ylim=(60, 1000),
           title="IRI-2020, Москва, 15.03.2025")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "dz1_iri_profiles.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = collision_table()
    with open(RESULTS / "collision_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["слой", "h_км", "n_см-3 (нейтралы для D, Ne для F)", "T_K",
                    "nu_с-1 (Schunk&Nagy)", "nu_с-1 (Davies 1.62a)"])
        w.writerows(rows)
    for row in rows:
        print(*row, sep="\t")
    plot_profiles()


if __name__ == "__main__":
    main()
