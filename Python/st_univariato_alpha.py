# GRID alpha — ST univariata: fit, inversione CDF↔PPF, UNIF, timing
# Dipendenze: numpy, pandas, scipy, skewt-scipy

import os, time
import numpy as np
import pandas as pd
from scipy import stats

# import skewt-scipy
try:
    from skewt_scipy.skewt import skewt
except Exception as e:
    raise ImportError(
        "Impossibile importare 'skewt' dal pacchetto 'skewt-scipy'. "
        "Installa/aggiorna con: pip install -U skewt-scipy"
    ) from e

# Griglia di valori di alpha per cui leggere i dataset e fare il fit.
alpha_grid = [-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0]  # come SN

# Directory base per dati, tabelle e log (coerenti con gli script R).
DATA_DIR = "common/data"
TAB_DIR  = "common/tables"
LOG_DIR  = "common/logs"

# Crea le cartelle per tabelle e log se non esistono già.
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Percorsi output complessivi (tabella risultati e tempi).
OUT_TAB = os.path.join(TAB_DIR, "st_uni_A_grid_Py.csv")
OUT_TIM = os.path.join(LOG_DIR, "st_uni_A_grid_times_Py.csv")

# Costruisce il percorso del file dati associato a un certo alpha.
# Esempio: alpha=-1.2 → "st_uni_A_alpha_-1.2" ma con il punto sostituito da "p".
def data_path_for_alpha(a: float) -> str:
    tag = str(round(a, 1)).replace(".", "p")
    return os.path.join(DATA_DIR, f"st_uni_A_alpha_{tag}.csv")

# Legge la colonna 'y' dal CSV corrispondente a un certo alpha e la restituisce come array NumPy.
def read_y(a: float) -> np.ndarray:
    fpath = data_path_for_alpha(a)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Dataset mancante per alpha={a} → {fpath}\n"
            "Generalo prima (RST in R) o crea il CSV con una colonna 'y'."
        )
    return pd.read_csv(fpath)["y"].to_numpy()

# Piccola utility per avere un timestamp ad alta risoluzione (per timing).
def now(): 
    return time.perf_counter()

print("=== ST univariata — GRID alpha (Python) — start ===", flush=True)

# Liste che accumulano risultati (righe) e tempi (per ogni alpha).
rows, times = [], []

# Loop principale sulla griglia di alpha.
for a in alpha_grid:
    # (A) Lettura dati per questo valore di alpha
    print(f"\n[alpha={a:+.1f}] Carico dati…", flush=True)
    y = read_y(a)
    n = y.size
    print(f"  → n = {n}", flush=True)

    # (B) Fit ST (stima alpha, nu, xi, omega con skewt.fit)
    print("  Fit skew-t…", flush=True)
    t0 = now()
    # skewt.fit restituisce: a≡alpha, df≡nu, loc≡xi, scale≡omega
    a_hat, df_hat, loc_hat, scale_hat = skewt.fit(y)
    t_fit = now() - t0
    print(
        f"  → stime: alpha={a_hat:.6g}, nu={df_hat:.6g}, xi={loc_hat:.6g}, omega={scale_hat:.6g}  "
        f"(fit {t_fit:.3f}s)",
        flush=True
    )

    # (C) Log-likelihood al massimo
    # Calcola la somma dei log-densità della skew-t stimata, per avere loglik.
    loglik = float(np.sum(skewt.logpdf(y, a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)))

    # (D) Inversione CDF↔PPF (core + code)
    # Costruisce una griglia di probabilità:
    # - p_core: parte centrale (0.001–0.999)
    # - p_ext: code estreme (1e-6, 1e-5, 1e-4, 1-1e-4, 1-1e-5, 1-1e-6)
    # Poi:
    #   q_inv  = PPF(p_grid)
    #   p_back = CDF(q_inv)
    # e misura l'errore massimo in core ed estremi separatamente.
    p_core = np.linspace(0.001, 0.999, 999)
    p_ext  = np.array([1e-6, 1e-5, 1e-4, 1-1e-4, 1-1e-5, 1-1e-6])
    p_grid = np.concatenate([p_ext[:3], p_core, p_ext[3:]])
    q_inv  = skewt.ppf(p_grid, a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)
    p_back = skewt.cdf(q_inv,   a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)

    # Errore massimo nel core (escludendo 3 punti di coda iniziali e finali).
    inv_core = float(np.max(np.abs(p_back[3:-3] - p_core)))
    # Errore massimo nelle code (primi 3 e ultimi 3 punti).
    inv_ext  = float(np.max(np.abs(np.concatenate([p_back[:3], p_back[-3:]]) - p_ext)))

    # (E) UNIF
    # Calcola u = F(y; theta_hat) (PIT) e applica il test KS contro Uniform(0,1).
    u = skewt.cdf(y, a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)
    ks_p = float(stats.kstest(u, "uniform", args=(0, 1)).pvalue)

    # (F) Accumulo risultati in memoria
    # row: contiene stime, loglik, misure di inversione e p-value KS per questo alpha.
    rows.append({
        "engine": "Python",
        "alpha_true": a,
        "xi_hat": float(loc_hat),
        "omega_hat": float(scale_hat),
        "alpha_hat": float(a_hat),
        "nu_hat": float(df_hat),
        "loglik": loglik,
        "n": int(n),
        "inv_core_maxerr": inv_core,
        "inv_ext_maxerr": inv_ext,
        "ks_pvalue": ks_p,
    })
    # times: logga solo il tempo di fit per scenario (alpha).
    times.append({"alpha_true": a, "step": "fit_skewt", "seconds": float(t_fit)})

# Converte la lista di dizionari in DataFrame e salva la tabella finale.
pd.DataFrame(rows).to_csv(OUT_TAB, index=False)
pd.DataFrame(times).to_csv(OUT_TIM, index=False)

print(f"\n✓ Scritto: {OUT_TAB}")
print(f"✓ Scritto: {OUT_TIM}")
print("\n=== ST univariata — GRID alpha (Python) — done ===")
