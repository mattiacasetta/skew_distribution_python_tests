# SN univariata — GRID su alpha (stress test)
# Per ogni valore di alpha nella griglia:
#   - legge il dataset generato in R (uno per alpha)
#   - stima i parametri SN via SciPy (DP: a, xi=loc, omega=scale)
#   - controlla l'accuratezza dell'inversione CDF↔PPF (core + code)
#   - valuta la trasformazione a uniformi (PIT) + test KS
#   - salva tabelle riepilogative (fit + tempi) per confronto R/Python

import os, numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize  # importato ma non usato qui (coerente con altri script)

# Griglia di valori di alpha vera utilizzata nella generazione dei dati (in R)
alpha_grid = [-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0]

# Directory per:
#   - dataset (già generati)
#   - tabelle di output
#   - log dei tempi
DATA_DIR = "common/data"
TAB_DIR  = "common/tables"
LOG_DIR  = "common/logs"
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Lettura dati per dato alpha
def read_y(a):
    """
    Data un certo valore di alpha (float), costruisce il tag per il file
    (es. -1.2 → 'm1p2' lato R → qui usiamo '-1.2' → ' -1p2', ma la convenzione
    è quella: sostituisce '.' con 'p') e legge la colonna 'y' come vettore NumPy.
    """
    tag = str(round(a, 1)).replace(".", "p")
    fpath = os.path.join(DATA_DIR, f"sn_uni_alpha_{tag}.csv")
    return pd.read_csv(fpath)["y"].to_numpy()

# Loop sulla griglia di alpha
rows = []   # accumula una riga per ogni alpha (stime + diagnostiche)
times = []  # accumula tempi di fit per ogni alpha

for a in alpha_grid:
    # (A) Lettura dataset associato a questo valore di alpha
    y = read_y(a)
    n = y.size

    # (B) Stima SciPy (DP: a, loc=xi, scale=omega)
    # Misura grossolanamente il tempo di skewnorm.fit tramite timestamp.
    t0 = pd.Timestamp.now().timestamp()
    a_hat, loc_hat, scale_hat = stats.skewnorm.fit(y)
    t_fit = pd.Timestamp.now().timestamp() - t0

    # (C) Log-verosimiglianza al massimo
    # Somma delle log-densità skew-normal con i parametri stimati.
    loglik = float(np.sum(stats.skewnorm.logpdf(y, a=a_hat, loc=loc_hat, scale=scale_hat)))

    # (D) Inversione CDF↔PPF (core + code)
    # Costruisce una griglia di probabilità:
    #   - p_core: parte centrale (0.001–0.999)
    #   - p_ext : punti estremi per le code
    # Poi:
    #   q_inv  = PPF(p_grid)   (quantili)
    #   p_back = CDF(q_inv)
    # e misura l'errore massimo nel core e negli estremi.
    p_core = np.linspace(0.001, 0.999, 999)
    p_ext  = np.array([1e-6, 1e-5, 1e-4, 1-1e-4, 1-1e-5, 1-1e-6])
    p_grid = np.concatenate([p_ext[:3], p_core, p_ext[3:]])
    q_inv  = stats.skewnorm.ppf(p_grid, a=a_hat, loc=loc_hat, scale=scale_hat)
    p_back = stats.skewnorm.cdf(q_inv, a=a_hat, loc=loc_hat, scale=scale_hat)
    # errore massimo sulla parte centrale
    inv_core = float(np.max(np.abs(p_back[3:-3] - p_core)))
    # errore massimo sulle code (primi 3 e ultimi 3 punti)
    inv_ext  = float(np.max(np.abs(np.concatenate([p_back[:3], p_back[-3:]]) - p_ext)))

    # (E) UNIF uniformità (PIT + KS)
    # Calcola i PIT u = F_SN(y; theta_hat) e applica test KS contro Uniform(0,1).
    u = stats.skewnorm.cdf(y, a=a_hat, loc=loc_hat, scale=scale_hat)
    ks_p = float(stats.kstest(u, "uniform", args=(0, 1)).pvalue)

    # (F) Accumulo risultati per questo valore di alpha
    rows.append({
        "engine": "Python",
        "alpha_true": a,
        "xi_hat":   float(loc_hat),
        "omega_hat": float(scale_hat),
        "alpha_hat": float(a_hat),
        "loglik": loglik,
        "n": int(n),
        "inv_core_maxerr": inv_core,
        "inv_ext_maxerr":  inv_ext,
        "ks_pvalue": ks_p
    })
    times.append({
        "alpha_true": a,
        "step": "fit_skewnorm",
        "seconds": t_fit
    })

# Scrittura output aggregati
# Tabella con tutte le stime e diagnostiche per ogni alpha_true
pd.DataFrame(rows).to_csv(
    os.path.join(TAB_DIR, "sn_uni_grid_Py.csv"),
    index=False
)

# Tabella con tempi di fit per ogni alpha_true
pd.DataFrame(times).to_csv(
    os.path.join(LOG_DIR, "sn_uni_grid_times_Py.csv"),
    index=False
)

print("Python grid: scritto")
