"""
SN multivariata (LEAN, Omega diagonale) — Stress test su alpha (Python)

Questo script è il *gemello* del grid univariato, adattato al caso multivariato.

Assunzioni operative (coerenti con il flusso R→Py):
- I dataset per ciascun valore di alpha (scalare) sono già stati generati in R
  e salvati in:  common/data/sn_mv_alpha_a_<TAG>.csv
  dove <TAG> = valore di 'a' formattato con un decimale e punto → 'p' (es: -1.2 → "-1p2").
- Ogni CSV contiene le colonne delle variabili (es. x1,x2,x3).
- Il test varia un **alpha scalare a** applicato in modo **omogeneo** a tutte le componenti,
  cioè il vettore di forma è alpha = (a, a, ..., a). Questo schema è semplice da replicare
  anche in R ed è utile per stressare congiuntamente tutte le marginali.

Cosa fa lo script:
1) per ciascun a in ALPHA_GRID, legge il dataset condiviso;
2) stima via ML (BFGS) un modello SN_p con Omega = diag(exp(eta)) > 0 (parametrizzazione stabile);
3) salva per ogni a:  xi_hat_j, alpha_hat_j, diag(Omega_hat)_j, logLik, tempi.

Output:
- common/tables/sn_mv_grid_Py.csv          (stime per a)
- common/logs/sn_mv_grid_times_Py.csv      (tempi: I/O e fit)

Dipendenze: numpy, pandas, scipy; modulo multivariate_skew_distribution (repo locale)
            contenente multivariate_skew_normal.pdf/logpdf/rvs.

Nota: per robustezza, se il file CSV non esiste lo script segnala e prosegue al successivo.
"""
from __future__ import annotations

import os
import math
import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from scipy.optimize import minimize

# import del generatore/densità SN multivariata (repo locale)
import sys
sys.path.append("multivariate_skew_distribution")
from multivariate_skew_normal import multivariate_skew_normal  # type: ignore

# Griglia di valori di alpha scalare da testare (stessi valori usati in R)
ALPHA_GRID = [-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0]

# Directory di lavoro: dati in ingresso, tabelle di output e log dei tempi
DATA_DIR = "common/data"
TAB_DIR  = "common/tables"
LOG_DIR  = "common/logs"
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def tag_from_alpha(a: float) -> str:
    """Costruisce il tag stile R a partire da a (es: -1.2 → 'm1p2').

    - Converte il numero a in stringa con un decimale.
    - Sostituisce il segno '-' con 'm'.
    - Sostituisce il punto decimale con 'p'.
    Esempi:
      a = -1.2 → 'm1p2'
      a =  0.2 → '0p2'
    """
    s = f"{a:.1f}"
    # formato R: "-" -> "m", "." -> "p"  (es. -1.2 -> "m1p2", 0.2 -> "0p2")
    tag_r_style = s.replace("-", "m").replace(".", "p")
    return tag_r_style

def _alternate_tag(a: float) -> str:
    """Formato alternativo di tag con segno esplicito (es: -1.2 → '-1p2').

    Questo serve per retrocompatibilità nel caso i file siano stati salvati
    con il segno '-' anziché con 'm'.
    """
    return f"{a:.1f}".replace(".", "p")


def read_matrix_for_alpha(a: float) -> np.ndarray | None:
    """Legge la matrice X dal CSV pre-generato in R per un dato valore di a.

    Percorso atteso:
      common/data/sn_mv_alpha_a_<TAG>.csv

    - Prova prima col tag stile R (m1p2, 0p2, ...)
    - Poi con il formato alternativo col segno (-1p2, 0p2, ...)
    - Se nessun file esiste, stampa un warning e restituisce None.
    - Se il file esiste, seleziona solo le colonne numeriche e le converte
      in matrice NumPy (n x p).
    """
    # prova prima il tag in stile R (m1p2), poi il formato alternativo con segno (-1p2)
    candidates = [
        tag_from_alpha(a),
        _alternate_tag(a)
    ]
    fpath = None
    for tg in candidates:
        cand = os.path.join(DATA_DIR, f"sn_mv_alpha_a_{tg}.csv")
        if os.path.exists(cand):
            fpath = cand
            break
    if fpath is None:
        # diagnostica più informativa (mostra i path provati)
        tried = ", ".join(os.path.join(DATA_DIR, f"sn_mv_alpha_a_{tg}.csv") for tg in candidates)
        print(f"[WARN] Dataset mancante per a={a}. Path provati: {tried}")
        return None

    df = pd.read_csv(fpath)
    # Seleziona solo colonne numeriche (esclude eventuali ID/stringhe)
    X = df.select_dtypes(include=["number"]).to_numpy()
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        print(f"[WARN] File privo di colonne numeriche utili: {fpath}")
        return None
    return X


def nll_msn_diag(params: np.ndarray, X: np.ndarray) -> float:
    """Negative log-likelihood per SN_p con Omega diagonale.

    Parametrizzazione:
      params = [xi(1..p), eta(1..p), alpha(1..p)]
      dove Omega = diag(exp(eta)) (varianze marginali positive).

    Passi:
      - estrae xi, eta, alpha dal vettore params;
      - costruisce la diagonale delle varianze come exp(eta);
      - valuta la log-densità multivariata skew-normal sul campione X;
      - restituisce la negativa della somma delle log-densità.
    """
    n, p = X.shape
    xi   = params[0:p]
    eta  = params[p:2*p]
    alp  = params[2*p:3*p]

    # Costruisci cov. diagonale (varianze positive via esponenziale)
    diag_var = np.exp(eta)
    # Se qualcosa è non finito, penalizza fortemente (per robustezza)
    if not np.all(np.isfinite(diag_var)):
        return 1e300

    # Log-likelihood (somma delle log-densità del modello SN multivariato)
    try:
        ll = float(np.sum(multivariate_skew_normal.logpdf(X, loc=xi, scale=np.diag(diag_var), shape=alp)))
        if not np.isfinite(ll):
            # Penalizza se la log-verosimiglianza contiene NaN/Inf
            return 1e300
        return -ll
    except Exception:
        # In caso di errori numerici o shape mismatch, penalizza fortemente
        return 1e300


def fit_msn_diag(X: np.ndarray, a_hint: float) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Stima ML per SN multivariata con Omega diagonale.

    - Inizializza:
        xi    = mediana di ciascuna colonna di X;
        eta   = log(varianza di ciascuna colonna, con floor minimo);
        alpha = segno di a_hint * c, con c ∈ [0.1, 2] (evita alpha=0).
    - Ottimizza la negativa log-verosimiglianza nll_msn_diag con BFGS.
    - Ritorna:
        scalars: dizionario di scalari riassuntivi (n, p, loglik, ecc.)
        vectors: dizionario di vettori (xi_hat, alpha_hat, omega_hat_diag)
    """
    n, p = X.shape

    # Start robusti per xi: mediane marginali (meno sensibili agli outlier)
    xi0  = np.nanmedian(X, axis=0)
    # Varianze marginali con piccolo floor per evitare zeri
    v0   = np.nanvar(X, axis=0, ddof=1)
    v0[v0 <= 1e-12] = 1e-12
    eta0 = np.log(v0)

    # alpha start: usa il segno di a_hint con ampiezza controllata in [0.1, 2]
    mag  = max(0.1, min(2.0, abs(a_hint)))
    alp0 = np.sign(a_hint) * np.full(p, mag, dtype=float)

    # Costruisce il vettore dei parametri iniziali: [xi0, eta0, alp0]
    x0 = np.concatenate([xi0, eta0, alp0])

    # Ottimizzazione BFGS della NLL
    t0 = time.perf_counter()
    opt = minimize(
        fun=nll_msn_diag,
        x0=x0,
        args=(X,),
        method="BFGS",
        options={"maxiter": 5000, "gtol": 1e-8}
    )
    fit_time = time.perf_counter() - t0

    # Parametri stimati
    par = opt.x
    xi_hat  = par[0:p]
    eta_hat = par[p:2*p]
    alp_hat = par[2*p:3*p]
    var_hat = np.exp(eta_hat)  # varianze marginali stimate (diagonale di Omega_hat)

    # log-verosimiglianza al massimo
    ll = -float(opt.fun)

    # scalars: valori riassuntivi per riga output
    scalars = {
        "n": float(n),
        "p": float(p),
        "loglik": ll,
        "fevals": float(opt.nfev if hasattr(opt, "nfev") else np.nan),
        "convergence": float(opt.status),
        "fit_time_sec": fit_time,
    }
    # vectors: vettori di parametri stimati (per colonne wide nell'output)
    vectors = {"xi_hat": xi_hat, "alpha_hat": alp_hat, "omega_hat_diag": var_hat}
    return scalars, vectors


# loop principale sulla griglia di alpha scalari
rows = []   # accumula righe "wide" di stime per ogni a
times = []  # accumula tempi di I/O e fit per ogni a

for a in ALPHA_GRID:
    # (A) Lettura del dataset per questo valore di alpha
    t0 = time.perf_counter()
    X = read_matrix_for_alpha(a)
    io_time = time.perf_counter() - t0
    if X is None:
        # se i dati mancano o sono inutilizzabili, passa al prossimo a
        continue

    # (B) Fit (Omega diagonale) con a come "hint" per l'inizializzazione di alpha
    scalars, vectors = fit_msn_diag(X, a_hint=a)

    # (C) Costruzione della riga di output "wide" per questo a
    p = int(scalars["p"]) if "p" in scalars else X.shape[1]
    row = {
        "engine": "Python",
        "alpha_true": float(a),                 # valore vero di alpha scalare usato in R
        "p": p,
        "loglik": float(scalars["loglik"]),
        "n": int(scalars["n"]),
    }
    # aggiunge le componenti xi_hat_j, alpha_hat_j, omega_hat_diag_j
    for j in range(p):
        row[f"xi_hat_{j+1}"] = float(vectors["xi_hat"][j])
        row[f"alpha_hat_{j+1}"] = float(vectors["alpha_hat"][j])
        row[f"omega_hat_diag_{j+1}"] = float(vectors["omega_hat_diag"][j])

    rows.append(row)

    # (D) Salvataggio dei tempi per questo a (I/O e fit)
    times.append({"alpha_true": float(a), "step": "io_read_csv_data", "seconds": io_time})
    times.append({"alpha_true": float(a), "step": "fit_ML_BFGS_diag", "seconds": float(scalars["fit_time_sec"])})



# Se abbiamo almeno una riga di stime, scriviamo la tabella aggregata
if rows:
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "sn_mv_grid_Py.csv"), index=False)
    print(f"[OK] Tabelle: {os.path.join(TAB_DIR, 'sn_mv_grid_Py.csv')}")

# Se abbiamo almeno una misura di tempo, scriviamo la tabella dei tempi
if times:
    pd.DataFrame(times).to_csv(os.path.join(LOG_DIR, "sn_mv_grid_times_Py.csv"), index=False)
    print(f"[OK] Tempi:   {os.path.join(LOG_DIR, 'sn_mv_grid_times_Py.csv')}")

print("Done.")
