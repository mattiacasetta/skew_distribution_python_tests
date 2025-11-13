"""
SN multivariata (LEAN) — Analisi complete in Python (R vs Py compatibile)

Obiettivo
- Eseguire l’intero flusso per la Skew-Normal multivariata (SN_d) con **Omega diagonale**:
  • lettura/generazione dati,
  • stima ML via BFGS con parametrizzazione stabile (log-varianze),
  • diagnostica (pairs plot),
  • salvataggi CSV compatibili col tuo script di *merge*,
  • log dei tempi per categoria (stessi nomi del merge).

Output prodotti
- common/tables/sn_mv_fit_Py.csv
  Contiene: engine, p, loglik, xi_hat_j, alpha_hat_j, omega_hat_diag (elenco separato da “;”).
- common/tables/sn_mv_fit_Omega_Py.csv (formato long i,j,value) — qui Omega è diagonale.
- common/fig/sn_mv_pairs_Py.png — pairs plot (triangolare superiore) per diagnostica visiva.
- common/logs/sn_mv_times_Py.csv — tempi per step (gen_rvs|io_read_csv_data, fit_ML_BFGS_diag, plot_pairs_save).

Dipendenze
  pip install numpy scipy pandas matplotlib

Note d’uso
- Se esiste "common/data/sn_mv.csv" (con colonne x1,..,xp) lo script lo usa; altrimenti simula dati
  da SN_d con (xi, Omega diag, alpha) predefiniti e seed fisso per replicabilità.
- Il fit è non vincolato (BFGS) grazie alle **log-varianze**: Omega_ii = exp(eta_i) > 0.
- Usa un modulo per SN multivariata (msn) se presente (con rvs/logpdf); altrimenti usa
  una composizione indipendente di skewnorm univariate da scipy.stats come fallback.
"""

from __future__ import annotations

import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interattivo per esecuzioni batch (niente finestre grafiche)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import importlib
import importlib.util
from types import SimpleNamespace


def _load_msn_module():
    """
    Prova a caricare un modulo per Skew-Normal multivariata (msn) con API rvs/logpdf.

    Ordine di ricerca:
      1) import standard con vari nomi di modulo plausibili;
      2) caricamento da file .py in cartelle note;
      3) fallback: composizione indipendente di skewnorm univariate da scipy.stats.
    """
    # 1) tentativi di import con nomi di modulo comuni
    candidates = [
        "multivariate_skew_normal",
        "multivariate_skew_distribution.multivariate_skew_normal",
        "multivariate_skew_distribution.multivariate_skew_normal.multivariate_skew_normal",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            # il modulo può esporre rvs/logpdf a livello top oppure dentro attributo multivariate_skew_normal
            if hasattr(mod, "rvs") and hasattr(mod, "logpdf"):
                print(f"[INFO] Loaded MSN module '{name}' via import")
                return SimpleNamespace(rvs=mod.rvs, logpdf=mod.logpdf)
            if hasattr(mod, "multivariate_skew_normal"):
                sub = getattr(mod, "multivariate_skew_normal")
                if hasattr(sub, "rvs") and hasattr(sub, "logpdf"):
                    print(f"[INFO] Loaded MSN from '{name}.multivariate_skew_normal'")
                    return SimpleNamespace(rvs=sub.rvs, logpdf=sub.logpdf)
        except Exception:
            continue

    # 2) prova a caricare da percorsi file plausibili attorno allo script
    here = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    cand_files = [
        os.path.join(here, "multivariate_skew_normal.py"),
        os.path.join(here, "multivariate_skew_distribution", "multivariate_skew_normal.py"),
        os.path.join(here, "..", "multivariate_skew_distribution", "multivariate_skew_normal.py"),
        os.path.join(os.getcwd(), "multivariate_skew_distribution", "multivariate_skew_normal.py"),
    ]
    for p in cand_files:
        p = os.path.normpath(p)
        if os.path.exists(p):
            try:
                # carica il modulo da file usando importlib.util
                spec = importlib.util.spec_from_file_location("multivariate_skew_normal_local", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "rvs") and hasattr(mod, "logpdf"):
                    print(f"[INFO] Loaded MSN from file: {p}")
                    return SimpleNamespace(rvs=mod.rvs, logpdf=mod.logpdf)
                if hasattr(mod, "multivariate_skew_normal"):
                    sub = getattr(mod, "multivariate_skew_normal")
                    if hasattr(sub, "rvs") and hasattr(sub, "logpdf"):
                        print(f"[INFO] Loaded MSN (sub) from file: {p}")
                        return SimpleNamespace(rvs=sub.rvs, logpdf=sub.logpdf)
            except Exception as e:
                print(f"[WARN] Failed loading MSN from {p}: {e}")
                continue

    # 3) fallback: composta indipendente di skew-normal univariate tramite scipy.stats.skewnorm
    try:
        from scipy.stats import skewnorm
    except Exception:
        raise ModuleNotFoundError(
            "multivariate_skew_normal module not found and scipy.stats.skewnorm unavailable. "
            "Install scipy or add multivariate_skew_distribution/multivariate_skew_normal.py to the repo."
        )

    class _FallbackMSN:
        """Implementazione di fallback: prodotto di skew-normal univariate indipendenti."""

        @staticmethod
        def rvs(loc, scale, shape, size, random_state=None):
            """
            Genera campioni indipendenti per ciascuna componente:
              X_j ~ SN_1(alpha_j, loc_j, sd_j).
            """
            loc = np.asarray(loc)
            shape = np.asarray(shape)
            if np.ndim(size) == 0:
                n = int(size)
            else:
                n = int(size)
            p = loc.size
            # se scale è matrice, prendi radice della diagonale, altrimenti interpretala come deviazione standard
            if np.ndim(scale) == 2:
                sd = np.sqrt(np.diag(scale))
            else:
                sd = np.asarray(scale)
            rng = np.random.default_rng() if random_state is None else random_state
            out = np.zeros((n, p))
            for j in range(p):
                a = float(shape[j])
                out[:, j] = skewnorm.rvs(a, loc=loc[j], scale=sd[j], size=n, random_state=rng)
            return out

        @staticmethod
        def logpdf(X, loc, scale, shape):
            """
            Log-densità del prodotto indipendente:
            somma delle log-densità skew-normal margini.
            """
            X = np.atleast_2d(X)
            loc = np.asarray(loc)
            shape = np.asarray(shape)
            if np.ndim(scale) == 2:
                sd = np.sqrt(np.diag(scale))
            else:
                sd = np.asarray(scale)
            lp = np.zeros(X.shape[0])
            for j in range(loc.size):
                a = float(shape[j])
                lp += skewnorm.logpdf(X[:, j], a, loc=loc[j], scale=sd[j])
            return lp

    print("[WARN] 'multivariate_skew_normal' not found: using independent-marginals fallback (skewnorm).")
    return SimpleNamespace(rvs=_FallbackMSN.rvs, logpdf=_FallbackMSN.logpdf)


# espone msn con l’API attesa (rvs, logpdf)
msn = _load_msn_module()


+PATHS = {
    "data":           "common/data/sn_mv.csv",           # opzionale: se manca, simuliamo
    "fig_pairs":      "common/fig/sn_mv_pairs_Py.png",
    "tab_fit":        "common/tables/sn_mv_fit_Py.csv",
    "tab_Omega_long": "common/tables/sn_mv_fit_Omega_Py.csv",
    "log_times":      "common/logs/sn_mv_times_Py.csv",
}
# crea tutte le directory necessarie
for d in {os.path.dirname(p) for p in PATHS.values()}:
    if d:
        os.makedirs(d, exist_ok=True)


# lista globale tempi
_timings: list[dict] = []  # lista globale di dizionari {step, seconds}

def _tic() -> float:
    """Restituisce l’orario corrente in secondi (alta risoluzione) per timing."""
    return time.perf_counter()

def _tock(t0: float) -> float:
    """Restituisce il tempo trascorso da t0 (secondi)."""
    return time.perf_counter() - t0

def _add_time(step: str, sec: float) -> None:
    """Accoda una misurazione di tempo (step, durata in secondi) alla lista globale."""
    _timings.append({"step": step, "seconds": float(sec)})


# Dati: lettura o simulazione SN_d

def _simulate_data(n: int, xi: np.ndarray, omega_diag: np.ndarray, alpha: np.ndarray,
                   seed: int = 20250303) -> np.ndarray:
    """Simula Y ~ SN_d(xi, Omega, alpha) usando il generatore del modulo msn.
    Omega è diagonale: Omega = diag(omega_diag^2).
    """
    rng = np.random.default_rng(seed)      # generatore pseudo-casuale con seed fisso
    d = xi.size
    Omega = np.diag(omega_diag ** 2)      # costruisce la matrice di covarianza diagonale
    t0 = _tic()
    # rvs del modulo: usa la rappresentazione di selezione/additiva (Azzalini–Capitanio) o fallback
    Y = msn.rvs(loc=xi, scale=Omega, shape=alpha, size=n,
                                         random_state=rng)
    _add_time("gen_rvs", _tock(t0))
    return np.atleast_2d(Y)               # garantisce output (n, p)


def _load_or_simulate() -> tuple[np.ndarray, int]:
    """Carica common/data/sn_mv.csv se presente; altrimenti simula un dataset standard.
    Ritorna (Y, p) con Y di shape (n, p) e p numero di variabili.
    """
    if os.path.exists(PATHS["data"]):
        # caso 1: file dati esistente → lo leggiamo
        t0 = _tic()
        df = pd.read_csv(PATHS["data"])  # atteso: colonne x1,..,xp
        _add_time("io_read_csv_data", _tock(t0))
        Y = df.to_numpy(dtype=float)
        if Y.ndim != 2 or Y.shape[1] < 1:
            raise ValueError("Il CSV dei dati deve avere almeno una colonna di numeri.")
        return Y, Y.shape[1]

    # caso 2: nessun file dati → simulazione default (p=3) coerente con l’elaborato
    p = 3
    n = 2000
    xi = np.array([0.0, 1.0, -1.0])                   # xi (posizione vera)
    omega_diag = np.array([1.0, 1.0, 1.0])            # radici delle varianze vere (deviazioni standard)
    alpha = np.array([2.0, -1.0, 1.5])                # alpha vera (forma)
    Y = _simulate_data(n, xi, omega_diag, alpha)
    return Y, p


# Log-verosimiglianza SN_d (Omega diag)

def _neg_loglik(theta: np.ndarray, Y: np.ndarray) -> float:
    """Negativa della log-verosimiglianza SN_d con Omega diagonale.

    Parametrizzazione (per stabilità numerica):
      theta = [ xi_1..xi_p, eta_1..eta_p, alpha_1..alpha_p ],  con  Omega_ii = exp(eta_i)
    (notare: qui usiamo Omega come matrice di varianza, non la radice; quindi diag(Omega) = exp(eta)).
    """
    n, p = Y.shape
    # scompone il vettore dei parametri nelle tre parti
    xi = theta[:p]
    eta = theta[p:2*p]
    alpha = theta[2*p:2*p+p]

    # Omega = diag(exp(eta))  ⇒ varianze > 0
    Omega_diag = np.exp(eta)                         # varianze marginali positive
    Omega = np.diag(Omega_diag)                      # matrice di covarianza diagonale

    # Log-likelihood: somma della log-densità sul campione
    # msn.logpdf richiede scale = matrice di covarianza, shape = alpha, loc = xi
    ll = msn.logpdf(Y, loc=xi, scale=Omega, shape=alpha)
    # msn.logpdf restituisce array per osservazione; sommiamo e mettiamo il segno meno
    return -float(np.sum(ll))


# Stima ML (BFGS)

def _fit_ml(Y: np.ndarray) -> tuple[dict, np.ndarray]:
    """Stima ML di (xi, Omega diag, alpha) via BFGS nel dominio unconstrained.

    Inizializzazione:
      • xi^(0)  = media marginale
      • eta^(0) = log(varianza marginale + eps)
      • alpha^(0) = 0.5 * sign(skewness marginale) (evita alpha=0 esatto)
    """
    n, p = Y.shape

    # Momenti campionari di base: medie e varianze per ciascuna colonna
    mu = np.mean(Y, axis=0)
    var = np.var(Y, axis=0, ddof=1)

    # Skewness campionaria (m3 / s^3) per determinare il segno di alpha iniziale
    m3 = np.mean((Y - mu) ** 3, axis=0)
    s = np.sqrt(var)
    with np.errstate(divide='ignore', invalid='ignore'):
        skew = np.where(s > 0, m3 / np.maximum(s, 1e-12) ** 3, 0.0)
    alpha0 = 0.5 * np.sign(skew)  # evita alpha=0 (punto non regolare)

    # Parametri iniziali nel dominio unconstrained
    eps = 1e-6
    xi0 = mu
    eta0 = np.log(np.maximum(var, eps))   # log-varianze iniziali
    theta0 = np.concatenate([xi0, eta0, alpha0])

    # Ottimizzazione della negativa log-verosimiglianza via BFGS
    t0 = _tic()
    res = minimize(
        _neg_loglik,
        theta0,
        args=(Y,),
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 2000}
    )
    _add_time("fit_ML_BFGS_diag", _tock(t0))

    if not res.success:
        print("[WARN] Ottimizzazione non convergente:", res.message)

    # Estrae le stime dai parametri finali
    th = res.x
    xi_hat    = th[:p]
    eta_hat   = th[p:2*p]
    alpha_hat = th[2*p:2*p+p]
    Omega_hat_diag = np.exp(eta_hat)   # varianze marginali stimate

    # Costruisce una riga di output compatibile con lo script R
    out = {
        "engine": "Python",
        "p": int(p),
        "loglik": float(-res.fun),
        **{f"xi_hat_{j+1}": float(xi_hat[j])       for j in range(p)},
        **{f"alpha_hat_{j+1}": float(alpha_hat[j]) for j in range(p)},
        "omega_hat_diag": ";".join(f"{v:.12g}" for v in Omega_hat_diag),
    }
    return out, Omega_hat_diag


# Pairs plot

def _pairs_plot(Y: np.ndarray, path_png: str) -> None:
    """Pairs-plot minimale (triangolo superiore) per diagnostica grafica.

    Diagonale: istogrammi marginali;
    Triangolo superiore: scatter plot coppie (x_j, x_i);
    Triangolo inferiore: vuoto (asse spento).
    """
    n, p = Y.shape
    fig, axes = plt.subplots(p, p, figsize=(2.6*p, 2.6*p))
    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if i == j:
                # istogramma marginale della variabile j-esima
                ax.hist(Y[:, j], bins=30)
            elif i < j:
                # scatter plot tra variabile j (asse x) e variabile i (asse y)
                ax.scatter(Y[:, j], Y[:, i], s=5)
            else:
                # triangolo inferiore: nessun contenuto
                ax.axis('off')
            # gestione etichette assi per non sovraccaricare il grafico
            if i == p-1:
                ax.set_xlabel(f"x{j+1}")
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(f"x{i+1}")
            elif j != 0:
                ax.set_yticklabels([])
    plt.tight_layout()

    # salva su file e registra il tempo impiegato
    t0 = _tic()
    fig.savefig(path_png, dpi=120)
    plt.close(fig)
    _add_time("plot_pairs_save", _tock(t0))


# Salvataggi

def _save_fit_table(row: dict, path_csv: str) -> None:
    """Salva una singola riga di stime (row) in un CSV."""
    pd.DataFrame([row]).to_csv(path_csv, index=False)


def _save_Omega_long(diag_vals: np.ndarray, path_csv: str) -> None:
    """Salva Omega (diagonale) in formato long i,j,value per compatibilità con il merge.
    Gli elementi fuori diagonale sono 0 (place-holder).
    """
    p = diag_vals.size
    rows = []
    for i in range(p):
        for j in range(p):
            v = diag_vals[i] if i == j else 0.0
            rows.append({"i": i+1, "j": j+1, "value": float(v)})
    pd.DataFrame(rows).to_csv(path_csv, index=False)


def _save_timings(path_csv: str) -> None:
    """Salva la tabella dei tempi raccolti in _timings."""
    pd.DataFrame(_timings).to_csv(path_csv, index=False)


# Main

def main() -> None:
    """Entry point: dati → fit ML → diagnostica → salvataggi → messaggi riepilogo."""
    # 1) Carica o simula i dati
    Y, p = _load_or_simulate()

    # 2) Fit ML con Omega diagonale
    fit_row, omega_diag = _fit_ml(Y)

    # 3) Diagnostica grafica: pairs plot
    _pairs_plot(Y, PATHS["fig_pairs"])

    # 4) Salvataggi CSV (stime, Omega in formato long, tempi)
    _save_fit_table(fit_row, PATHS["tab_fit"])
    _save_Omega_long(omega_diag, PATHS["tab_Omega_long"])
    _save_timings(PATHS["log_times"])

    # 5) Messaggio riassuntivo su stdout
    print("OK Python — Fit:", PATHS["tab_fit"])
    print("OK Python — Omega (long):", PATHS["tab_Omega_long"])
    print("OK Python — Figure:", PATHS["fig_pairs"])
    print("OK Python — Tempi:", PATHS["log_times"])


if __name__ == "__main__":
    main()
