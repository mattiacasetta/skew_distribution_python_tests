# Implementazione della funzione di ripartizione della SUN
# Vengono utilizzate solo le librerie SciPy e NumPy
# Utilizzo la funzione di densità SUN definita in densita_sun.py
# Basata sulla funzione del pacchetto "sn" di R

from __future__ import annotations

import os
import sys
import warnings
import importlib.util
import numpy as np
from typing import Optional
from scipy.stats import multivariate_normal
from scipy.linalg import LinAlgError  # noqa: F401 (import coerente con il file originale)
from scipy.special import ndtr        # noqa: F401 (import coerente con il file originale)

# -----------------------------------------------------------------------------
# Caricamento esplicito di densita_sun.py dalla stessa cartella
# (utile quando la directory non è nel PYTHONPATH o non è un package)
# -----------------------------------------------------------------------------
here = os.path.dirname(__file__)
densita_path = os.path.join(here, "densita_sun.py")

if not os.path.exists(densita_path):
    # Se il file non esiste, prova comunque un import "normale" per un messaggio più informativo
    try:
        from SUN.densita_sun import dsun, make_PD, _is_simdefpos  # pragma: no cover
    except Exception as e:
        raise ImportError(
            f"densita_sun.py non trovato in {here} e import semplice fallito: {e}"
        )
else:
    spec = importlib.util.spec_from_file_location("densita_sun_local", densita_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    dsun = getattr(mod, "dsun")
    make_PD = getattr(mod, "make_PD")
    _is_simdefpos = getattr(mod, "_is_simdefpos")


# -----------------------------------------------------------------------------
# Funzione di ripartizione SUN (stima Monte Carlo importance sampling)
# -----------------------------------------------------------------------------
def psun(
    x: np.ndarray,
    xi: np.ndarray,
    Omega: np.ndarray,
    Delta: np.ndarray,
    tau: np.ndarray,
    Gamma: np.ndarray,
    log: bool = False,
    max_dim: int = 20,
    nsamples: int = 20000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Calcola la funzione di ripartizione (CDF) della distribuzione SUN nel/i punto/i x.

    Metodo:
    - Stima Monte Carlo con importance sampling usando come proposta φ_d(·; xi, Omega).
    - Il peso è w(y) = f_SUN(y) / φ_d(y; xi, Omega).
    - Per ogni x_i si stima P(Y ≤ x_i) ≈ media( w(Y_j) * 1{Y_j ≤ x_i} ).

    Parametri
    ---------
    x : (n,d) oppure (d,)
        Punti in cui valutare la CDF. Se (d,), viene trattato come un singolo punto.
    xi : (d,)
        Vettore di posizione.
    Omega : (d,d)
        Matrice di covarianza (come in dsun).
    Delta : (d,m)
        Matrice di selezione/skewness.
    tau : (m,)
        Vettore di soglia.
    Gamma : (m,m)
        Matrice di covarianza aggiuntiva.
    log : bool
        Se True, restituisce log(CDF) (con clipping numerico).
    max_dim : int
        Soglia massima su d+m (default 20), come in "sn" di R: evita CDF multivariate instabili.
    nsamples : int
        Numero di campioni Monte Carlo per la stima (proposta φ_d).
    seed : int | None
        Semina del generatore pseudo-casuale (per riproducibilità).

    Ritorna
    -------
    np.ndarray
        Vettore (n,) con le stime della CDF in [0,1] per ciascuna riga di x
        (o log-CDF se log=True).
    """
    # -------------------------------
    # Validazioni e conversioni input
    # -------------------------------
    X = np.asarray(x)
    xi = np.asarray(xi)
    Omega = np.asarray(Omega)
    Delta = np.asarray(Delta)
    tau = np.asarray(tau)
    Gamma = np.asarray(Gamma)

    # Accetta (d,) → (1,d)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, d = X.shape
    m = tau.size

    # Controlli dimensionali
    if xi.size != d:
        raise ValueError("Dimensione di xi non coerente con X.")
    if Omega.shape != (d, d):
        raise ValueError("Omega deve essere d x d.")
    if Delta.shape != (d, m):
        raise ValueError("Delta deve essere d x m.")
    if Gamma.shape != (m, m):
        raise ValueError("Gamma deve essere m x m.")

    # Soglia pratica su d+m (come in 'sn' di R) per stabilità numerica
    if d + m > max_dim:
        raise ValueError(
            f"Dimensione complessiva d+m={d+m} supera la soglia max_dim={max_dim}."
        )

    # Verifica simmetria/PD su Omega e Gamma
    if not _is_simdefpos(Omega):
        raise ValueError("Omega non simmetrica/PD.")
    if not _is_simdefpos(Gamma):
        raise ValueError("Gamma non simmetrica/PD.")

    # ----------------------------------------
    # Standardizzazione e regolarizzazione SPD
    # ----------------------------------------
    # Diagonale delle varianze marginali e matrice standardizzata Omega_bar
    omega = np.sqrt(np.diag(Omega))
    if np.any(omega <= 0):
        raise ValueError("Omega non definita positiva.")
    inv_omega = np.diag(1.0 / omega)
    Omega_bar = inv_omega @ Omega @ inv_omega
    Omega_bar = 0.5 * (Omega_bar + Omega_bar.T)  # simmetria numerica
    # Garantisce SPD (per coerenza con densità e stabilità)
    Omega_bar = make_PD(Omega_bar, eps_start=1e-10, max_iter=20)

    # -----------------------------
    # Proposta Monte Carlo: φ_d(·)
    # -----------------------------
    Omega_pd = make_PD(Omega, eps_start=1e-12, max_iter=20)
    if not np.allclose(Omega_pd, Omega):
        warnings.warn("Omega regularized to be positive-definite for sampling.")

    rng = np.random.default_rng(seed)
    # Campioni dalla proposta φ_d(y; xi, Omega_pd)
    Y = rng.multivariate_normal(mean=xi, cov=Omega_pd, size=int(nsamples))

    # Densità della proposta φ(Y) (normale multivariata)
    mvn_prop = multivariate_normal(mean=xi, cov=Omega_pd, allow_singular=True)
    phi_vals = mvn_prop.pdf(Y)

    # Densità bersaglio f(Y) = dsun(Y, ...)
    f_vals = np.asarray(dsun(Y, xi, Omega, Delta, tau, Gamma, log=False))

    # Evita divisioni per zero e calcola i pesi d'importanza
    phi_vals = np.clip(phi_vals, 1e-300, None)
    weights = f_vals / phi_vals

    # -----------------------------------------
    # Stima CDF per ciascun punto x_i richiesto
    # -----------------------------------------
    out = np.empty(n, dtype=float)
    stderr = np.empty(n, dtype=float)  # stima dell'errore standard (informativa)

    for i in range(n):
        x_i = X[i]
        # Indicatore 1{Y ≤ x_i} (confronto componente per componente)
        inds = np.all(Y <= x_i.reshape(1, -1), axis=1).astype(float)

        # Stimatore d'importanza: E[ w(Y) * 1{Y ≤ x_i} ]
        vals = weights * inds
        est = float(np.mean(vals))

        # Varianza Monte Carlo (solo informativa; non restituita)
        var = float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0
        se = float(np.sqrt(var / max(1, len(vals))))

        # Proietta entro [0,1] per robustezza numerica
        out[i] = min(max(est, 0.0), 1.0)
        stderr[i] = se  # disponibile per diagnostica, non usato in output

    # -----------------
    # Output (log o no)
    # -----------------
    if log:
        with np.errstate(divide="ignore"):
            return np.log(np.clip(out, 1e-300, None))
    return out


# -----------------------------------------------------------------------------
# Esempio di utilizzo (eseguibile come script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parametri SUN di esempio (d = 2, m = 1)
    xi = np.array([0.0, 0.0])
    Omega = np.array([[1.0, 0.5],
                      [0.5, 1.0]])
    Delta = np.array([[0.7],
                      [0.7]])
    tau = np.array([0.0])
    Gamma = np.array([[1.0]])

    # Punti di valutazione
    points = np.array([[0.0,  0.0],
                       [1.0,  1.0],
                       [-1.0, -1.0],
                       [0.5, -0.5]])

    # Stima della CDF SUN
    cdf_values = psun(points, xi, Omega, Delta, tau, Gamma)
    print("CDF values at test points:")
    for pt, val in zip(points, cdf_values):
        print(f"psun({pt}) = {val}")