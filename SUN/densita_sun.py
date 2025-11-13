# Implementazione della funzione di densità della SUN 
# Vengono utilizzate solo le librerie SciPy e NumPy
# Basata sulla funzione del pacchetto "sn" di R

from __future__ import annotations

import warnings
import numpy as np
import scipy
from typing import Optional, Tuple
from scipy.stats import multivariate_normal
from scipy.linalg import LinAlgError


# -----------------------------------------------------------------------------
# Utilità numeriche
# -----------------------------------------------------------------------------

def _chol_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Risolve il sistema (L Lᵀ) x = B usando forward e backward substitution,
    dato il fattore di Cholesky inferiore L (ovvero: A = L Lᵀ).

    Parametri
    ---------
    L : np.ndarray
        Fattore di Cholesky inferiore (triangolare) di A.
    B : np.ndarray
        Termine noto (vettore o matrice). Le ultime dimensioni devono
        essere compatibili con L.

    Ritorna
    -------
    np.ndarray
        Soluzione x che soddisfa (L Lᵀ) x = B.
    """
    # forward solve: L Y = B
    Y = scipy.linalg.solve(L, B, assume_a='pos', lower=True, check_finite=False)
    # backward solve: Lᵀ x = Y
    return scipy.linalg.solve(L.T, Y, assume_a='pos', lower=False, check_finite=False)


def _is_simdefpos(A: Optional[np.ndarray], tol: float = 1e-12) -> bool:
    """
    Verifica se una matrice è simmetrica (entro tolleranza) e definita positiva.

    Note:
    - Usa eigvalsh (autovalori reali) perché A è simmetrica.
    - Consente un piccolo margine negativo (~ -1e-12) per tolleranze numeriche.

    Parametri
    ---------
    A : np.ndarray | None
        Matrice candidata.
    tol : float
        Tolleranza per la simmetria.

    Ritorna
    -------
    bool
        True se la matrice è simmetrica e definita positiva entro tolleranza.
    """
    if A is None:
        return False
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    # simmetria numerica
    if not np.allclose(A, A.T, atol=max(tol, 1e-15)):
        return False
    # positività (entro piccola tolleranza negativa)
    try:
        vals = np.linalg.eigvalsh(A)
    except Exception:
        return False
    return np.min(vals) > -1e-12


def make_PD(A: np.ndarray, eps_start: float = 1e-8, max_iter: int = 10) -> np.ndarray:
    """
    Rende una matrice simmetrica e (quasi) definita positiva aggiungendo un jitter
    diagonale crescente fino a ottenere una fattorizzazione di Cholesky.

    Strategia:
    - Simmetrizza A.
    - Prova cholesky; se fallisce, sposta gli autovalori minimi verso la positività
      con un incremento diagonale crescente.
    - Come fallback, forza positività diagonale minima.

    Parametri
    ---------
    A : np.ndarray
        Matrice di ingresso.
    eps_start : float
        Jitter diagonale iniziale.
    max_iter : int
        Numero massimo di tentativi di regolarizzazione.

    Ritorna
    -------
    np.ndarray
        Matrice simmetrica (quasi) definita positiva.
    """
    A = 0.5 * (np.asarray(A) + np.asarray(A).T)
    eps = eps_start

    for _ in range(max_iter):
        try:
            # Se Cholesky riesce, A è già PD
            scipy.linalg.cholesky(A, check_finite=False)
            return A
        except LinAlgError:
            # Stima lo shift necessario a rendere A PD
            try:
                min_ev = np.min(scipy.linalg.eigvalsh(A))
            except Exception:
                min_ev = -abs(eps)
            shift = max(eps, -min_ev + eps)
            A = A + np.eye(A.shape[0]) * shift
            eps *= 10

    # Fallback: garantisce positività minima sulla diagonale
    diag = np.diag(A).copy()
    diag[diag <= 0] = eps
    A = A + np.eye(A.shape[0]) * eps
    return A


# -----------------------------------------------------------------------------
# Densità della SUN
# -----------------------------------------------------------------------------

def dsun(
    x: np.ndarray,
    xi: np.ndarray,
    Omega: np.ndarray,
    Delta: np.ndarray,
    tau: np.ndarray,
    Gamma: np.ndarray,
    log: bool = False
) -> np.ndarray:
    """
    Calcola la (log-)densità della distribuzione SUN nel/i punto/i x.

    Parametri
    ---------
    x : np.ndarray
        Punto di valutazione. Accetta forma (p,) per un singolo punto o (n, p) per n punti.
    xi : np.ndarray
        Vettore di localizzazione (dimensione p,).
    Omega : np.ndarray
        Matrice di covarianza (p x p).
    Delta : np.ndarray
        Matrice di skewness (p x m).
    tau : np.ndarray
        Vettore di soglia (m,).
    Gamma : np.ndarray
        Matrice di covarianza aggiuntiva (m x m).
    log : bool
        Se True, restituisce il logaritmo della densità.

    Ritorna
    -------
    np.ndarray
        Vettore (n,) di densità o log-densità della SUN in corrispondenza delle righe di x.
    """
    # --- Validazioni e conversioni ---
    X = np.asarray(x)
    xi = np.asarray(xi)
    Omega = np.asarray(Omega)
    Delta = np.asarray(Delta)
    tau = np.asarray(tau)
    Gamma = np.asarray(Gamma)

    # Consenti x singolo (p,) oppure matrice (n, p)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, p = X.shape
    m = tau.size

    # Controlli dimensionali coerenti
    if xi.size != p:
        raise ValueError("Dimensione di xi non coerente con X.")
    if Omega.shape != (p, p):
        raise ValueError("Omega deve essere p x p.")
    if Delta.shape[0] != p or Delta.shape[1] != m:
        raise ValueError("Delta deve essere p x m.")
    if Gamma.shape != (m, m):
        raise ValueError("Gamma deve essere m x m.")

    # Verifica simmetria/PD per Omega e Gamma (entro tolleranza)
    if not _is_simdefpos(Omega):
        raise ValueError("Omega non simmetrica/PD.")
    if not _is_simdefpos(Gamma):
        raise ValueError("Gamma non simmetrica/PD.")

    # --- Standardizzazione marginale ---
    # omega: deviazioni standard marginali; inv_omega: inv(diag(omega))
    omega = np.sqrt(np.diag(Omega))
    if np.any(omega == 0):
        raise ValueError("Elementi diagonali di Omega nulli.")
    inv_omega = np.diag(1.0 / omega)

    # Omega_bar = D^{-1} Omega D^{-1} con D = diag(omega)
    Omega_bar = inv_omega @ Omega @ inv_omega
    Omega_bar = 0.5 * (Omega_bar + Omega_bar.T)  # simmetrizza numericamente
    Omega_bar = make_PD(Omega_bar)

    # Fattorizzazione di Cholesky di Omega_bar
    L_bar = scipy.linalg.cholesky(Omega_bar, check_finite=False)

    # M = Omega_bar^{-1} Delta (risolto tramite Cholesky)
    M = _chol_solve(L_bar, Delta)

    # Sigma_m = Gamma - Deltaᵀ Omega_bar^{-1} Delta
    Sigma_m = Gamma - Delta.T @ M
    Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)  # simmetrizza

    # Assicurati che Sigma_m sia SPD (aggiusta con jitter se necessario)
    if not _is_simdefpos(Sigma_m):
        Sigma_m = make_PD(Sigma_m, eps_start=1e-10, max_iter=20)
        if not _is_simdefpos(Sigma_m):
            raise ValueError("Sigma_m non convertibile in SPD.")

    # Cholesky di Sigma_m (ridondante per robustezza numerica, come da implementazione originale)
    L_Sm = scipy.linalg.cholesky(Sigma_m, check_finite=False)

    # Ulteriore salvaguardia: riapplica make_PD e rifai Cholesky
    Sigma_m = make_PD(Sigma_m, eps_start=1e-10, max_iter=20)
    L_Sm = scipy.linalg.cholesky(Sigma_m, check_finite=False)

    # --- Preparazione covarianze regolarizzate per valutazioni PDF/CDF ---
    Gamma_pd = make_PD(Gamma, eps_start=1e-10, max_iter=20)
    if not np.allclose(Gamma_pd, Gamma):
        warnings.warn("Gamma regularized to be positive-definite for CDF evaluation.")

    Omega_pd = make_PD(Omega, eps_start=1e-12, max_iter=20)
    if not np.allclose(Omega_pd, Omega):
        warnings.warn("Omega regularized to be positive-definite for PDF evaluation.")

    # Distribuzioni normali multivariate per le componenti CDF/PDF
    mvn_Gamma = multivariate_normal(mean=np.zeros(m), cov=Gamma_pd, allow_singular=True)
    pi = mvn_Gamma.cdf(tau)
    pi = float(np.clip(pi, 1e-300, None))  # stabilità log
    log_pi = np.log(pi)

    # Prealloca output (log-densità) e PDF con Omega regolarizzata
    out_log = np.empty(n, dtype=float)
    mvn_Omega = multivariate_normal(mean=xi, cov=Omega_pd, allow_singular=True)

    # --- Ciclo sui punti X ---
    for i in range(n):
        # Scostamento dal centro
        xi_minus = X[i] - xi

        # Standardizzazione z = D^{-1} (x - xi)
        z = inv_omega @ xi_minus

        # u = Omega_bar^{-1} z (via Cholesky solve)
        u = _chol_solve(L_bar, z)

        # s = tau + Deltaᵀ u  (argomento CDF m-dimensionale)
        s = tau + Delta.T @ u

        # log φ_p(x; xi, Omega)
        log_phi = mvn_Omega.logpdf(X[i])

        # Calcolo CDF_m( s ; 0, Sigma_m ) con Sigma_m regolarizzata
        Sm_pd = make_PD(Sigma_m, eps_start=1e-10, max_iter=20)
        if not np.allclose(Sm_pd, Sigma_m):
            warnings.warn("Sigma_m regularized prior to CDF(s) evaluation.")
        mvn_Sm = multivariate_normal(mean=np.zeros(m), cov=Sm_pd, allow_singular=True)
        pi2 = mvn_Sm.cdf(s)
        pi2 = float(np.clip(pi2, 1e-300, None))
        log_pi2 = np.log(pi2)

        # log-densità SUN (formula: log φ + log Π - log Π(tau))
        out_log[i] = log_phi + log_pi2 - log_pi

    # Ritorna log-densità o densità
    return out_log if log else np.exp(out_log)


# -----------------------------------------------------------------------------
# Esempio di utilizzo (eseguibile come script)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Parametri di esempio (p = 2, m = 2)
    xi = np.array([0.0, 0.0])
    Omega = np.array([[1.0, 0.5],
                      [0.5, 1.0]])
    Delta = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
    tau = np.array([0.0, 0.0])
    Gamma = np.array([[1.0, 0.2],
                      [0.2, 1.0]])

    # Punti di valutazione (n = 2)
    x = np.array([[0.5, 0.5],
                  [1.0, 1.0]])

    # Calcolo della densità SUN
    dens = dsun(x, xi, Omega, Delta, tau, Gamma, log=False)
    print("Densità SUN:", dens)