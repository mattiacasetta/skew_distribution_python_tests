# UNIF — ST multivariata: stress test su alpha (griglia su scalare s che moltiplica una direzione fissa)
# Fit MLE (Sigma diagonale) con modulo locale multivariate_skew_t, bounds robusti su log_var e nu.
# Input attesi (uno per s): common/data/st_mv_A_alpha_<tag>.csv  (scritti dallo script R)
# Output:
#   - common/tables/st_mv_A_grid_Py.csv
#   - common/logs/st_mv_A_grid_times_Py.csv

import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Individua la root del progetto (due livelli sopra questo file).
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Cartelle candidate dove ci si aspetta di trovare i moduli multivariati.
CANDIDATES = [
    PROJ_ROOT / "multivariate_skew_distributions",
    PROJ_ROOT / "multivariate_skew_distribution",
]

# Cerca il file multivariate_skew_t.py nelle cartelle candidate e,
# se trovato, aggiunge quella cartella al sys.path per poter importare.
found = None
for cand in CANDIDATES:
    if (cand / "multivariate_skew_t.py").exists():
        sys.path.insert(0, str(cand))
        found = cand
        break

# Se non viene trovato il file, genera un errore esplicativo.
if found is None:
    raise FileNotFoundError(
        "Impossibile trovare multivariate_skew_t.py. "
        "Atteso in:\n  - multivariate_skew_distributions/\n  - multivariate_skew_distribution/\n"
        f"Root progetto: {PROJ_ROOT}"
    )

# Importa SOLO la versione 'generator' del modello skew-t multivariato (mst).
from multivariate_skew_distributions.multivariate_skew_t import multivariate_skew_t as mst  # noqa: E402


# Griglia dei valori di scala alpha (come nello script R).
alpha_grid = [-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0]  # stessi tag dell'R grid

# Directory per dati, tabelle e log (relative alla root del progetto).
DATA_DIR = PROJ_ROOT / "common/data"
TAB_DIR  = PROJ_ROOT / "common/tables"
LOG_DIR  = PROJ_ROOT / "common/logs"

# Crea directory per tabelle e log se non esistono già.
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Direzione di skew (coerente con R): v = (2, -1, 1.5) / ||v||
alpha_dir_base = np.array([2.0, -1.0, 1.5], dtype=float)
alpha_dir = alpha_dir_base / np.linalg.norm(alpha_dir_base)

# Bounds robuste (coerenti con lo script ST mv "lean") per log-varianze e nu.
LOG_VAR_MIN = float(os.getenv("ST_MV_LOG_VAR_MIN", "-10.0"))  # var >= exp(-10)
LOG_VAR_MAX = float(os.getenv("ST_MV_LOG_VAR_MAX", "10.0"))   # var <= exp(+10)
NU_MIN      = float(os.getenv("ST_MV_NU_MIN", "2.05"))        # >2 per var finita
NU_MAX      = float(os.getenv("ST_MV_NU_MAX", "40.0"))

# message helper
def _format_tag(a: float) -> str:
    """Tag canonico che preserva il segno (es. -1.2 -> '-1p2')."""
    return f"{a:.1f}".replace(".", "p")

def tag_of(a: float) -> str:
    """Helper legacy: converte il segno meno in 'm' (vecchio naming tipo 'm1p2')."""
    return _format_tag(a).replace("-", "m")

def _candidate_dataset_names(a: float):
    """Genera i possibili nomi di file per il dataset (prima convenzione nuova, poi legacy)."""
    canonical = _format_tag(a)        # es: "-1p2"
    legacy = canonical.replace("-", "m")  # es: "m1p2"
    seen = set()
    for tag in (canonical, legacy):
        fname = f"st_mv_A_alpha_{tag}.csv"
        if fname not in seen:
            seen.add(fname)
            yield fname

def read_Y(a: float) -> pd.DataFrame:
    """
    Prova a leggere il dataset per un certo alpha_scale.
    Verifica tutte le convenzioni di naming possibili e restituisce un DataFrame.
    """
    candidates = list(_candidate_dataset_names(a))
    for fname in candidates:
        fp = DATA_DIR / fname
        if fp.exists():
            return pd.read_csv(fp)
    # Se nessun file esiste, errore esplicativo.
    raise FileNotFoundError(
        f"Dataset mancante per alpha_scale={a:.1f}. "
        "Esegui prima lo script R grid. "
        f"Ricercati: {', '.join(str(DATA_DIR / fn) for fn in candidates)}"
    )

def fmt(x, k=6):
    """Formattazione numerica 'g' simile a quella degli script R."""
    try:
        return f"{float(x):.{k}g}"
    except Exception:
        return str(x)

# Parametrizzazione del vettore di parametri:
#   theta = (mu[0..p-1], log_var[0..p-1], alpha[0..p-1], nu)
# con:
#   - mu: vettore location
#   - log_var: log delle varianze marginali
#   - alpha: vettore skew
#   - nu: gradi di libertà

def pack(mu, log_var, alpha, nu):
    """Impacchetta mu, log_var, alpha, nu in un unico vettore theta."""
    return np.concatenate([mu, log_var, alpha, np.atleast_1d(nu)])

def unpack(theta, p):
    """Estrae mu, log_var, alpha, nu da un vettore theta dato p."""
    mu = theta[0:p]
    log_var = theta[p:2*p]
    alpha = theta[2*p:3*p]
    nu = float(theta[3*p])
    return mu, log_var, alpha, nu

def stable_var(log_var):
    """Applica clip a log_var e poi exp per ottenere varianze positive e finite."""
    lv = np.clip(log_var, LOG_VAR_MIN, LOG_VAR_MAX)
    return np.exp(lv)

def stable_nu(nu):
    """Clampa nu nell'intervallo [NU_MIN, NU_MAX] per stabilità numerica."""
    return float(np.clip(nu, NU_MIN, NU_MAX))

def make_nll(Y):
    """
    Costruisce una funzione negativa log-verosimiglianza che chiude Y nello scope.
    La NLL risultante sarà compatibile con minimize (accetta theta e restituisce uno scalare).
    """
    def nll(theta):
        p = Y.shape[1]
        # Decodifica i parametri dal vettore theta.
        mu, log_var, alpha, nu = unpack(theta, p)
        # Varianze marginali stabilizzate.
        var = stable_var(log_var)
        if not np.all(np.isfinite(var)) or np.any(var <= 0):
            return np.inf
        # Stabilizza nu.
        nu = stable_nu(nu)
        # Costruisce Sigma diagonale dalle varianze.
        Sigma = np.diag(var).astype(float)
        try:
            # Usa la logpdf del modello skew-t multivariato locale.
            lp = mst.logpdf(Y, df=nu, loc=mu, scale=Sigma, shape=alpha)
            if not np.all(np.isfinite(lp)):
                return np.inf
            ll = float(np.sum(lp))
        except Exception:
            # In caso di errori numerici/dominali, penalizza con infinito.
            return np.inf
        # Restituisce la negativa log-verosimiglianza (obiettivo del minimize).
        return -ll
    return nll

# loop principale sulla griglia di alpha_scale
rows = []   # lista di dizionari per le righe della tabella finale
times = []  # lista di dizionari per tempi per alpha_scale e step

for a in alpha_grid:
    # (A) Lettura dati
    # Misura il tempo di lettura del dataset per questo alpha_scale.
    t0 = time.perf_counter()
    df = read_Y(a)
    t_read = time.perf_counter() - t0

    # Converte il DataFrame in array NumPy (solo valori float).
    Y = df.to_numpy(dtype=float)
    n, p = Y.shape

    # (B) Start e bounds
    # Stime iniziali:
    #   - mu0: media campionaria.
    #   - var0: varianza campionaria (con piccolo offset).
    #   - log_var0: log(var0) clippato.
    #   - alpha0: vettore di zeri.
    #   - nu0: valore iniziale moderato (8).
    mu0 = Y.mean(axis=0)
    var0 = Y.var(axis=0) + 1e-6
    log_var0 = np.log(var0)
    log_var0 = np.clip(log_var0, LOG_VAR_MIN + 1.0, LOG_VAR_MAX - 1.0)
    alpha0 = np.zeros(p, dtype=float)
    nu0 = 8.0

    # Vettore iniziale per l'ottimizzazione.
    x0 = pack(mu0, log_var0, alpha0, nu0)

    # Bounds per L-BFGS-B:
    #   - mu: non vincolato
    #   - log_var: [LOG_VAR_MIN, LOG_VAR_MAX]
    #   - alpha: non vincolato
    #   - nu: [NU_MIN, NU_MAX]
    bounds = (
        [(None, None)] * p +
        [(LOG_VAR_MIN, LOG_VAR_MAX)] * p +
        [(None, None)] * p +
        [(NU_MIN, NU_MAX)]
    )

    # (C) Fit MLE (L-BFGS-B)
    # Costruisce la NLL fissando Y e lancia minimize.
    nll = make_nll(Y)
    t1 = time.perf_counter()
    opt = minimize(
        nll, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 8000, "ftol": 1e-9, "maxfun": 300000}
    )
    t_fit = time.perf_counter() - t1

    # Estrae le stime finali dai parametri ottimizzati.
    mu_hat, log_var_hat, alpha_hat, nu_hat = unpack(opt.x, p)
    var_hat = stable_var(log_var_hat)
    Sigma_hat = np.diag(var_hat)
    nu_hat = stable_nu(nu_hat)
    loglik = -float(opt.fun)

    # (D) Riga output
    # Costruisce una riga "larga" con tutte le info di interesse per questo alpha_scale:
    # - parametri stimati
    # - dimensione, loglik
    # - diagonale di Sigma (omega_hat_diag)
    row = {
        "engine": "Python",
        "alpha_scale": float(a),
        "p": int(p),
        "n": int(n),
        "loglik": float(loglik),
        "nu_hat": float(nu_hat),
        "omega_hat_diag": ";".join(fmt(v) for v in np.diag(Sigma_hat)),
    }
    # Aggiunge le componenti di mu_hat e alpha_hat (xi_hat_j e alpha_hat_j).
    for j in range(p):
        row[f"xi_hat_{j+1}"] = float(mu_hat[j])
        row[f"alpha_hat_{j+1}"] = float(alpha_hat[j])

    # alpha_true per trasparenza/confronto:
    # vettore alpha true = alpha_scale * direzione normalizzata (alpha_dir).
    alpha_true_vec = float(a) * alpha_dir
    for j in range(p):
        row[f"alpha_true_{j+1}"] = float(alpha_true_vec[j])

    # Aggiunge la riga alla lista complessiva.
    rows.append(row)

    # (E) Tempi
    # Registra i tempi di lettura e di fit per questo alpha_scale.
    times.append({"alpha_scale": float(a), "step": "io_read_csv_data",       "seconds": float(t_read)})
    times.append({"alpha_scale": float(a), "step": "fit_MST_LBFGSB_diag",    "seconds": float(t_fit)})

    # Log sintetico a video per questo scenario.
    print(f"✓ a={a:+.1f}  n={n} p={p}  loglik={loglik:.6f}  nu_hat={nu_hat:.6g}", flush=True)

# Converte rows e times in DataFrame e li salva nei percorsi previsti.
out_tab = TAB_DIR / "st_mv_A_grid_Py.csv"
out_tim = LOG_DIR / "st_mv_A_grid_times_Py.csv"
pd.DataFrame(rows).to_csv(out_tab, index=False)
pd.DataFrame(times).to_csv(out_tim, index=False)

# Messaggio finale riassuntivo.
print(f"UNIF — ST multivariata — grid alpha: scritto\n  → {out_tab}\n  → {out_tim}")
