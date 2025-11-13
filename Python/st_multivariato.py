# UNIF — ST multivariata: dataset condiviso, MLE (Sigma diagonale) con modulo locale,
# pairs plot opzionale, salvataggi, timing.
# Usa SOLO i file locali in:
#   multivariate_skew_distributions/  (oppure) multivariate_skew_distribution/
#     ├─ multivariate_skew_t.py
#     ├─ multivariate_t.py
#     └─ multivariate_skew_normal.py
#
# Output:
#   - common/tables/st_mv_A_fit_Py.csv
#   - common/tables/st_mv_A_fit_Omega_Py.csv
#   - common/logs/st_mv_A_times_Py.csv
#   - (fig opzionale) common/fig/st_mv_A_pairs_Py.png

import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # usa backend non interattivo (salva solo su file, niente GUI)
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.optimize import minimize

# Se True, genera anche il pairs plot (diagnostica grafica veloce).
DO_PAIRS = True   # pairs plot su sottocampione

# Determina la root del progetto (due livelli sopra questo file).
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Cartelle candidate che dovrebbero contenere i moduli multivariati locali.
CANDIDATES = [
    PROJ_ROOT / "multivariate_skew_distributions",
    PROJ_ROOT / "multivariate_skew_distribution",
]

# Cerca multivariate_skew_t.py nelle cartelle candidate; se trovato, lo aggiunge al sys.path.
found = None
for cand in CANDIDATES:
    if (cand / "multivariate_skew_t.py").exists():
        sys.path.insert(0, str(cand))
        found = cand
        break

# Se non trova il file, solleva errore con spiegazione sul layout atteso.
if found is None:
    raise FileNotFoundError(
        "Impossibile trovare multivariate_skew_t.py. "
        "Atteso in:\n  - multivariate_skew_distributions/\n  - multivariate_skew_distribution/\n"
        f"Root progetto: {PROJ_ROOT}"
    )

# Importa SOLO la versione 'generator' del modello skew-t multivariato.
from multivariate_skew_t import multivariate_skew_t as mst  # modulo locale

# Definisce i percorsi per dati, figure, tabelle e log (relativi alla root progetto).
paths = {
    "data":         str(PROJ_ROOT / "common/data/st_mv_A.csv"),
    "fig_pairs":    str(PROJ_ROOT / "common/fig/st_mv_A_pairs_Py.png"),
    "tab_fit":      str(PROJ_ROOT / "common/tables/st_mv_A_fit_Py.csv"),
    "tab_fit_Om":   str(PROJ_ROOT / "common/tables/st_mv_A_fit_Omega_Py.csv"),
    "log_times":    str(PROJ_ROOT / "common/logs/st_mv_A_times_Py.csv"),
}
# Crea tutte le cartelle necessarie in modo idempotente.
for p in paths.values():
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

# Piccole utility per logging e timing.
def info(msg): 
    print(msg, flush=True)

def tic(): 
    return time.perf_counter()

def tock(t0): 
    return time.perf_counter() - t0

# Lista globale dove accumuliamo i tempi per ciascuno step.
timings = []

# Dati: leggi se esiste, altrimenti genera
info("=== UNIF — ST multivariata (Python) — start ===")
if os.path.exists(paths["data"]):
    # Caso 1: esiste già il CSV → lo leggiamo (coerenza con R).
    info(f"[A] Lettura dataset: {paths['data']}")
    t0 = tic()
    df = pd.read_csv(paths["data"])
    timings.append({"step": "io_read_csv_data", "seconds": tock(t0)})
else:
    # Caso 2: CSV assente → generiamo dati sintetici con mst.rvs e li salviamo.
    info(f"[A] Dataset non trovato → genero dati sintetici e salvo in {paths['data']}")
    # Configurazione parametri veri per la generazione (n, p, xi, Omega, alpha, nu).
    n = 2000
    p = 3
    xi_true = np.array([0.0, 1.0, -1.0], dtype=float)
    Omega_true = np.array([
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ], dtype=float)
    alpha_true = np.array([2.0, -1.0, 1.5], dtype=float)
    nu_true = 8.0
    rng = np.random.default_rng(20250303)

    # Generazione con l’interfaccia “generator”: mst.rvs
    # Parametri: df=nu, loc=xi, scale=Omega, shape=alpha, size=n.
    t0 = tic()
    Ygen = mst.rvs(df=nu_true, loc=xi_true, scale=Omega_true, shape=alpha_true,
                   size=n, random_state=rng)
    timings.append({"step": "gen_rmst", "seconds": tock(t0)})

    # Converte in DataFrame con colonne y01, y02, ...
    df = pd.DataFrame(Ygen, columns=[f"y{j:02d}" for j in range(1, Ygen.shape[1]+1)])
    df.to_csv(paths["data"], index=False)
    info(f"  → generati n={n}, p={Ygen.shape[1]}")

# Converte i dati in array NumPy per l’uso nella log-verosimiglianza.
Y = df.to_numpy()
n, p = Y.shape
info(f"  → dataset: n={n}, p={p}")

# MLE (Sigma diagonale) con mst.logpdf — ROBUSTA
# Obiettivo: stimare parametri ST multivariata imponendo Sigma diagonale (varianze indipendenti).
# Parametrizzazione vettore theta:
#   theta = (mu[0..p-1], log_var[0..p-1], alpha[0..p-1], nu)
# dove:
#   - mu: vettore di location (dimensione p)
#   - log_var: log(varianza marginale) per ogni componente (garantisce var>0 via exp)
#   - alpha: vettore di skewness (dimensione p)
#   - nu: gradi di libertà comuni.
# Patch robusta: vincoli su log_var e nu + L-BFGS-B per limitare overflow/underflow.

# Limiti per log-varianza e per nu letti da ENV o default.
LOG_VAR_MIN = float(os.getenv("ST_MV_LOG_VAR_MIN", "-10.0"))  # ⇒ var ≥ exp(-10) ≈ 4.54e-5
LOG_VAR_MAX = float(os.getenv("ST_MV_LOG_VAR_MAX", "10.0"))   # ⇒ var ≤ exp(+10) ≈ 2.20e4
NU_MIN      = float(os.getenv("ST_MV_NU_MIN", "2.05"))        # sopra 2 per var finita
NU_MAX      = float(os.getenv("ST_MV_NU_MAX", "40.0"))

# Funzione helper: impacchetta mu, log_var, alpha e nu in un unico vettore theta.
def pack(mu, log_var, alpha, nu):
    return np.concatenate([mu, log_var, alpha, np.atleast_1d(nu)])

# Inversa di pack: dato theta, restituisce mu, log_var, alpha, nu separati.
def unpack(theta):
    mu = theta[0:p]
    log_var = theta[p:2*p]
    alpha = theta[2*p:3*p]
    nu = theta[3*p]
    return mu, log_var, alpha, nu

# Stabilizza log_var applicando clip ai limiti e poi esponenziale.
def stable_var(log_var):
    lv = np.clip(log_var, LOG_VAR_MIN, LOG_VAR_MAX)
    return np.exp(lv)

# Stabilizza nu forzandolo nel range [NU_MIN, NU_MAX].
def stable_nu(nu):
    # clamp per sicurezza (oltre alle bounds dell'optimizer)
    return float(np.clip(nu, NU_MIN, NU_MAX))

# Negativa log-verosimiglianza (obiettivo da minimizzare).
def nll(theta):
    # Scompatta parametri dal vettore.
    mu, log_var, alpha, nu = unpack(theta)
    # Varianze marginali stabili (positive e finite).
    var = stable_var(log_var)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return np.inf
    # Stabilizza nu.
    nu = stable_nu(nu)
    # Costruisce Sigma diagonale dalla varianza marginale.
    Sigma = np.diag(var).astype(float)

    try:
        # Versione 'generator': mst.logpdf(x, df=..., loc=..., scale=..., shape=...)
        lp = mst.logpdf(Y, df=nu, loc=mu, scale=Sigma, shape=alpha)
        if not np.all(np.isfinite(lp)):
            return np.inf
        ll = float(np.sum(lp))
    except Exception:
        # In caso di problemi numerici o di dominio, penalizza pesantemente (ritorna infinito).
        return np.inf

    # Restituisce la negativa log-verosimiglianza.
    return -ll

# Start: momenti campionari per mu e var, alpha=0, nu iniziale moderato (8).
mu0 = Y.mean(axis=0)
var0 = Y.var(axis=0) + 1e-6  # piccolo offset per evitare var=0
log_var0 = np.log(var0)
log_var0 = np.clip(log_var0, LOG_VAR_MIN + 1.0, LOG_VAR_MAX - 1.0)
alpha0 = np.zeros(p)
nu0 = 8.0
x0 = pack(mu0, log_var0, alpha0, nu0)

# Bounds per l’ottimizzatore L-BFGS-B:
# - mu: non vincolato
# - log_var: vincolato tra LOG_VAR_MIN e LOG_VAR_MAX
# - alpha: non vincolato
# - nu: vincolato tra NU_MIN e NU_MAX.
bounds = (
    [(None, None)] * p +
    [(LOG_VAR_MIN, LOG_VAR_MAX)] * p +
    [(None, None)] * p +
    [(NU_MIN, NU_MAX)]
)

info("[B] Stima MLE (L-BFGS-B) su (mu, log_var, alpha, nu) con Sigma diagonale …")
t0 = tic()
opt = minimize(
    nll, x0, method="L-BFGS-B", bounds=bounds,
    options={"maxiter": 8000, "ftol": 1e-9, "maxfun": 300000}
)
timings.append({"step": "fit_MST_LBFGSB_diag", "seconds": tock(t0)})
info(f"  → success={opt.success}, fun={opt.fun:.6f}, nit={opt.nit}")

# Estrae le stime finali e costruisce Sigma stimata.
mu_hat, log_var_hat, alpha_hat, nu_hat = unpack(opt.x)
var_hat = stable_var(log_var_hat)
Sigma_hat = np.diag(var_hat)
nu_hat = stable_nu(nu_hat)
loglik = -float(opt.fun)

# Pairs plot
if DO_PAIRS:
    info("[C] Pairs plot …")
    # Sottocampiona se n > 3000 per non appesantire la grafica.
    sub_idx = np.arange(n) if n <= 3000 else np.random.default_rng(123).choice(
        n, size=3000, replace=False
    )
    t0 = tic()
    # pairs plot con scatter_matrix di pandas.
    scatter_matrix(
        df.iloc[sub_idx, :],
        figsize=(10, 9),
        diagonal='hist',
        range_padding=0.05
    )
    plt.suptitle("UNIF — ST multivariata — pairs plot (Python)", y=1.02)
    plt.tight_layout()
    plt.savefig(paths["fig_pairs"], dpi=120, bbox_inches="tight")
    plt.close()
    timings.append({"step": "plot_pairs_save", "seconds": tock(t0)})
    info(f"  → salvato: {paths['fig_pairs']}")

# Salvataggi
# Funzione di formattazione numerica compatibile con R (g-notation).
def fmt(x, k=6):
    try:
        return f"{float(x):.{k}g}"
    except Exception:
        return str(x)

# Costruisce una riga di output con stime sintetiche (compatibile con script R).
fit_out = {
    "engine": "Python",
    "p": int(p),
    "n": int(n),
    "loglik": float(loglik),
    "nu_hat": float(nu_hat),
    # omega_hat_diag contiene le varianze marginali (diag di Sigma_hat) in formato "v1;v2;...;vp".
    "omega_hat_diag": ";".join(fmt(v) for v in np.diag(Sigma_hat))
}
# Aggiunge xi_hat_j e alpha_hat_j per ogni dimensione j.
for j in range(p):
    fit_out[f"xi_hat_{j+1}"]    = float(mu_hat[j])
for j in range(p):
    fit_out[f"alpha_hat_{j+1}"] = float(alpha_hat[j])

# Salva la tabella delle stime in CSV.
pd.DataFrame([fit_out]).to_csv(paths["tab_fit"], index=False)

# Costruisce la matrice Omega (Sigma_hat) in formato "long" (i,j,value).
Om_long = []
for i in range(p):
    for j in range(p):
        Om_long.append({"i": i+1, "j": j+1, "value": float(Sigma_hat[i, j])})
pd.DataFrame(Om_long).to_csv(paths["tab_fit_Om"], index=False)

# Salva anche i tempi raccolti durante l’esecuzione.
pd.DataFrame(timings).to_csv(paths["log_times"], index=False)

# ------------------------- Messaggi finali -------------------------
info(f"OK Py — Stime:  {paths['tab_fit']} (loglik = {fmt(loglik)}, nu_hat = {fmt(nu_hat)})")
if DO_PAIRS:
    info(f"OK Py — Pairs:  {paths['fig_pairs']}")
info(f"OK Py — Omega:  {paths['tab_fit_Om']}")
info(f"OK Py — Tempi:  {paths['log_times']}")
info("=== UNIF — ST multivariata (Python) — done ===")
