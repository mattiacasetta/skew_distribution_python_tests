# SN univariata (dataset, stima, diagnostica, timing)
# Primitive: scipy.stats.skewnorm {pdf, cdf, ppf, rvs, logpdf}

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # usa un backend non interattivo (salva su file, niente GUI)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# Dizionario con tutti i percorsi usati dallo script:
#   - dataset in ingresso
#   - figure diagnostiche
#   - tabelle di risultati
#   - file di log dei tempi
paths = {
    "data":          "common/data/sn_uni.csv",
    "fig_QQ_sn":     "common/fig/sn_uni_QQ_Py.png",
    "fig_UNIF":      "common/fig/sn_uni_UNIF_Py.png",
    "fig_QQ_chi2":   "common/fig/sn_uni_QQ_chi2_Py.png",
    "fig_profile":   "common/fig/sn_uni_profile_alpha_Py.png",
    "tab_fit":       "common/tables/sn_uni_fit_Py.csv",
    "log_times":     "common/logs/sn_uni_times_Py.csv",
    "res_inversion": "common/tables/sn_uni_inversion_Py.csv",
    "res_unif":      "common/tables/sn_uni_unif_Py.csv",
    "res_reflect":   "common/tables/sn_uni_reflection_Py.csv",
    "res_profile":   "common/tables/sn_uni_profile_alpha_Py.csv",
}
# Crea tutte le cartelle necessarie (idempotente: non dà errore se già esistono)
for d in {os.path.dirname(p) for p in paths.values()}:
    if d:
        os.makedirs(d, exist_ok=True)

# lista globale che accumula i tempi per ogni step
timings = []

# funzioni tic/tock per misurare intervalli temporali
def tic() -> float: 
    return time.perf_counter()

def tock(t0: float) -> float: 
    return time.perf_counter() - t0

# aggiunge una riga (step, seconds) alla lista timings
def add_time(step: str, sec: float) -> None:
    timings.append({"step": step, "seconds": float(sec)})

# Carica dataset condiviso
# Legge il dataset SN generato dallo script R, con colonna 'y'
t0 = tic()
df = pd.read_csv(paths["data"])
add_time("io_read_csv_data", tock(t0))

# estrae il vettore y come array NumPy e la dimensione campionaria n
y = df["y"].to_numpy()
n = y.size

# ML estimation (skewnorm.fit)
# Stima dei parametri della skew-normal tramite MLE con skewnorm.fit:
#   a    ≡ alpha (slant)
#   loc  ≡ xi (location)
#   scale ≡ omega (scale)
t0 = tic()
a_hat, loc_hat, scale_hat = stats.skewnorm.fit(y)  # a ≡ alpha, loc ≡ xi, scale ≡ omega
add_time("fit_skewnorm", tock(t0))

# Calcolo della log-verosimiglianza al massimo usando la densità logaritmica
t0 = tic()
loglik = float(np.sum(stats.skewnorm.logpdf(y, a=a_hat, loc=loc_hat, scale=scale_hat)))
add_time("logpdf_sum", tock(t0))

# Diagnostica: QQ teorico SN
# Costruisce il QQ-plot dei dati rispetto alla skew-normal stimata
# 1) quantili teorici SN: q_th = F^{-1}(p; theta_hat)
# 2) quantili empirici ordinati di y
p = (np.arange(1, n + 1) - 0.5) / n
t0 = tic()
q_th = stats.skewnorm.ppf(p, a=a_hat, loc=loc_hat, scale=scale_hat)
add_time("ppf_vector", tock(t0))

q_emp = np.sort(y)
t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.plot(q_th, q_emp, ".", ms=4)
plt.plot([q_th.min(), q_th.max()], [q_th.min(), q_th.max()], "-", lw=2)
plt.title("QQ-plot SN (Python)")
plt.xlabel("Quantili teorici SN")
plt.ylabel("Quantili empirici")
plt.tight_layout()
plt.savefig(paths["fig_QQ_sn"], dpi=120)
plt.close()
add_time("plot_QQ_SN_save", tock(t0))

# Diagnostica uniformità
# Calcola i PIT (Probability Integral Transform):
#   u_i = F_SN(y_i; theta_hat)
# Se il modello è ben specificato, u dovrebbe essere ~ U(0,1).
t0 = tic()
u = stats.skewnorm.cdf(y, a=a_hat, loc=loc_hat, scale=scale_hat)
add_time("cdf_vector", tock(t0))

# Istogramma di u con 30 classi
t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.hist(u, bins=30)
plt.title("UNIF (Python)")
plt.xlabel("u = F_SN(y; theta_hat)")
plt.tight_layout()
plt.savefig(paths["fig_UNIF"], dpi=120)
plt.close()
add_time("plot_UNIF_save", tock(t0))

# Test KS di uniformità(0,1) sui PIT e salvataggio p-value
ks_p = float(stats.kstest(u, "uniform", args=(0, 1)).pvalue)
pd.DataFrame([{"ks_pvalue": ks_p, "n": int(n)}]).to_csv(paths["res_unif"], index=False)

# Inversione CDF↔PPF
# Verifica numerica dell'inversione CDF↔PPF:
#   1) definisce una griglia p_grid che copre core e code
#   2) calcola q_inv = PPF(p_grid)
#   3) calcola p_back = CDF(q_inv)
#   4) salva errore assoluto |p_back - p_grid|
p_grid = np.concatenate([
    [1e-6, 1e-5, 1e-4],
    np.linspace(0.001, 0.999, 999),
    [1 - 1e-4, 1 - 1e-5, 1 - 1e-6]
])
t0 = tic()
q_inv = stats.skewnorm.ppf(p_grid, a=a_hat, loc=loc_hat, scale=scale_hat)
p_back = stats.skewnorm.cdf(q_inv, a=a_hat, loc=loc_hat, scale=scale_hat)
add_time("invert_ppf_cdf", tock(t0))

inv_err = pd.DataFrame({
    "p": p_grid,
    "p_back": p_back,
    "abs_err": np.abs(p_back - p_grid)
})
inv_err.to_csv(paths["res_inversion"], index=False)

# QQ su z^2 vs chi^2_1
# Trasforma i dati tramite lo z-score stimato:
#   z_hat = (y - xi_hat) / omega_hat
# poi considera z_hat^2 e lo confronta con la distribuzione chi^2_1 via QQ-plot.
z_hat = (y - loc_hat) / scale_hat
z2 = np.sort(z_hat ** 2)
chi2_q = stats.chi2(df=1).ppf((np.arange(1, n + 1) - 0.5) / n)

t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.plot(chi2_q, z2, ".", ms=4)
plt.plot([chi2_q.min(), chi2_q.max()], [chi2_q.min(), chi2_q.max()], "-", lw=2)
plt.title("QQ di z^2 vs chi^2_1 (Python)")
plt.xlabel("Quantili chi^2_1")
plt.ylabel("Quantili di z_hat^2")
plt.tight_layout()
plt.savefig(paths["fig_QQ_chi2"], dpi=120)
plt.close()
add_time("plot_QQ_chi2_save", tock(t0))

# Simmetria parametrica (riflessione)
# Riflette i dati y → -y e rifitta la skew-normal:
# ci si aspetta, idealmente:
#   xi_ref ≈ -xi_hat
#   omega_ref ≈ omega_hat
#   alpha_ref ≈ -alpha_hat
y_neg = -y
t0 = tic()
a2, loc2, scale2 = stats.skewnorm.fit(y_neg)
add_time("fit_ML_reflection", tock(t0))

# Confronto tra stime originali e stime sui dati riflessi
cmp_reflect = pd.DataFrame([{
    "xi_hat": float(loc_hat),       "xi_hat_ref": float(loc2),
    "omega_hat": float(scale_hat),  "omega_hat_ref": float(scale2),
    "alpha_hat": float(a_hat),      "alpha_hat_ref": float(a2),
}])
cmp_reflect.to_csv(paths["res_reflect"], index=False)

# Profilo di verosimiglianza in alpha:
#   - fissa alpha su una griglia
#   - per ogni alpha, ottimizza (xi, log(omega)) con BFGS
#   - calcola loglik(alpha) e la deviance rispetto al massimo globale
alpha_grid = np.linspace(-3.0, 3.0, 61)
prof = []

def nll_cond_alpha(a, par):
    """Negativa log-verosimiglianza condizionata su alpha, con parametrizzazione (xi, log(omega))."""
    xi = par[0]
    om = np.exp(par[1])  # vincolo omega>0 tramite esponenziale
    return -np.sum(stats.skewnorm.logpdf(y, a=a, loc=xi, scale=om))

t0 = tic()
for a in alpha_grid:
    # punto iniziale dell'ottimizzazione: stime globali (loc_hat, log(scale_hat))
    x0 = np.array([loc_hat, np.log(scale_hat)])
    # ottimizzazione BFGS su (xi, log(omega)) mantenendo alpha fissato
    res = minimize(
        lambda v: nll_cond_alpha(a, v),
        x0,
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-8}
    )
    xi_hat_a = float(res.x[0])
    om_hat_a = float(np.exp(res.x[1]))
    ll_a     = -float(res.fun)
    prof.append((a, xi_hat_a, om_hat_a, ll_a))
add_time("profile_alpha", tock(t0))

# Costruisce DataFrame con il profilo: alpha, xi_hat(alpha), omega_hat(alpha), loglik(alpha)
prof_df = pd.DataFrame(prof, columns=["alpha", "xi_hat", "omega_hat", "loglik"])
# Deviance rispetto al massimo globale: D(alpha) = 2[loglik_max - loglik(alpha)]
prof_df["deviance"] = 2.0 * (loglik - prof_df["loglik"])
prof_df.to_csv(paths["res_profile"], index=False)

# Grafico del profilo di verosimiglianza in alpha
t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.plot(prof_df["alpha"], prof_df["deviance"], "-")
plt.axvline(0.0, ls="--")
plt.title("Profilo di verosimiglianza in alpha (Python)")
plt.xlabel("alpha")
plt.ylabel("Deviance D(alpha)")
plt.tight_layout()
plt.savefig(paths["fig_profile"], dpi=120)
plt.close()
add_time("plot_profile_save", tock(t0))

# Salvataggi finali (stime + tempi)
# Tabella di output con stime globali e loglik
out = pd.DataFrame([{
    "engine": "Python",
    "xi_hat": float(loc_hat),
    "omega_hat": float(scale_hat),
    "alpha_hat": float(a_hat),
    "loglik": float(loglik),
    "n": int(n)
}])
out.to_csv(paths["tab_fit"], index=False)

# Salva il log dei tempi
pd.DataFrame(timings).to_csv(paths["log_times"], index=False)

# Messaggi riassuntivi a video
print("OK Python — Dataset:", paths["data"])
print("OK Python — Stime:",   paths["tab_fit"])
print("OK Python — Figure:",  paths["fig_QQ_sn"], ",", paths["fig_UNIF"], ",", paths["fig_QQ_chi2"], ",", paths["fig_profile"])
print("OK Python — Result tables:", paths["res_inversion"], ",", paths["res_unif"], ",", paths["res_reflect"], ",", paths["res_profile"])
print("OK Python — Tempi:",   paths["log_times"])
