# ST univariata (dataset, stima, diagnostica, inversione, riflessione,
#                profilo in alpha, profilo in nu con IC)
# Primitive: skewt_scipy.skewt {pdf, cdf, ppf, rvs, logpdf, fit}

import os, time, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interattivo
import matplotlib.pyplot as plt
from scipy import stats

# -------- import skewt-scipy (nome modulo corretto) --------
try:
    from skewt_scipy.skewt import skewt
except Exception as e:
    raise ImportError(
        "Impossibile importare 'skewt' dal pacchetto 'skewt-scipy'. "
        "Installa/aggiorna con: pip install -U skewt-scipy"
    ) from e

# ------------------------- Configurazione -------------------------
# Dizionario di tutti i percorsi usati (dati, figure, tabelle, log).
paths = {
    "data":             "common/data/st_uni_A.csv",
    "fig_QQ_st":        "common/fig/st_uni_A_QQ_Py.png",
    "fig_UNIF":         "common/fig/st_uni_A_UNIF_Py.png",
    "fig_profile_a":    "common/fig/st_uni_A_profile_alpha_Py.png",
    "fig_profile_n":    "common/fig/st_uni_A_profile_nu_Py.png",
    "tab_fit":          "common/tables/st_uni_A_fit_Py.csv",
    "log_times":        "common/logs/st_uni_A_times_Py.csv",
    "res_inversion":    "common/tables/st_uni_A_inversion_Py.csv",
    "res_unif":         "common/tables/st_uni_A_unif_Py.csv",
    "res_reflect":      "common/tables/st_uni_A_reflection_Py.csv",
    "res_profile_a":    "common/tables/st_uni_A_profile_alpha_Py.csv",
    "res_profile_n":    "common/tables/st_uni_A_profile_nu_Py.csv",
    "res_profile_n_CI": "common/tables/st_uni_A_profile_nu_CI_Py.csv",
    "res_nu_summary":   "common/tables/st_uni_A_nu_summary_Py.csv",
}
# Crea cartelle necessarie se non esistono.
for d in {os.path.dirname(p) for p in paths.values()}:
    if d:
        os.makedirs(d, exist_ok=True)

# Soglia per nu (solo per report, NON vincola la fit).
# Si legge dall'ambiente ST_SEM_NU_MIN, se non presente default 2.05.
SEM_NU_MIN = float(os.getenv("ST_SEM_NU_MIN", "2.05"))
if not np.isfinite(SEM_NU_MIN) or SEM_NU_MIN <= 0:
    SEM_NU_MIN = 2.05

# Lista globale dove memorizziamo i tempi per ciascuno step.
timings = []

# misurare intervalli temporali.
def tic() -> float: 
    return time.perf_counter()

def tock(t0: float) -> float: 
    return time.perf_counter() - t0

# Aggiunge una riga di timing alla lista e stampa a video.
def add_time(step: str, sec: float) -> None:
    timings.append({"step": step, "seconds": float(sec)})
    print(f"⏱️  {step}: {sec:.4f}s", flush=True)

# Stampa messaggi informativi a schermo (con flush immediato).
def info(msg: str) -> None:
    print(msg, flush=True)

# ------------------------- RUN -------------------------
info("=== ST univariata (Python) — start ===")
info(f"• Soglia semantica nu (ST_SEM_NU_MIN) = {SEM_NU_MIN}")
info("• Verifica percorsi output… ok")

# Carica dataset condiviso
# Legge il dataset generato in R (stessa struttura, colonna 'y').
info(f"[A] Lettura dataset: {paths['data']}")
if not os.path.exists(paths["data"]):
    raise FileNotFoundError(
        f"Dataset mancante: {paths['data']}. Genera prima i dati in R (rst) oppure crea il CSV."
    )
t0 = tic()
df = pd.read_csv(paths["data"])
add_time("io_read_csv_data", tock(t0))

# Converte la colonna y in array NumPy e ne memorizza la dimensione.
y = df["y"].to_numpy()
n = y.size
info(f"  → n = {n}")

# Stima ML con skewt.fit
# Usa skewt.fit per stimare (alpha, nu, xi, omega) con massimo di verosimiglianza.
info("[B] Stima ML con skewt.fit(y) …")
t0 = tic()
a_hat, df_hat, loc_hat, scale_hat = skewt.fit(y)  # a ≡ alpha, df ≡ nu, loc ≡ xi, scale ≡ omega
add_time("fit_skewt", tock(t0))
info(f"  → stime: alpha={a_hat:.6g}, nu={df_hat:.6g}, xi={loc_hat:.6g}, omega={scale_hat:.6g}")

# Calcola la log-verosimiglianza al massimo usando la densità logaritmica.
info("    Calcolo log-verosimiglianza al massimo …")
t0 = tic()
loglik = float(np.sum(skewt.logpdf(y, a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)))
add_time("logpdf_sum", tock(t0))
info(f"  → loglik = {loglik:.6f}")

# Diagnostica: QQ teorico ST
info("[C] QQ-plot teorico vs empirico …")
p = (np.arange(1, n + 1) - 0.5) / n
t0 = tic()
q_th = skewt.ppf(p, a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)
add_time("ppf_vector", tock(t0))

q_emp = np.sort(y)
t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.plot(q_th, q_emp, ".", ms=4)
mn, mx = float(np.min(q_th)), float(np.max(q_th))
plt.plot([mn, mx], [mn, mx], "-", lw=2)
plt.title("QQ-plot ST (Python)")
plt.xlabel("Quantili teorici ST")
plt.ylabel("Quantili empirici")
plt.tight_layout()
plt.savefig(paths["fig_QQ_st"], dpi=120)
plt.close()
add_time("plot_QQ_ST_save", tock(t0))
info(f"  → salvato: {paths['fig_QQ_st']}")

# Diagnostica: UNIF + KS
# Applica la CDF ST ai dati stimati e produce:
# - istogramma (u = F(y; theta_hat))
# - test KS di uniformità(0,1).
info("[D] UNIF + test KS di uniformità …")
t0 = tic()
u = skewt.cdf(y, a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)
add_time("cdf_vector", tock(t0))

t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.hist(u, bins=30)
plt.title("UNIF (Python)")
plt.xlabel("u = F_ST(y; theta_hat)")
plt.tight_layout()
plt.savefig(paths["fig_UNIF"], dpi=120)
plt.close()
add_time("plot_UNIF_save", tock(t0))
info(f"  → salvato: {paths['fig_UNIF']}")

# Test KS contro Uniform(0,1) e salvataggio p-value.
ks_p = float(stats.kstest(u, "uniform", args=(0, 1)).pvalue)
pd.DataFrame([{"ks_pvalue": ks_p, "n": int(n)}]).to_csv(paths["res_unif"], index=False)
info(f"  → KS p-value = {ks_p:.4g}; scritto: {paths['res_unif']}")

# Inversione CDF↔PPF
info("[E] Inversione CDF↔PPF su griglia code+core …")
p_grid = np.concatenate([
    [1e-6, 1e-5, 1e-4],
    np.linspace(0.001, 0.999, 999),
    [1 - 1e-4, 1 - 1e-5, 1 - 1e-6]
])
t0 = tic()
q_inv  = skewt.ppf(p_grid, a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)
p_back = skewt.cdf(q_inv,   a=a_hat, df=df_hat, loc=loc_hat, scale=scale_hat)
add_time("invert_ppf_cdf", tock(t0))

# Salva la griglia con p, p_back e errore assoluto per analisi successiva.
pd.DataFrame({
    "p": p_grid,
    "p_back": p_back,
    "abs_err": np.abs(p_back - p_grid)
}).to_csv(paths["res_inversion"], index=False)
info(f"  → scritto: {paths['res_inversion']}")

# Simmetria parametrica (riflessione)
# Riflette i dati (y → -y) e rifitta il modello: ci si aspetta alpha_ref ≈ -alpha,
# xi_ref ≈ -xi, omega e nu coerenti.
info("[F] Riflesso dei dati (y → -y) + refit …")
y_neg = -y
t0 = tic()
a2, df2, loc2, scale2 = skewt.fit(y_neg)
add_time("fit_ML_reflection", tock(t0))

# Salva confronto tra stime originali e stime sui dati riflessi.
pd.DataFrame([{
    "xi_hat": float(loc_hat),      "xi_hat_ref": float(loc2),
    "omega_hat": float(scale_hat), "omega_hat_ref": float(scale2),
    "alpha_hat": float(a_hat),     "alpha_hat_ref": float(a2),
    "nu_hat": float(df_hat),       "nu_hat_ref": float(df2),
}]).to_csv(paths["res_reflect"], index=False)
info(f"  → salvato confronto riflessione: {paths['res_reflect']}")

# Profilo in alpha
# Costruisce il profilo di verosimiglianza in alpha:
# - alpha bloccato su una griglia
# - fit su (nu, xi, omega)
# - calcola loglik per ciascun alpha fissato.
info("[G] Profilo di verosimiglianza in alpha (alpha bloccato, fit su nu, xi, omega) …")
alpha_grid = np.linspace(-3.0, 3.0, 61)
prof_a = []

# Funzione di supporto per loglik dato (alpha, nu, xi, omega).
def ll_given(a: float, df_: float, loc_: float, scale_: float) -> float:
    return float(np.sum(skewt.logpdf(y, a=a, df=df_, loc=loc_, scale=scale_)))

t0 = tic()
for a in alpha_grid:
    # Fit con alpha bloccato (fa=a): ottimizza df, loc, scale
    a_fix, df_fix, loc_fix, scale_fix = skewt.fit(y, fa=float(a))
    ll = ll_given(a, df_fix, loc_fix, scale_fix)
    prof_a.append((a, loc_fix, scale_fix, df_fix, ll))
add_time("profile_alpha", tock(t0))

# Converte il profilo in DataFrame e calcola la deviance rispetto al massimo.
prof_a_df = pd.DataFrame(prof_a, columns=["alpha", "xi_hat", "omega_hat", "nu_hat", "loglik"])
prof_a_df["deviance"] = 2.0 * (loglik - prof_a_df["loglik"])
prof_a_df.to_csv(paths["res_profile_a"], index=False)
info(f"  → scritto profilo alpha: {paths['res_profile_a']}")

# Grafico del profilo di verosimiglianza in alpha.
t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.plot(prof_a_df["alpha"], prof_a_df["deviance"], "-")
plt.axvline(0.0, ls="--")
plt.title("Profilo di verosimiglianza in alpha (Python, ST)")
plt.xlabel("alpha")
plt.ylabel("Deviance D(alpha)")
plt.tight_layout()
plt.savefig(paths["fig_profile_a"], dpi=120)
plt.close()
add_time("plot_profile_alpha_save", tock(t0))
info(f"  → salvato: {paths['fig_profile_a']}")

# Profilo in nu + IC 95%
# Profilo di verosimiglianza in nu:
# - nu bloccato su una griglia
# - fit su (alpha, xi, omega)
# - da questo profilo, costruisce IC al 95% e un semaforo sui momenti.
info("[H] Profilo di verosimiglianza in nu (nu bloccato, fit su alpha, xi, omega) …")
nu_grid = np.exp(np.linspace(np.log(max(SEM_NU_MIN, 2.05)), np.log(40.0), 60))
prof_n = []

t0 = tic()
for nv in nu_grid:
    # Fit con df bloccato (fdf=nv): ottimizza a, loc, scale
    a_fix, df_fix, loc_fix, scale_fix = skewt.fit(y, fdf=float(nv))
    ll = ll_given(a_fix, nv, loc_fix, scale_fix)
    prof_n.append((nv, loc_fix, scale_fix, a_fix, ll))
add_time("profile_nu", tock(t0))

# Converte il profilo in DataFrame e calcola la deviance rispetto al massimo.
prof_n_df = pd.DataFrame(prof_n, columns=["nu", "xi_hat", "omega_hat", "alpha_hat", "loglik"])
prof_n_df["deviance"] = 2.0 * (loglik - prof_n_df["loglik"])
prof_n_df.to_csv(paths["res_profile_n"], index=False)
info(f"  → scritto profilo nu: {paths['res_profile_n']}")

# IC da profilo per nu (df=1 → uso distribuzione chi-quadro con 1 g.d.l.).
chi2_1_095 = stats.chi2(df=1).ppf(0.95)
Dmin = float(np.nanmin(prof_n_df["deviance"]))
ok = np.where(prof_n_df["deviance"].to_numpy() <= Dmin + chi2_1_095)[0]
if ok.size >= 1:
    nu_CI_low = float(np.min(prof_n_df["nu"].to_numpy()[ok]))
    nu_CI_upp = float(np.max(prof_n_df["nu"].to_numpy()[ok]))
else:
    nu_CI_low = float("nan")
    nu_CI_upp = float("nan")

pd.DataFrame([{
    "nu_hat": float(df_hat),
    "nu_CI_low": nu_CI_low,
    "nu_CI_upp": nu_CI_upp,
    "Dmin": Dmin,
    "chi2_1_095": float(chi2_1_095),
}]).to_csv(paths["res_profile_n_CI"], index=False)
info(f"  → IC profilo nu: [{nu_CI_low:.4g}, {nu_CI_upp:.4g}] (se definiti)")

# Semaforo (momenti) + trend verso SN:
# - VERDE: IC completamente > 4 (momenti fino al 4° ben definiti)
# - ROSSO: IC completamente ≤ 4 (kurtosi non affidabile)
# - GIALLO: caso intermedio.
if not (math.isnan(nu_CI_low) or math.isnan(nu_CI_upp)):
    if nu_CI_low > 4.0:
        semaforo = "VERDE"
    elif nu_CI_upp <= 4.0:
        semaforo = "ROSSO"
    else:
        semaforo = "GIALLO"
else:
    semaforo = "NA"

# trend_to_SN: True se il minimo del profilo è all'estremo superiore della griglia di nu
# (tendenza verso skew-normal quando nu → infinito).
trend_to_SN = int(np.nanargmin(prof_n_df["deviance"].to_numpy())) == (len(nu_grid) - 1)

pd.DataFrame([{
    "nu_hat": float(df_hat),
    "nu_CI_low": nu_CI_low,
    "nu_CI_upp": nu_CI_upp,
    "semaforo": semaforo,
    "trend_to_SN": bool(trend_to_SN),
    "nu_sem_threshold": float(SEM_NU_MIN),
}]).to_csv(paths["res_nu_summary"], index=False)
info(f"  → semaforo momenti = {semaforo}; trend_to_SN={trend_to_SN}")

# Grafico del profilo di verosimiglianza in nu con:
# - soglia Dmin + chi2_1_0.95 (linea orizzontale)
# - nu_hat (linea verticale)
# - segmento orizzontale che evidenzia l'intervallo di confidenza.
t0 = tic()
plt.figure(figsize=(7.5, 6))
plt.plot(prof_n_df["nu"], prof_n_df["deviance"], "-")
plt.axhline(Dmin + chi2_1_095, ls=":", color="black")
plt.axvline(df_hat, ls="--", color="black")
if not (math.isnan(nu_CI_low) or math.isnan(nu_CI_upp)):
    plt.hlines(Dmin + chi2_1_095, nu_CI_low, nu_CI_upp, linewidth=3)
plt.title("Profilo di verosimiglianza in nu (Python, ST)")
plt.xlabel("nu")
plt.ylabel("Deviance D(nu)")
plt.tight_layout()
plt.savefig(paths["fig_profile_n"], dpi=120)
plt.close()
add_time("plot_profile_nu_save", tock(t0))
info(f"  → salvato: {paths['fig_profile_n']}")

# Salvataggi finali (stime + tempi)
# Salva le stime finali e la log-verosimiglianza in un'unica riga (per merge con R).
info("[I] Salvataggi finali (stime + tempi) …")
pd.DataFrame([{
    "engine": "Python",
    "xi_hat": float(loc_hat),
    "omega_hat": float(scale_hat),
    "alpha_hat": float(a_hat),
    "nu_hat": float(df_hat),
    "loglik": float(loglik),
    "n": int(n),
    "nu_sem_threshold": float(SEM_NU_MIN),
}]).to_csv(paths["tab_fit"], index=False)

# Salva anche tutti i tempi misurati durante gli step.
pd.DataFrame(timings).to_csv(paths["log_times"], index=False)
info(f"  → stime: {paths['tab_fit']}")
info(f"  → tempi: {paths['log_times']}")

# Riepilogo tempi a video, step per step.
info("\n— RIEPILOGO TEMPI —")
for row in timings:
    info(f"  {row['step']:<24} {row['seconds']:.4f}s")

info("\n=== ST univariata (Python) — done ===")
