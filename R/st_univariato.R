# ST univariata (dataset, stima, diagnostica, inversione, riflessione,
# profilo in alpha, profilo in nu con IC
# Primitive: dst/pst/qst/rst (pacchetto 'sn').

# Pacchetto 'sn' (skew-t).
library(sn)

## ------------------------- Configurazione -------------------------
set.seed(20250303)

# Parametri della ST usati per simulare i dati.
# n = campioni; xi = posizione; omega = scala (>0);
# alpha = asimmetria; nu = gradi di libertà.
n      <- 2000 # campioni
xi     <- 0 # posizione
omega  <- 1 # scala
alpha  <- 1.2 # asimmetria
nu     <- 6 # gradi di libertà >4 così che esitano tutti

# Soglia minima per nu: garantisce esistenza della varianza (≥2.05)
# o anche della kurtosi (≥4.05). Legge da ENV ST_SEM_NU_MIN, default 2.05.
# suppressWarnings per evitare rumore se variabile non numerica.
# Se valore non finito o ≤0, ripristina 2.05.
SEM_NU_MIN <- suppressWarnings(as.numeric(Sys.getenv("ST_SEM_NU_MIN", unset = "2.05")))
if (!is.finite(SEM_NU_MIN) || SEM_NU_MIN <= 0) SEM_NU_MIN <- 2.05

# Bound inferiori in scala log per omega e nu usati nei vincoli di ottimizzazione.
LOG_OM_MIN <- log(1e-8)         # omega > 0 
LOG_NU_MIN <- log(SEM_NU_MIN)   # nu >= soglia

# Controlli per L-BFGS-B: max iterazioni, precisione, gradiente.
LBFGSB_CTRL <- list( # nolint: object_name_linter.
  maxit = 5000,
  factr = 1e7, # corrisponde a circa 1e-8 in termini di precisione
  pgtol = 1e-8
)

# Percorsi d’output centralizzati (file dati, figure, tabelle, log tempi).
paths <- list(
  data              = "common/data/st_uni_A.csv",
  fig_QQ_st         = "common/fig/st_uni_A_QQ_R.png",
  fig_UNIF          = "common/fig/st_uni_A_UNIF_R.png",
  fig_profile_a     = "common/fig/st_uni_A_profile_alpha_R.png",
  fig_profile_n     = "common/fig/st_uni_A_profile_nu_R.png",
  tab_fit           = "common/tables/st_uni_A_fit_R.csv",
  log_times         = "common/logs/st_uni_A_times_R.csv",
  res_inversion     = "common/tables/st_uni_A_inversion_R.csv",
  res_unif          = "common/tables/st_uni_A_unif_R.csv",
  res_reflect       = "common/tables/st_uni_A_reflection_R.csv",
  res_profile_a     = "common/tables/st_uni_A_profile_alpha_R.csv",
  res_profile_n     = "common/tables/st_uni_A_profile_nu_R.csv",
  res_profile_n_CI  = "common/tables/st_uni_A_profile_nu_CI_R.csv",
  res_nu_summary    = "common/tables/st_uni_A_nu_summary_R.csv"
)

# Crea automaticamente le directory dei percorsi sopra, se non esistono.
lapply(unique(dirname(unlist(paths))), function(d) dir.create(d, recursive = TRUE, showWarnings = FALSE)) # nolint: line_length_linter.

# misura dei tempi per step e funzione di logging
timings <- data.frame(step = character(), seconds = numeric(), stringsAsFactors = FALSE) # nolint: line_length_linter.
tic  <- function() proc.time()[["elapsed"]]
tock <- function(t0) proc.time()[["elapsed"]] - t0
add_time <- function(step, sec) {
  timings <<- rbind(timings, data.frame(step = step, seconds = sec))
}

# >>> NLL (negative log-likelihood) “safe” per la ST.
#     Parametrizzazione: (xi, log(omega), alpha, log(nu))
#     - Controlli di finitudine su log(omega) e log(nu).
#     - Guard-rails su omega > 0 e nu ≥ soglia.
#     - try-catch sulla densità: se fallisce o genera valori non positivi, restituisce penalità alta.
nll_st_par <- function(par, y) {
  xi  <- par[1]
  lom <- par[2]
  al  <- par[3]
  lnu <- par[4]
  if (!is.finite(lom) || !is.finite(lnu)) return(1e50)

  om <- exp(lom); nv <- exp(lnu) #
  if (!is.finite(om) || om <= 0)          return(1e50)
  if (!is.finite(nv) || nv <  SEM_NU_MIN) return(1e50)

  dens <- suppressWarnings(try(dst(y, xi = xi, omega = om, alpha = al, nu = nv), silent = TRUE)) # nolint: line_length_linter.
  if (inherits(dens, "try-error") || any(!is.finite(dens)) || any(dens <= 0)) return(1e50) # nolint # nolint

  -sum(log(dens))
}

# Simula n osservazioni dalla ST con i parametri dati e misura il tempo.
t0 <- tic()
y  <- rst(n, xi = xi, omega = omega, alpha = alpha, nu = nu)
add_time("data_rst_generate", tock(t0))
# Salva i dati simulati su CSV.
t0 <- tic()
write.csv(data.frame(y = y), paths$data, row.names = FALSE)
add_time("io_write_csv_data", tock(t0))

# Stima ML (L-BFGS-B con vincoli)
# Inizializzazione “robusta”: xi≈mediana; log(omega) da IQR/1.349; alpha=0.5;
# log(nu) parte da max(soglia+0.5,6) per favorire stabilità.
start <- c(median(y), log(IQR(y)/1.349), 0.5, log(max(SEM_NU_MIN + 0.5, 6)))

# Ottimizza la NLL con L-BFGS-B imponendo:
# - log(omega) ≥ LOG_OM_MIN (omega > 0)
# - log(nu)    ≥ LOG_NU_MIN (nu ≥ soglia)
# Restituisce stime ML in scala originale per omega, nu.
t0 <- tic()
fit <- optim(
  par     = start,
  fn      = function(p) nll_st_par(p, y),
  method  = "L-BFGS-B",
  lower   = c(-Inf, LOG_OM_MIN, -Inf, LOG_NU_MIN),
  control = LBFGSB_CTRL
)
add_time("fit_ML_LBFGSB", tock(t0))

# Estrae parametri stimati (parhat), log-lik massimo (llhat), conteggio valutazioni e info convergenza.
parhat <- c(
  xi    = fit$par[1],
  omega = exp(fit$par[2]),
  alpha = fit$par[3],
  nu    = exp(fit$par[4])
)
llhat  <- -fit$value
fevals <- if (!is.null(fit$counts["function"])) as.integer(fit$counts["function"]) else NA_integer_
conv   <- fit$convergence
msg    <- if (!is.null(fit$message)) as.character(fit$message) else ""

# Diagnostica: QQ teorico ST vs empirico
# Costruisce i percentili teorici p e i corrispondenti quantili ST q_th,
# poi QQ-plot contro i quantili empirici ordinati.
p <- (1:n - 0.5) / n
t0 <- tic()
q_th <- qst(p, xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"], nu = parhat["nu"])
add_time("qst_vector", tock(t0))
q_emp <- sort(y)

# Salva il QQ-plot su PNG.
t0 <- tic()
png(paths$fig_QQ_st, width = 900, height = 700, res = 120)
plot(q_th, q_emp, pch = 20, main = "QQ-plot ST (R)",
     xlab = "Quantili teorici ST", ylab = "Quantili empirici")
abline(0, 1, lwd = 2, col = "gray40")
dev.off()
add_time("plot_QQ_ST_save", tock(t0))

# Diagnostica: UNIF + KS
# Trasforma i dati stimati via CDF ST (u ~ Uniforme(0,1) se il modello è adeguato).
t0 <- tic()
u <- pst(y, xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"], nu = parhat["nu"])
add_time("pst_vector", tock(t0))

# Istogramma di u con benchmark della densità uniforme attesa.
t0 <- tic()
png(paths$fig_UNIF, width = 900, height = 700, res = 120)
hist(u, breaks = 30, freq = TRUE, main = "UNIF (R)", xlab = "u = F_ST(y; θ̂)")
abline(h = length(u)/30, lwd = 2, col = "gray40") 
dev.off()
add_time("plot_UNIF_save", tock(t0))

# Test di Kolmogorov–Smirnov su Uniforme(0,1); salva p-value e n.
ks_p <- suppressWarnings(ks.test(u, "punif")$p.value)
write.csv(data.frame(ks_pvalue = ks_p, n = n), paths$res_unif, row.names = FALSE)

# Inversione CDF↔PPF
# Griglia di probabilità inclusa in coda profonda; verifica numerica di qst(p) seguito da pst(·).
p_grid <- c(1e-6, 1e-5, 1e-4, seq(0.001, 0.999, by = 0.001), 1 - 1e-4, 1 - 1e-5, 1 - 1e-6)

t0 <- tic()
q_inv  <- suppressWarnings(qst(p_grid, xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"], nu = parhat["nu"]))
p_back <- suppressWarnings(pst(q_inv,     xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"], nu = parhat["nu"]))
add_time("invert_ppf_cdf", tock(t0))

# Salva l’errore assoluto di inversione per controllo di coerenza numerica.
inv_err <- data.frame(p = p_grid, p_back = p_back, abs_err = abs(p_back - p_grid))
write.csv(inv_err, paths$res_inversion, row.names = FALSE)

# Simmetria parametrica (riflessione)
# Riflette i dati (y_neg = -y) e ristima: attende alpha_hat_ref ≈ -alpha_hat
# e (xi, omega, nu) coerenti, come verifica di simmetria parametrica.
y_neg <- -y
t0 <- tic()
fit_neg <- optim(
  par     = c(median(y_neg), log(IQR(y_neg)/1.349), -0.5, log(parhat["nu"])), # nolint: infix_spaces_linter.
  fn      = function(p) nll_st_par(p, y_neg),
  method  = "L-BFGS-B",
  lower   = c(-Inf, LOG_OM_MIN, -Inf, LOG_NU_MIN),
  control = LBFGSB_CTRL
)
add_time("fit_ML_reflection", tock(t0))

# Confronta stime originali vs riflessione e salva tabella.
parhat_neg <- c(
  xi    = fit_neg$par[1],
  omega = exp(fit_neg$par[2]),
  alpha = fit_neg$par[3],
  nu    = exp(fit_neg$par[4])
)
cmp_reflect <- data.frame(
  xi_hat = parhat["xi"], xi_hat_ref = parhat_neg["xi"],
  omega_hat = parhat["omega"], omega_hat_ref = parhat_neg["omega"],
  alpha_hat = parhat["alpha"], alpha_hat_ref = parhat_neg["alpha"],
  nu_hat = parhat["nu"], nu_hat_ref = parhat_neg["nu"]
)
write.csv(cmp_reflect, paths$res_reflect, row.names = FALSE)

# Profilo in alpha
# Costruisce profilo di verosimiglianza in alpha:
# per ogni alpha nella griglia, massimizza su (xi, log(omega), log(nu)), alpha fissato.
alpha_grid <- seq(-3, 3, length.out = 61)
t0 <- tic()
prof_a <- lapply(alpha_grid, function(a) {
  f <- function(pr) {  # pr = (xi, log(omega), log(nu))
    xi <- pr[1]; lom <- pr[2]; lnu <- pr[3]
    nll_st_par(c(xi, lom, a, lnu), y)
  }
  o <- optim(c(parhat["xi"], log(parhat["omega"]), log(parhat["nu"])),
             f, method = "L-BFGS-B",
             lower = c(-Inf, LOG_OM_MIN, LOG_NU_MIN),
             control = LBFGSB_CTRL)
  c(alpha = a, xi_hat = o$par[1], omega_hat = exp(o$par[2]),
    nu_hat = exp(o$par[3]), loglik = -o$value)
})
add_time("profile_alpha", tock(t0))

# Calcola deviance del profilo e salva risultati.
prof_a_df <- as.data.frame(do.call(rbind, prof_a))
prof_a_df$deviance <- 2 * (llhat - prof_a_df$loglik)
write.csv(prof_a_df, paths$res_profile_a, row.names = FALSE)

# Plot del profilo in alpha con linea verticale in 0.
t0 <- tic()
png(paths$fig_profile_a, width = 900, height = 700, res = 120)
plot(prof_a_df$alpha, prof_a_df$deviance, type = "l", lwd = 2,
     main = "Profilo di verosimiglianza in α (R, ST)", xlab = "α", ylab = "Deviance D(α)")
abline(v = 0, lty = 2, col = "gray40")
dev.off()
add_time("plot_profile_alpha_save", tock(t0))

# Profilo in nu + IC da profile likelihood
# Griglia per nu (in scala log per coprire bene la coda bassa): da max(soglia,2.05) a 40.
nu_grid <- exp(seq(log(max(SEM_NU_MIN, 2.05)), log(40), length.out = 60))

# Profilo su nu: per ogni nu fissato massimizza su (xi, log(omega), alpha).
t0 <- tic()
prof_n <- lapply(nu_grid, function(nv) {
  f <- function(pr) {  # pr = (xi, log(omega), alpha)
    xi <- pr[1]; lom <- pr[2]; al <- pr[3]
    nll_st_par(c(xi, lom, al, log(nv)), y)  # blocca log(nu)=log(nv)
  }
  o <- optim(c(parhat["xi"], log(parhat["omega"]), parhat["alpha"]),
             f, method = "L-BFGS-B",
             lower = c(-Inf, LOG_OM_MIN, -Inf),
             control = LBFGSB_CTRL)
  c(nu = nv, xi_hat = o$par[1], omega_hat = exp(o$par[2]),
    alpha_hat = o$par[3], loglik = -o$value)
})
add_time("profile_nu", tock(t0))

# Calcola deviance del profilo in nu e salva la tabella completa.
prof_n_df <- as.data.frame(do.call(rbind, prof_n))
prof_n_df$deviance <- 2 * (llhat - prof_n_df$loglik)
write.csv(prof_n_df, paths$res_profile_n, row.names = FALSE)

# Intervallo di confidenza 95% da profile likelihood:
# {nu: D(nu) ≤ D_min + χ^2_{1,0.95}}. Se vuoto, restituisce NA.
chi2_1_095 <- qchisq(0.95, df = 1)
Dmin <- min(prof_n_df$deviance, na.rm = TRUE)
ok   <- which(prof_n_df$deviance <= Dmin + chi2_1_095)
if (length(ok) >= 1) {
  nu_CI_low <- min(prof_n_df$nu[ok]); nu_CI_upp <- max(prof_n_df$nu[ok])
} else {
  nu_CI_low <- NA_real_; nu_CI_upp <- NA_real_
}
write.csv(data.frame(nu_hat = parhat["nu"], nu_CI_low = nu_CI_low, nu_CI_upp = nu_CI_upp,
                     Dmin = Dmin, chi2_1_095 = chi2_1_095),
          paths$res_profile_n_CI, row.names = FALSE)

# Momenti in base all’IC di nu:
#     - VERDE se tutto l’IC è >4 (kurtosi ben definita)
#     - ROSSO se tutto l’IC ≤4 (kurtosi non affidabile)
#     - GIALLO altrimenti (borderline).
#     trend_to_SN: TRUE se il minimo del profilo è all’estremo superiore della griglia → tendenza verso SN (nu→∞).
semaforo <- if (is.na(nu_CI_low) || is.na(nu_CI_upp)) {
  "NA"
} else if (nu_CI_low > 4) {
  "VERDE"   # momenti fino al 4° ben definiti
} else if (nu_CI_upp <= 4) {
  "ROSSO"   # kurtosi non affidabile
} else {
  "GIALLO"  # kurtosi borderline
}
trend_to_SN <- { i_min <- which.min(prof_n_df$deviance); i_min == nrow(prof_n_df) } # nolint: brace_linter.


# Salva riepilogo su nu (stima, IC, semaforo, trend, soglia semantica usata).
write.csv(data.frame(nu_hat = parhat["nu"],
                     nu_CI_low = nu_CI_low, nu_CI_upp = nu_CI_upp,
                     semaforo = semaforo, trend_to_SN = trend_to_SN,
                     nu_sem_threshold = SEM_NU_MIN),
          paths$res_nu_summary, row.names = FALSE)

# Plot del profilo in nu con:
#     - soglia D_min + χ^2 (linea orizzontale)
#     - stima nu (linea verticale)
#     - segmento orizzontale che evidenzia l’IC.
t0 <- tic()
png(paths$fig_profile_n, width = 900, height = 700, res = 120)
plot(prof_n_df$nu, prof_n_df$deviance, type = "l", lwd = 2,
     main = "Profilo di verosimiglianza in ν (R, ST)",
     xlab = "ν", ylab = "Deviance D(ν)")
abline(h = Dmin + chi2_1_095, lty = 3, col = "gray40")
abline(v = parhat["nu"], lty = 2, col = "gray40")
if (!is.na(nu_CI_low) && !is.na(nu_CI_upp)) {
  segments(nu_CI_low, Dmin + chi2_1_095, nu_CI_upp, Dmin + chi2_1_095, lwd = 3, col = "gray30")
}
dev.off()
add_time("plot_profile_nu_save", tock(t0))

# Salvataggi finali (stime + tempi)
# Raccoglie stime finali, loglik, info di convergenza, n, soglia nu_sem e salva su CSV.
out <- data.frame(
  engine = "R",
  xi_hat = parhat["xi"], omega_hat = parhat["omega"], alpha_hat = parhat["alpha"], nu_hat = parhat["nu"],
  loglik = llhat, n = n, fevals = fevals, convergence = conv, message = msg,
  nu_sem_threshold = SEM_NU_MIN
)
write.csv(out, paths$tab_fit, row.names = FALSE)
write.csv(timings, paths$log_times, row.names = FALSE)

# Messaggi di riepilogo a console con percorsi di output generati.
cat("OK R — Dataset:", paths$data, "\n")
cat("OK R — Stime:",   paths$tab_fit, " (fevals =", fevals, ", conv =", conv, ")\n")
cat("OK R — Figure:",  paths$fig_QQ_st, ",", paths$fig_UNIF, ",", paths$fig_profile_a, ",", paths$fig_profile_n, "\n")
cat("OK R — Result tables:", paths$res_inversion, ",", paths$res_unif, ",", paths$res_reflect, ",",
    paths$res_profile_a, ",", paths$res_profile_n, ",", paths$res_profile_n_CI, ",", paths$res_nu_summary, "\n")
cat("OK R — Tempi:",   paths$log_times, "\n")
