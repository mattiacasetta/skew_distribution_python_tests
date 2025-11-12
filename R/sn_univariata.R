# SN univariata (dataset, stima, diagnostica, timing)
# Primitive: dsn/psn/qsn/rsn del pacchetto 'sn'.  (API e solver in qsn: v. manuale) 


# Note: evitare start con alpha=0 (non regolare, manuale sn). 

library(sn)

set.seed(20250303)

# Parametri per la generazione
n      <- 2000
xi     <- 0
omega  <- 1
alpha  <- 1.2

# Percorsi
paths <- list(
  data = "common/data/sn_uni.csv",
  fig_dir = "common/fig",
  fig_QQ_sn = "common/fig/sn_uni_QQ_R.png",
  fig_UNIF   = "common/fig/sn_uni_UNIF_R.png",
  fig_QQ_chi2 = "common/fig/sn_uni_QQ_chi2_R.png",
  fig_profile = "common/fig/sn_uni_profile_alpha_R.png",
  tab_fit = "common/tables/sn_uni_fit_R.csv",
  log_times = "common/logs/sn_uni_times_R.csv",
  res_inversion = "common/tables/sn_uni_inversion_R.csv",
  res_unif       = "common/ables/sn_uni_unif_R.csv",
  res_reflect   = "common/tables/sn_uni_reflection_R.csv",
  res_profile   = "common/tables/sn_uni_profile_alpha_R.csv"
)

# Crea tutte le cartelle necessarie 
lapply(unique(dirname(unlist(paths))), function(d) dir.create(d, recursive = TRUE, showWarnings = FALSE))

# Funzioni di timing e formattazione
timings <- data.frame(step = character(), seconds = numeric(), stringsAsFactors = FALSE)
tic  <- function() proc.time()[["elapsed"]]
tock <- function(t0) proc.time()[["elapsed"]] - t0
add_time <- function(step, sec) {
  timings <<- rbind(timings, data.frame(step = step, seconds = sec))
}

# Log-verosimiglianza SN (DP: xi, omega, alpha)
loglik_sn <- function(par, y) {
  xi <- par[1]; om <- abs(par[2]); al <- par[3]
  sum(log(dsn(y, xi = xi, omega = om, alpha = al)))
}

# Dati: generazione + salvataggio
t0 <- tic()
y  <- rsn(n, xi = xi, omega = omega, alpha = alpha)
add_time("data_rsn_generate", tock(t0))

t0 <- tic()
write.csv(data.frame(y = y), paths$data, row.names = FALSE)
add_time("io_write_csv_data", tock(t0))

# Stima ML (trasparente)
start <- c(median(y), IQR(y)/1.349, 0.5)

t0 <- tic()
fit <- optim(
  par = start,
  fn  = function(p) -loglik_sn(p, y),
  method = "BFGS",
  control = list(reltol = 1e-10, maxit = 2000)
)
add_time("fit_ML_BFGS", tock(t0))

parhat <- c(xi = fit$par[1], omega = abs(fit$par[2]), alpha = fit$par[3])
llhat  <- -fit$value
fevals <- if (!is.null(fit$counts["function"])) as.integer(fit$counts["function"]) else NA_integer_
conv   <- fit$convergence
msg    <- if (!is.null(fit$message)) as.character(fit$message) else ""

# Diagnostica: QQ teorico SN
p <- (1:n - 0.5) / n
t0 <- tic()
q_th <- qsn(p, xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"])  # solver default
add_time("qsn_vector", tock(t0))

q_emp <- sort(y)

t0 <- tic()
png(paths$fig_QQ_sn, width = 900, height = 700, res = 120)
plot(q_th, q_emp, pch = 20, main = "QQ-plot SN (R)",
     xlab = "Quantili teorici SN", ylab = "Quantili empirici")
abline(0, 1, lwd = 2, col = "gray40")
dev.off()
add_time("plot_QQ_SN_save", tock(t0))

# Diagnostica: UNIF
t0 <- tic()
u <- psn(y, xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"])
add_time("psn_vector", tock(t0))

t0 <- tic()
png(paths$fig_UNIF, width = 900, height = 700, res = 120)
hist(u, breaks = 30, freq = TRUE, main = "UNIF (R)", xlab = "u = F_SN(y; θ̂)")
abline(h = length(u)/30, lwd = 2, col = "gray40")
dev.off()
add_time("plot_UNIF_save", tock(t0))

# Test KS di uniformità(0,1)
ks_p <- suppressWarnings(ks.test(u, "punif")$p.value)
write.csv(data.frame(ks_pvalue = ks_p, n = n), paths$res_unif, row.names = FALSE)

# Inversione CDF↔PPF
# Griglia spinta sulle code; usare solver="RFB" per robustezza (manuale sn).
p_grid <- c(1e-6, 1e-5, 1e-4, seq(0.001, 0.999, by = 0.001), 1 - 1e-4, 1 - 1e-5, 1 - 1e-6)

t0 <- tic()
q_inv  <- qsn(p_grid, xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"], solver = "RFB")
p_back <- psn(q_inv,  xi = parhat["xi"], omega = parhat["omega"], alpha = parhat["alpha"])
add_time("invert_ppf_cdf", tock(t0))

inv_err <- data.frame(p = p_grid, p_back = p_back, abs_err = abs(p_back - p_grid))
write.csv(inv_err, paths$res_inversion, row.names = FALSE)

# QQ su z^2 vs χ^2_1
z_hat <- (y - parhat["xi"]) / parhat["omega"]
z2    <- sort(z_hat^2)
chi2_q <- qchisq((1:n - 0.5) / n, df = 1)

t0 <- tic()
png(paths$fig_QQ_chi2, width = 900, height = 700, res = 120)
plot(chi2_q, z2, pch = 20, main = "QQ di z^2 vs χ^2_1 (R)",
     xlab = "Quantili χ^2_1", ylab = "Quantili di ẑ^2")
abline(0, 1, lwd = 2, col = "gray40")
dev.off()
add_time("plot_QQ_chi2_save", tock(t0))

# Simmetria parametrica (riflessione)
y_neg <- -y
t0 <- tic()
fit_neg <- optim(
  par = c(median(y_neg), IQR(y_neg)/1.349, -0.5),
  fn  = function(p) -loglik_sn(p, y_neg),
  method = "BFGS",
  control = list(reltol = 1e-10, maxit = 2000)
)
add_time("fit_ML_reflection", tock(t0))

parhat_neg <- c(xi = fit_neg$par[1], omega = abs(fit_neg$par[2]), alpha = fit_neg$par[3])
cmp_reflect <- data.frame(
  xi_hat = parhat["xi"], xi_hat_ref = parhat_neg["xi"],
  omega_hat = parhat["omega"], omega_hat_ref = parhat_neg["omega"],
  alpha_hat = parhat["alpha"], alpha_hat_ref = parhat_neg["alpha"]
)
write.csv(cmp_reflect, paths$res_reflect, row.names = FALSE)

# Profilo di verosimiglianza in alpha
alpha_grid <- seq(-3, 3, length.out = 61)
t0 <- tic()
profile <- lapply(alpha_grid, function(a) {
  f <- function(pr) -sum(log(dsn(y, xi = pr[1], omega = abs(pr[2]), alpha = a)))
  o <- optim(c(parhat["xi"], parhat["omega"]), f, method = "BFGS",
             control = list(reltol = 1e-10, maxit = 2000))
  c(alpha = a, xi_hat = o$par[1], omega_hat = abs(o$par[2]), loglik = -o$value)
})
add_time("profile_alpha", tock(t0))

prof_df <- as.data.frame(do.call(rbind, profile))
prof_df$deviance <- 2 * (llhat - prof_df$loglik)
write.csv(prof_df, paths$res_profile, row.names = FALSE)

t0 <- tic()
png(paths$fig_profile, width = 900, height = 700, res = 120)
plot(prof_df$alpha, prof_df$deviance, type = "l", lwd = 2,
     main = "Profilo di verosimiglianza in α (R)",
     xlab = "α", ylab = "Deviance D(α)")
abline(v = 0, lty = 2, col = "gray40")
dev.off()
add_time("plot_profile_save", tock(t0))

# Salvataggi finali (stime + tempi)
out <- data.frame(
  engine = "R",
  xi_hat = parhat["xi"], omega_hat = parhat["omega"], alpha_hat = parhat["alpha"],
  loglik = llhat, n = n, fevals = fevals, convergence = conv, message = msg
)
write.csv(out, paths$tab_fit, row.names = FALSE)

write.csv(timings, paths$log_times, row.names = FALSE)

# Messaggi riassuntivi
cat("OK R — Dataset:", paths$data, "\n")
cat("OK R — Stime:",   paths$tab_fit, " (fevals =", fevals, ", conv =", conv, ")\n")
cat("OK R — Figure:",  paths$fig_QQ_sn, ",", paths$fig_UNIF, ",", paths$fig_QQ_chi2, ",", paths$fig_profile, "\n")
cat("OK R — Result tables:", paths$res_inversion, ",", paths$res_unif, ",", paths$res_reflect, ",", paths$res_profile, "\n")
cat("OK R — Tempi:",   paths$log_times, "\n")