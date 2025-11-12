# ST multivariata (LEAN): dati, fit via mselm/selm (family="ST"),
# pairs plot opzionale, salvataggi, timing. Pacchetto: sn

 library(sn)

# Settaggi diagnostiche opzionali
DO_PAIRS      <- TRUE
DO_MDQQ       <- FALSE
DO_PIT_MV     <- FALSE
DO_REFLECTION <- FALSE


set.seed(20250303)
n <- 2000; p <- 3
xi_true    <- c(0, 1, -1)
Omega_true <- matrix(c(
  1.0, 0.4, 0.2,
  0.4, 1.0, 0.3,
  0.2, 0.3, 1.0
), nrow = p, byrow = TRUE)
alpha_true <- c(2.0, -1.0, 1.5)
nu_true    <- 8

# Definizione percorsi output (dataset, figure, tabelle, log).
paths <- list(
  data        = "common/data/st_mv_A.csv",
  fig_pairs   = "common/fig/st_mv_A_pairs_R.png",
  fig_md_qq   = "common/fig/st_mv_A_MD_QQ_R.png",
  fig_pit     = "common/fig/st_mv_A_PIT_R.png",
  tab_fit     = "common/tables/st_mv_A_fit_R.csv",
  tab_fit_Om  = "common/tables/st_mv_A_fit_Omega_R.csv",
  tab_reflect = "common/tables/st_mv_A_reflection_R.csv",
  log_times   = "common/logs/st_mv_A_times_R.csv"
)
lapply(unique(dirname(unlist(paths))), function(d) dir.create(d, recursive = TRUE, showWarnings = FALSE))

# funzioni di timing e formattazione
timings <- data.frame(step = character(), seconds = numeric(), stringsAsFactors = FALSE)
tic  <- function() proc.time()[["elapsed"]]
tock <- function(t0) proc.time()[["elapsed"]] - t0
add_time <- function(step, sec) timings <<- rbind(timings, data.frame(step = step, seconds = sec))
fmt <- function(x, k = 6) formatC(x, digits = k, format = "g")

# forza una matrice a essere definita positiva (utile per Omega stimate mal condizionate).
make_PD <- function(S, eps_start = 1e-8, max_iter = 6) {
  S <- 0.5*(S + t(S)); eps <- eps_start
  for (i in 0:max_iter) {
    ev <- try(eigen(S, symmetric = TRUE, only.values = TRUE)$values, silent = TRUE)
    if (!inherits(ev, "try-error") && min(ev) > eps_start) return(S)
    S <- S + (abs(min(ev)) + eps) * diag(ncol(S)); eps <- eps * 10
  }
  S
}

# Dati
# Generazione dati ST multivariata con parametri veri (xi_true, Omega_true, alpha_true, nu_true).
t0 <- tic()
Y  <- sn::rmst(n, xi = xi_true, Omega = Omega_true, alpha = alpha_true, nu = nu_true)
add_time("data_rmst_generate", tock(t0))

# Salvataggio dataset simulato su CSV.
colnames(Y) <- sprintf("y%02d", seq_len(ncol(Y)))
t0 <- tic(); write.csv(as.data.frame(Y), paths$data, row.names = FALSE); add_time("io_write_csv_data", tock(t0))

# (B) Stima ML via selm (family = "ST")
# Nota: in questa configurazione usiamo selm con sola intercetta.
Ydf <- as.data.frame(Y)
form <- as.formula(paste0("cbind(", paste(names(Ydf), collapse = ","), ") ~ 1"))

# Fit con selm (MLE per MST con intercetta) e log dei tempi.
t0  <- tic()
fit <- sn::selm(form, data = Ydf, family = "ST")    # MLE per MST con intercetta
add_time("fit_MST_MLE", tock(t0))                   # step-name usato dai merge

# Estrazione parametri DP (xi, Omega, alpha, nu)
cf <- sn::coef(fit, "DP", vector = FALSE)
# Con ~1: 'beta' è il vettore xi (intercette), 'Omega' matrice, 'alpha' vettore, 'nu' scalare.
xi_hat    <- as.numeric(cf$beta)
Omega_hat <- as.matrix(cf$Omega)
alpha_hat <- as.numeric(cf$alpha)
nu_hat    <- as.numeric(cf$nu)

# Calcolo log-verosimiglianza (se non salvata automaticamente nel modello).
llhat <- if (!is.null(fit@logL)) as.numeric(fit@logL) else {
  dens <- sn::dmst(as.matrix(Y), xi = xi_hat, Omega = Omega_hat, alpha = alpha_hat, nu = nu_hat)
  sum(log(dens))
}

# (C) Diagnostiche leggere
# Pairs plot per ispezione qualitativa delle dipendenze e asimmetrie.
if (DO_PAIRS) {
  t0 <- tic()
  png(paths$fig_pairs, width = 1000, height = 900, res = 120)
  sub_idx <- if (nrow(Y) > 3000) sample.int(nrow(Y), 3000) else seq_len(nrow(Y))
  pairs(as.data.frame(Y[sub_idx, , drop = FALSE]), main = "ST multivariata — pairs plot (R)")
  dev.off()
  add_time("plot_pairs_save", tock(t0))
}

# QQ-plot di Mahalanobis per confrontare D² con chi² teorica.
if (DO_MDQQ) {
  Omega_PD <- make_PD(Omega_hat, 1e-8)
  Sinv <- try(solve(Omega_PD), silent = TRUE)
  if (!inherits(Sinv, "try-error")) {
    Z   <- sweep(as.matrix(Y), 2, xi_hat, "-")
    md2 <- rowSums((Z %*% Sinv) * Z); md2 <- md2[is.finite(md2)]; m <- length(md2)
    if (m >= 5) {
      probs  <- (seq_len(m) - 0.5) / m; chi2_q <- qchisq(probs, df = ncol(Y))
      png(paths$fig_md_qq, width = 900, height = 700, res = 120)
      plot(chi2_q, sort(md2), pch = 20,
           main = bquote("ST mv — QQ " ~ D^2 ~ " vs " ~ chi^2[.(ncol(Y))]),
           xlab = bquote("Quantili " ~ chi^2[.(ncol(Y))]),
           ylab = expression("Quantili di " * D^2))
      abline(0, 1, lwd = 2, col = "gray40"); dev.off()
      add_time("plot_MD_QQ_save", 0)
    } else warning(sprintf("Mahalanobis disponibili = %d (<5): salto QQ-plot.", m))
  } else warning("solve(Omega_PD) fallita: salto MD-QQ.")
}

# PIT multivariato (Probability Integral Transform) opzionale per testare l’adattamento del modello.
if (DO_PIT_MV) {
  Omega_PD <- make_PD(Omega_hat, 1e-8)
  u <- vapply(seq_len(nrow(Y)), function(i) {
    val <- try(sn::pmst(x = as.numeric(Y[i, ]), xi = xi_hat, Omega = Omega_PD, alpha = alpha_hat, nu = nu_hat), silent = TRUE)
    if (inherits(val, "try-error") || !is.finite(val)) NA_real_ else as.numeric(val)
  }, numeric(1))
  u_ok <- u[is.finite(u)]
  if (length(u_ok) >= 10) {
    png(paths$fig_pit, width = 900, height = 700, res = 120)
    hist(u_ok, breaks = 30, freq = TRUE, main = "ST mv — PIT (R)", xlab = "u = F_MST(y; theta hat)")
    abline(h = length(u_ok)/30, lwd = 2, col = "gray40"); dev.off()
    add_time("plot_PIT_save", 0)
  } else warning(sprintf("PIT validi = %d (<10): salto istogramma.", length(u_ok)))
}

# (D) Riflesso
# Riflette i dati Y (Y_neg = -Y) e ripete la stima per verificare simmetria parametrica:
# ci si attende xi_ref ≈ -xi e alpha_ref ≈ -alpha, Omega simile e stesso nu.
if (DO_REFLECTION) {
  Y_neg <- -as.matrix(Y)
  fit_neg <- sn::selm(form, data = as.data.frame(Y_neg), family = "ST")
  cf2 <- sn::coef(fit_neg, "DP", vector = FALSE)
  xi_hat_ref    <- as.numeric(cf2$beta)
  Omega_hat_ref <- as.matrix(cf2$Omega)
  alpha_hat_ref <- as.numeric(cf2$alpha)
  nu_hat_ref    <- as.numeric(cf2$nu)
  refl_out <- data.frame(
    metric = c(
      paste0("xi_", seq_len(ncol(Y)), " + xi_ref_", seq_len(ncol(Y))),
      paste0("alpha_", seq_len(ncol(Y)), " + alpha_ref_", seq_len(ncol(Y))),
      "Omega - Omega_ref (Frobenius)",
      "nu - nu_ref"
    ),
    value  = c(
      xi_hat + xi_hat_ref,
      alpha_hat + alpha_hat_ref,
      sqrt(sum((Omega_hat - Omega_hat_ref)^2)),
      nu_hat - nu_hat_ref
    )
  )
  write.csv(refl_out, paths$tab_reflect, row.names = FALSE)
}

# (E) Salvataggi 
fit_out <- data.frame(
  engine = "R",
  p      = ncol(Y),
  n      = nrow(Y),
  loglik = llhat,
  nu_hat = nu_hat
)
for (j in seq_len(ncol(Y))) fit_out[[paste0("xi_hat_", j)]]    <- xi_hat[j]
for (j in seq_len(ncol(Y))) fit_out[[paste0("alpha_hat_", j)]] <- alpha_hat[j]
fit_out$omega_hat_diag <- paste(fmt(diag(Omega_hat)), collapse = ";")
write.csv(fit_out, paths$tab_fit, row.names = FALSE)

# Salva Omega in formato long (i, j, valore).
Om_long <- do.call(rbind, lapply(seq_len(ncol(Y)), function(i)
  data.frame(i = i, j = seq_len(ncol(Y)), value = Omega_hat[i, ])))
write.csv(Om_long, paths$tab_fit_Om, row.names = FALSE)

# Log dei tempi di esecuzione.
write.csv(timings, paths$log_times, row.names = FALSE)

# Riepilogo console con percorsi dei file generati e diagnostiche eseguite.
cat("OK R — Dataset:", paths$data, "\n")
cat("OK R — Stime:",   paths$tab_fit, " (loglik =", fmt(llhat), ", nu_hat =", fmt(nu_hat), ")\n")
if (DO_PAIRS) cat("OK R — Pairs:",  paths$fig_pairs, "\n")
if (DO_MDQQ)  cat("OK R — MD QQ:",  paths$fig_md_qq, "\n")
if (DO_PIT_MV)cat("OK R — PIT mv:", paths$fig_pit, "\n")
if (DO_REFLECTION) cat("OK R — Rifless.:", paths$tab_reflect, "\n")
cat("OK R — Omega:",   paths$tab_fit_Om, "\n")
cat("OK R — Tempi:",   paths$log_times, "\n")
