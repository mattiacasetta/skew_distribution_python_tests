# SN multivariata
# - Se esiste common/data/sn_mv.csv: legge i dati (colonne x1..xp)
# - Altrimenti: simula da SN_p con Omega diagonale via sn::rmsn
# - Stima ML con parametrizzazione DP (xi, Omega=diag(exp(eta)), alpha)
#   * eta = log(varianze marginali) per garantire Omega ≻ 0
#   * ottimizzazione con optim(method="BFGS")
# - Esporta:
#   * common/tables/sn_mv_fit_R.csv            (stime, loglik, diag(Omega))
#   * common/tables/sn_mv_fit_Omega_R.csv      (Omega in formato long i,j,value)
#   * common/fig/sn_mv_pairs_R.png             (pairs plot)
#   * common/logs/sn_mv_times_R.csv            (tempi per categoria)
# - Naming passi (per il merge LEAN):
#   data_rmsn_generate | io_write_csv_data | fit_MSN_MLE | plot_pairs_save

library(sn)


set.seed(20250303)


# Funzioni di timing e formattazione
.tic <- function() proc.time()[["elapsed"]]
.tock <- function(t0) proc.time()[["elapsed"]] - t0
.add_time <- local({
  timings <- data.frame(step = character(), seconds = numeric(),
                        stringsAsFactors = FALSE)
  function(step, sec) {
    timings <<- rbind(timings, data.frame(step = step, seconds = sec))
  }
})

# percorsi output
paths <- list(
  data_in      = "common/data/sn_mv.csv",
  fig_dir      = "common/fig",
  fig_pairs    = "common/fig/sn_mv_pairs_R.png",
  tab_dir      = "common/tables",
  fit_csv      = "common/tables/sn_mv_fit_R.csv",
  fit_Omega    = "common/tables/sn_mv_fit_Omega_R.csv",
  log_dir      = "common/logs",
  log_times    = "common/logs/sn_mv_times_R.csv"
)

# crea cartelle
lapply(unique(dirname(unlist(paths))), function(d) dir.create(d, recursive = TRUE, showWarnings = FALSE))

set.seed(20250303)

# Carica o simula dati
load_or_simulate <- function() {
  if (file.exists(paths$data_in)) {
    df <- read.csv(paths$data_in)
    X  <- as.matrix(df)
    colnames(X) <- if (is.null(colnames(X))) paste0("x", seq_len(ncol(X))) else colnames(X)
    return(list(X = X, from_file = TRUE))
  }
  # Simulazione di default (p=3) con Omega diagonale
  p  <- 3L
  n  <- 2000L
  xi <- c(0, 1, -1)[seq_len(p)]
  alpha <- c(2, -1, 1.5)[seq_len(p)]
  Omega <- diag(rep(1, p))  # varianze = 1 (diagonale)

  t0 <- .tic();
  Y  <- sn::rmsn(n = n, xi = xi, Omega = Omega, alpha = alpha)
  .add_time("data_rmsn_generate", .tock(t0))

  colnames(Y) <- paste0("x", seq_len(ncol(Y)))
  t0 <- .tic();
  write.csv(as.data.frame(Y), paths$data_in, row.names = FALSE)
  .add_time("io_write_csv_data", .tock(t0))

  list(X = Y, from_file = FALSE)
}

ld <- load_or_simulate()
Y  <- ld$X
n  <- nrow(Y)
p  <- ncol(Y)

# NLL (DP con Omega diagonale)
# Parametri in un unico vettore: theta = (xi_1..xi_p, eta_1..eta_p, alpha_1..alpha_p)
# con Omega = diag(exp(eta)) —> var_i = exp(eta_i) > 0.

.unpack_theta <- function(theta) {
  stopifnot(length(theta) == 3L * p)
  xi    <- theta[1:p]
  eta   <- theta[(p + 1):(2 * p)]
  alpha <- theta[(2 * p + 1):(3 * p)]
  list(xi = xi, eta = eta, alpha = alpha)
}

nll_diag <- function(theta, Y) {
  pa <- .unpack_theta(theta)
  Omega <- diag(exp(pa$eta), nrow = p, ncol = p)  # varianze marginali
  # Somma di log-densità (log=TRUE) → NLL
  -sum(sn::dmsn(Y, xi = pa$xi, Omega = Omega, alpha = pa$alpha, log = TRUE))
}

# Inizializzazione robusta
# Evitare alpha ≡ 0 (punto non regolare). 
# Usare segno della skewness campionaria come guida.
col_skew <- function(y) {
  # skewness classica: E[((Y-mean)/sd)^3]
  yc <- sweep(y, 2L, colMeans(y), FUN = "-")
  s  <- sqrt(colMeans(yc^2))
  z  <- sweep(yc, 2L, s, FUN = "/")
  colMeans(z^3)
}

xi0   <- colMeans(Y)
eta0  <- log(pmax(1e-6, diag(var(Y))))  # log-varianze iniziali
sgn   <- sign(col_skew(Y)); sgn[sgn == 0] <- 1
alpha0 <- 0.5 * sgn  # lontano da 0, ma moderato

th0 <- c(xi0, eta0, alpha0)

# Ottimizzazione ML
cat(sprintf("[INFO] Avvio ottimizzazione (p=%d, n=%d) ...\n", p, n))

ctrl <- list(reltol = 1e-10, maxit = 2000)

t0 <- .tic();
fit <- optim(par = th0, fn = nll_diag, Y = Y, method = "BFGS", control = ctrl)
.add_time("fit_MSN_MLE", .tock(t0))

stopifnot(is.list(fit), is.finite(fit$value))

par_hat <- .unpack_theta(fit$par)
Omega_hat <- diag(exp(par_hat$eta), nrow = p, ncol = p)
loglik    <- -fit$value
conv      <- fit$convergence
fevals    <- if (!is.null(fit$counts[["function"]])) as.integer(fit$counts[["function"]]) else NA_integer_

cat(sprintf("[OK] Ottimizzazione terminata (conv=%d, loglik=%.6f)\n", conv, loglik))

# Pairs plot
png(paths$fig_pairs, width = 1100, height = 900, res = 120)
par(mfrow = c(1,1))
pairs(Y, pch = 20, main = sprintf("Pairs plot SN (R) — p=%d, n=%d", p, n))
dev.off()
.add_time("plot_pairs_save", 0)  # plotting time marginale; omesso per semplicità

# Salvataggi risultati
# (1) Stime sintetiche in un'unica riga (compatibile con merge LEAN)

omega_diag_vals <- as.numeric(diag(Omega_hat))  # varianze marginali
omega_diag_str  <- paste(format(omega_diag_vals, digits = 10), collapse = ";")

out <- data.frame(
  engine = "R",
  p = p,
  loglik = loglik,
  fevals = fevals,
  convergence = conv,
  omega_hat_diag = omega_diag_str,
  stringsAsFactors = FALSE
)
for (j in seq_len(p)) out[[sprintf("xi_hat_%d", j)]]    <- par_hat$xi[j]
for (j in seq_len(p)) out[[sprintf("alpha_hat_%d", j)]] <- par_hat$alpha[j]

write.csv(out, paths$fit_csv, row.names = FALSE)

# (2) Omega in formato long (i,j,value)
long_Omega <- do.call(rbind, lapply(seq_len(p), function(i)
  do.call(rbind, lapply(seq_len(p), function(j)
    data.frame(i = i, j = j, value = Omega_hat[i, j])
  ))
))
write.csv(long_Omega, paths$fit_Omega, row.names = FALSE)

# (3) Tempi
# Nota: se i dati sono stati letti da file, non registriamo data_rmsn_generate/io_write
write.csv(get("timings", envir = environment(.add_time)), paths$log_times, row.names = FALSE)

# Messaggi finali
cat("\n=== Output ===\n")
cat("Fit:       ", paths$fit_csv, "\n")
cat("Omega (L): ", paths$fit_Omega, "\n")
cat("Figure:    ", paths$fig_pairs, "\n")
cat("Logs:      ", paths$log_times, "\n")
