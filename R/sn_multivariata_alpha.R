# SN multivariata — Stress test su alpha
# - Genera dati condivisi per ciascun alpha su griglia
# - Stima ML con vincolo Omega diagonale via optim (BFGS)
# - Salva risultati (fit + tempi) in common/tables e common/logs

library(sn)

# Configurazione
alpha_grid <- c(-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0)  # griglia alpha (slant)
n <- 2000                                            # numerosità per scenario
p <- 3                                               # dimensione
xi_true    <- c(0, 1, -1)                            # location di verità-terreno
Omega_true <- diag(rep(1, p))                        # Omega vera: diagonale unità

# Validazione input (matrici/vettori finiti)
validate_finite_matrix <- function(x, name = "object") {
  if (is.null(x)) stop(sprintf("%s is NULL", name))
  if (!is.numeric(x) && !is.matrix(x)) stop(sprintf("%s deve essere numerico/matrice", name))
  if (is.vector(x)) x <- as.matrix(x)
  if (!is.matrix(x)) stop(sprintf("%s non è una matrice valida", name))
  if (any(!is.finite(x))) stop(sprintf("%s contiene NA/Inf: %s", name, paste(head(as.character(x), 6), collapse = ",")))
  TRUE
}

# Directory
dirs <- list(
  data = "common/data",
  tab  = "common/tables",
  log  = "common/logs"
)
for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)

# Utility timing
tic  <- function() proc.time()[["elapsed"]]
tock <- function(t0) proc.time()[["elapsed"]] - t0

# Funzione per generare tag stringa per alpha
alphatag <- function(a) {
  s <- sprintf("%.1f", a)
  s <- gsub("-", "m", s, fixed = TRUE)
  gsub("\\.", "p", s)
}

# Stima ML (Omega diag)
# Parametri da ottimizzare: theta = (xi_1..xi_p, eta_1..eta_p, alpha_1..alpha_p)
# con Omega = diag(exp(eta)), quindi positività assicurata via exp.
nll_msn_diag <- function(theta, X) {
  d <- ncol(X)
  xi    <- theta[1:d]
  eta   <- theta[(d+1):(2*d)]
  alpha <- theta[(2*d+1):(3*d)]
  Omega <- diag(exp(eta), d, d)

  # controllo validità Omega 
  if (any(!is.finite(Omega))) {
    return(.Machine$double.xmax * 1e-5)
  }

  # log-likelihood negativa: -sum log f_SN
  val <- tryCatch({
    # dmsn may internally check symmetry etc.; guard against errors
    llvec <- sn::dmsn(X, xi = xi, Omega = Omega, alpha = alpha, log = TRUE)
    if (any(!is.finite(llvec))) {
      .Machine$double.xmax * 1e-5
    } else {
      -sum(llvec)
    }
  }, error = function(e) {
    # return large penalty, not NA/stop
    .Machine$double.xmax * 1e-5
  })

  if (!is.finite(val)) val <- .Machine$double.xmax * 1e-5
  val
}

# Costruzione start robusti per (xi, eta, alpha) dato X e uno "scale" per alpha
make_start <- function(X, a_scalar) {
  d <- ncol(X)
  # xi start: mediane colonna
  xi0 <- apply(X, 2, median)
  # eta start: log(var robusta): (IQR/1.349)^2, con floor minimo
  sc  <- (apply(X, 2, IQR) / 1.349)
  # proteggo eventuali NA/non-finite prodotte da IQR/divisioni
  var0 <- sc^2
  var0[!is.finite(var0)] <- 1e-3
  var0 <- pmax(var0, 1e-3)
  eta0 <- log(var0)
  # alpha start: stesso segno di a, magnitudo ragionevole (0.1..2)
  a0 <- sign(a_scalar) * max(0.1, min(2, abs(a_scalar)))
  alpha0 <- rep(a0, d)
  c(xi0, eta0, alpha0)
}

# Loop su griglia alpha
all_fit   <- list()
all_times <- list()

for (a in alpha_grid) {
  # (A) Dati condivisi
  set.seed(20250303 + as.integer(round((a + 10) * 1000)))
  t0 <- tic()
  # Validazione input prima di rmsn
  tryCatch({
    validate_finite_matrix(as.matrix(xi_true), "xi_true")
    # Omega deve essere matrice 
    if (!is.matrix(Omega_true)) Omega_true <- as.matrix(Omega_true)
    validate_finite_matrix(Omega_true, "Omega_true")
    validate_finite_matrix(matrix(rep(a, p), nrow=p, ncol=1), "alpha (candidate)")
  }, error = function(e) {
    stop(sprintf("Input validation failed before rmsn: %s", e$message))
  })

  X  <- tryCatch({
    sn::rmsn(n = n, xi = xi_true, Omega = Omega_true, alpha = rep(a, p))
  }, error = function(e) {
    stop(sprintf("sn::rmsn failed for alpha=%s: %s\nxi_true=%s\nOmega_true(diag)=%s",
                 format(a), e$message,
                 paste(capture.output(print(xi_true)), collapse = ","),
                 paste(diag(Omega_true), collapse = ",")))
  })
  t_gen <- tock(t0)

  # Salva dataset
  dfX <- as.data.frame(X)
  colnames(dfX) <- paste0("x", seq_len(p))
  fdata <- file.path(dirs$data, sprintf("sn_mv_alpha_a_%s.csv", alphatag(a)))
  write.csv(dfX, fdata, row.names = FALSE)

  # Stima ML (Omega diag)
  start <- make_start(X, a)
  t0 <- tic()
  fit <- optim(
    par = start,
    fn  = function(th) nll_msn_diag(th, X),
    method = "BFGS",
    control = list(reltol = 1e-10, maxit = 5000)
  )
  t_fit <- tock(t0)

  # Estrai parametri e logLik
  xi_hat    <- fit$par[1:p]
  eta_hat   <- fit$par[(p+1):(2*p)]
  alpha_hat <- fit$par[(2*p+1):(3*p)]
  omega_hat_diag <- exp(eta_hat) # varianze diagonali stimate
  loglik <- -fit$value

  # (C) Accumula output
  # Riga "larga" con colonne: engine, p, alpha_true, n, loglik,
  # xi_hat_1..p, omega_hat_diag_1..p, alpha_hat_1..p
  row <- c(
    list(engine = "R", p = p, alpha_true = a, n = n, loglik = loglik),
    setNames(as.list(xi_hat),    paste0("xi_hat_",    seq_len(p))),
    setNames(as.list(omega_hat_diag), paste0("omega_hat_diag_", seq_len(p))),
    setNames(as.list(alpha_hat), paste0("alpha_hat_", seq_len(p)))
  )
  all_fit[[length(all_fit) + 1]] <- as.data.frame(row, check.names = FALSE)

  # Tempi
  all_times[[length(all_times) + 1]] <- data.frame(
    alpha_true = a, step = "rmsn_generate",    seconds = t_gen
  )
  all_times[[length(all_times) + 1]] <- data.frame(
    alpha_true = a, step = "optim_ML_diag",    seconds = t_fit
  )
}

# Salvataggi finali
fit_out   <- do.call(rbind, all_fit)
times_out <- do.call(rbind, all_times)

write.csv(fit_out,   file.path(dirs$tab, "sn_mv_grid_R.csv"),           row.names = FALSE)
write.csv(times_out, file.path(dirs$log, "sn_mv_grid_times_R.csv"),     row.names = FALSE)

cat("✓ R mv grid: scritto\n",
    "  - common/tables/sn_mv_grid_R.csv\n",
    "  - common/logs/sn_mv_grid_times_R.csv\n", sep = "")
