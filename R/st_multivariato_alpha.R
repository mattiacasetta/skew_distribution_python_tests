# UNIF — ST multivariata: stress test su alpha (griglia su uno scalare s che moltiplica una direzione fissa)
# Fit via sn::selm(..., family="ST"), senza diagnostiche pesanti (solo stime + tempi).
# Output:
#   - common/data/st_mv_A_alpha_<tag>.csv            (dataset per ogni s)
#   - common/tables/st_mv_A_grid_R.csv               (tabella riepilogo stime)
#   - common/logs/st_mv_A_grid_times_R.csv           (tempi per step)

library(sn)

alpha_grid <- c(-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0)  # intensità (scalari)
n <- 2000; p <- 3

# Parametri fissi
xi_true    <- c(0, 1, -1)
Omega_true <- matrix(c(
  1.0, 0.4, 0.2,
  0.4, 1.0, 0.3,
  0.2, 0.3, 1.0
), nrow = p, byrow = TRUE)
alpha_dir_base <- c(2.0, -1.0, 1.5)   # direzione di skew di riferimento
alpha_dir <- alpha_dir_base / sqrt(sum(alpha_dir_base^2))  # normalizzata (‖v‖=1)
nu_true <- 8

# Directory
dirs <- list(
  data = "common/data",
  tab  = "common/tables",
  log  = "common/logs"
)
for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)

# Funzioni di timing e formattazione
tic  <- function() proc.time()[["elapsed"]]
tock <- function(t0) proc.time()[["elapsed"]] - t0
fmt  <- function(x, k = 6) formatC(x, digits = k, format = "g")

# Storage risultati
all_fit   <- list()
all_times <- list()

# Loop sulla griglia di alpha
for (s in alpha_grid) {
  set.seed(20250303 + as.integer(round((s + 10)*1000)))
  alpha_true <- as.numeric(s * alpha_dir)  # alpha (vettore) = s · v

  # (A) Generazione dati
  t0 <- tic()
  Y  <- sn::rmst(n, xi = xi_true, Omega = Omega_true, alpha = alpha_true, nu = nu_true)
  t_gen <- tock(t0)

  colnames(Y) <- sprintf("y%02d", seq_len(ncol(Y)))
  tag <- gsub("\\.", "p", sprintf("%.1f", s))
  fdata <- file.path(dirs$data, sprintf("st_mv_A_alpha_%s.csv", tag))
  write.csv(as.data.frame(Y), fdata, row.names = FALSE)

  # (B) Fit via selm (family = "ST")
  Ydf  <- as.data.frame(Y)
  form <- as.formula(paste0("cbind(", paste(names(Ydf), collapse = ","), ") ~ 1"))
  t0   <- tic()
  fit  <- sn::selm(form, data = Ydf, family = "ST")
  t_fit <- tock(t0)

  # Estrai parametri DP (xi, Omega, alpha, nu)
  cf <- sn::coef(fit, "DP", vector = FALSE)
  xi_hat    <- as.numeric(cf$beta)
  Omega_hat <- as.matrix(cf$Omega)
  alpha_hat <- as.numeric(cf$alpha)
  nu_hat    <- as.numeric(cf$nu)

  # Log-likelihood
  llhat <- if (!is.null(fit@logL)) as.numeric(fit@logL) else {
    dens <- sn::dmst(as.matrix(Y), xi = xi_hat, Omega = Omega_hat, alpha = alpha_hat, nu = nu_hat)
    sum(log(dens))
  }

  # (C) Riepilogo per riga
  row <- data.frame(
    engine          = "R",
    alpha_scale     = s,
    p               = ncol(Y),
    n               = nrow(Y),
    loglik          = llhat,
    nu_hat          = nu_hat,
    omega_hat_diag  = paste(fmt(diag(Omega_hat)), collapse = ";"),
    stringsAsFactors = FALSE
  )
  for (j in seq_len(ncol(Y))) row[[paste0("xi_hat_", j)]]    <- xi_hat[j]
  for (j in seq_len(ncol(Y))) row[[paste0("alpha_hat_", j)]] <- alpha_hat[j]
  for (j in seq_len(ncol(Y))) row[[paste0("alpha_true_", j)]] <- alpha_true[j]

  all_fit[[length(all_fit) + 1]] <- row

  # Tempi
  all_times[[length(all_times) + 1]] <- data.frame(alpha_scale = s, step = "rmst_generate", seconds = t_gen)
  all_times[[length(all_times) + 1]] <- data.frame(alpha_scale = s, step = "selm_MST_MLE", seconds = t_fit)

  cat(sprintf("✓ s=%s → file: %s | loglik=%.6g | nu=%.6g\n", tag, fdata, llhat, nu_hat))
}

#  Scritture finali
fit_out   <- do.call(rbind, all_fit)
times_out <- do.call(rbind, all_times)

write.csv(fit_out,   file.path(dirs$tab, "st_mv_A_grid_R.csv"),          row.names = FALSE)
write.csv(times_out, file.path(dirs$log, "st_mv_A_grid_times_R.csv"),     row.names = FALSE)

cat("UNIF — ST multivariata — grid α: scritto\n")
