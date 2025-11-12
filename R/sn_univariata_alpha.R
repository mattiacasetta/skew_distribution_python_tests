# R script per simulazioni e stime ML per la distribuzione SN univariata
# con diversi valori di alpha (intensità di skew).
# Output:
#   - common/data/sn_uni_alpha_<tag>.csv            (dataset per ogni alpha)
#   - common/tables/sn_uni_grid_R.csv               (tabella riepilogo st
#   - common/logs/sn_uni_grid_times_R.csv           (tempi per step)

library(sn)   

alpha_grid <- c(-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0)
n      <- 2000
xi     <- 0
omega  <- 1

# Directory
dirs <- list(
  data = "common/data",
  fig  = "common/fig",
  tab  = "common/tables",
  log  = "common/logs"
)
for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)

# Utilità timing
tic  <- function() proc.time()[["elapsed"]]
tock <- function(t0) proc.time()[["elapsed"]] - t0

all_fit   <- list()
all_times <- list()

for (a in alpha_grid) {
  # (A) Dati condivisi
  set.seed(20250303 + as.integer(round((a + 10)*1000)))
  t0 <- tic()
  y  <- rsn(n, xi = xi, omega = omega, alpha = a)
  t_gen <- tock(t0)
  fdata <- sprintf("%s/sn_uni_alpha_%s.csv", dirs$data, gsub("\\.", "p", sprintf("%.1f", a)))
  write.csv(data.frame(y=y), fdata, row.names = FALSE)

  # (B) Stima ML trasparente
  loglik_sn <- function(par, y) {
    xi <- par[1]; om <- abs(par[2]); al <- par[3]
    sum(log(dsn(y, xi=xi, omega=om, alpha=al)))
  }
  start <- c(median(y), IQR(y)/1.349, sign(a)*max(0.1, min(2, abs(a))))  # start non-degenere
  t0 <- tic()
  fit <- optim(start, fn=function(p) -loglik_sn(p, y),
               method="BFGS", control=list(reltol=1e-10, maxit=5000))
  t_fit <- tock(t0)
  parhat <- c(xi=fit$par[1], omega=abs(fit$par[2]), alpha=fit$par[3])
  llhat  <- -fit$value

  # (C) Diagnostiche sintetiche: inversione CDF↔PPF (core + code)
  p_core <- seq(0.001, 0.999, by=0.001)
  p_ext  <- c(1e-6, 1e-5, 1e-4, 1-1e-4, 1-1e-5, 1-1e-6)
  p_grid <- c(p_ext[1:3], p_core, p_ext[4:6])

  q_inv  <- qsn(p_grid, xi=parhat["xi"], omega=parhat["omega"], alpha=parhat["alpha"])
  p_back <- psn(q_inv, xi=parhat["xi"], omega=parhat["omega"], alpha=parhat["alpha"])
  inv_core <- max(abs(p_back[4:(length(p_back)-3)] - p_core))
  inv_ext  <- max(abs(p_back[c(1:3, (length(p_back)-2):length(p_back))] -
                       c(p_ext[1:3], p_ext[4:6])))

  # (D) UNIF uniformità (KS)
  u    <- psn(y, xi=parhat["xi"], omega=parhat["omega"], alpha=parhat["alpha"])
  ks_p <- suppressWarnings(ks.test(u, "punif")$p.value)

  # (E) Salvataggi
  all_fit[[length(all_fit)+1]] <- data.frame(
    engine="R", alpha_true=a, xi_hat=parhat["xi"], omega_hat=parhat["omega"], alpha_hat=parhat["alpha"],
    loglik=llhat, n=n, inv_core_maxerr=inv_core, inv_ext_maxerr=inv_ext, ks_pvalue=ks_p
  )
  all_times[[length(all_times)+1]] <- data.frame(
    alpha_true=a, step="rsn_generate", seconds=t_gen
  )
  all_times[[length(all_times)+1]] <- data.frame(
    alpha_true=a, step="optim_ML", seconds=t_fit
  )
}

fit_out   <- do.call(rbind, all_fit)
times_out <- do.call(rbind, all_times)
write.csv(fit_out,   file.path(dirs$tab, "sn_uni_grid_R.csv"),  row.names=FALSE)
write.csv(times_out, file.path(dirs$log, "sn_uni_grid_times_R.csv"), row.names=FALSE)
cat("R grid: scritto\n")