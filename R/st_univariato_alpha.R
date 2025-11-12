# GRID alpha — ST univariata (generazione per alpha, stima, diagnostiche UNIF + inversione)
# Output:
#   - common/tables/st_uni_A_grid_R.csv
#   - common/logs/st_uni_A_grid_times_R.csv

library(sn)

# Griglia di asimmetria alpha su cui simulare e stimare; n osservazioni per ciascun alpha.
# Parametri di generazione: xi (posizione), omega (scala), nu_true (gradi di libertà).
alpha_grid <- c(-4.0, -1.2, -0.2, 0.0, 0.2, 1.2, 4.0)
n       <- 2000
xi      <- 0
omega   <- 1
nu_true <- 6   # momenti fino al 4° ben definiti

# Soglia semantica per nu: garantisce almeno varianza esistente (≥2.05).
# Per pretendere anche kurtosi (4° momento), imposta ST_SEM_NU_MIN=4.05 nell'ambiente.
SEM_NU_MIN <- suppressWarnings(as.numeric(Sys.getenv("ST_SEM_NU_MIN", unset = "2.05")))
if (!is.finite(SEM_NU_MIN) || SEM_NU_MIN <= 0) SEM_NU_MIN <- 2.05

# Bound inferiori in scala log per omega e nu usati nei vincoli di ottimizzazione.
LOG_OM_MIN <- log(1e-8)
LOG_NU_MIN <- log(SEM_NU_MIN)

# Controlli per L-BFGS-B: iterazioni massime, factr (precisione), pgtol (gradiente).
LBFGSB_CTRL <- list(maxit = 5000, factr = 1e7, pgtol = 1e-8)

# Directory di output: dati, figure, tabelle, log. Le crea se non esistono.
dirs <- list(
  data = "common/data",
  fig  = "common/fig",
  tab  = "common/tables",
  log  = "common/logs"
)
for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)

# funzioni di timing
tic  <- function() proc.time()[["elapsed"]]
tock <- function(t0) proc.time()[["elapsed"]] - t0

# NLL per ST con penalizzazione dura fuori dai vincoli.
#     Parametrizzazione: (xi, log(omega), alpha, log(nu)).
#     - Controlli di finitezza.
#     - Guard-rails: omega>0, nu≥soglia semantica.
#     - try su densità: se fallisce o produce non-finiti/non-positivi, restituisce 1e50.
#     Ritorna la somma negativa dei log-densità (valore da minimizzare).
# parametri: (xi, log(omega), alpha, log(nu))
nll_st_par <- function(par, y) {
  xi  <- par[1]; lom <- par[2]; al <- par[3]; lnu <- par[4]
  if (!is.finite(lom) || !is.finite(lnu)) return(1e50)
  om <- exp(lom); nv <- exp(lnu)
  if (!is.finite(om) || om <= 0)          return(1e50)
  if (!is.finite(nv) || nv <  SEM_NU_MIN) return(1e50)

  dens <- suppressWarnings(try(dst(y, xi = xi, omega = om, alpha = al, nu = nv), silent = TRUE))
  if (inherits(dens, "try-error") || any(!is.finite(dens)) || any(dens <= 0)) return(1e50)
  -sum(log(dens))
}

# Loop su alpha
# Collezioni per accumulare risultati di stima e tempi per ogni alpha.
all_fit   <- list()
all_times <- list()

for (a in alpha_grid) {
  # (A) Dati
  # Seed deterministico che dipende da alpha (così ogni alpha produce sempre lo stesso dataset).
  set.seed(20250303 + as.integer(round((a + 10)*1000)))
  t0 <- tic()
  y  <- rst(n, xi = xi, omega = omega, alpha = a, nu = nu_true)
  t_gen <- tock(t0)

  # Salva il dataset simulato per questo alpha in un file separato (nome con alpha formattato).
  fdata <- sprintf("%s/st_uni_A_alpha_%s.csv", dirs$data, gsub("\\.", "p", sprintf("%.1f", a)))
  write.csv(data.frame(y = y), fdata, row.names = FALSE)

  # (B) Stima ML (xi, log(omega), alpha, log(nu)) con vincoli
  # Punto iniziale per alpha: 0 se a=0, altrimenti segno(a) e ampiezza limitata tra 0.1 e 2.
  #     Start per log(nu): almeno soglia+0.5 o nu_true, per stabilità numerica.
  start_alpha <- if (a == 0) 0 else sign(a) * max(0.1, min(2, abs(a)))
  start <- c(median(y), log(IQR(y)/1.349), start_alpha, log(max(SEM_NU_MIN + 0.5, nu_true)))

  # Ottimizzazione L-BFGS-B della NLL con vincoli su log(omega) e log(nu).
  t0 <- tic()
  fit <- optim(
    par     = start,
    fn      = function(p) nll_st_par(p, y),
    method  = "L-BFGS-B",
    lower   = c(-Inf, LOG_OM_MIN, -Inf, LOG_NU_MIN),
    control = LBFGSB_CTRL
  )
  t_fit <- tock(t0)

  # Estrae stime sullo spazio originale per omega e nu; log-likelihood massimo.
  parhat <- c(
    xi    = fit$par[1],
    omega = exp(fit$par[2]),
    alpha = fit$par[3],
    nu    = exp(fit$par[4])
  )
  llhat  <- -fit$value

  # (C) Diagnostiche sintetiche: inversione CDF↔PPF (core + code) e UNIF (KS)
  # Griglia di probabilità: “core” (0.001–0.999) + “extreme tails” (1e-6,1e-5,1e-4, ...).
  p_core <- seq(0.001, 0.999, by = 0.001)
  p_ext  <- c(1e-6, 1e-5, 1e-4, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6)
  p_grid <- c(p_ext[1:3], p_core, p_ext[4:6])

  # Verifica di inversione: qst(p) seguito da pst(q). Calcola max errore assoluto
  # separatamente per core ed estremi.
  q_inv  <- suppressWarnings(qst(p_grid, xi = parhat["xi"], omega = parhat["omega"],
                                 alpha = parhat["alpha"], nu = parhat["nu"]))
  p_back <- suppressWarnings(pst(q_inv,  xi = parhat["xi"], omega = parhat["omega"],
                                 alpha = parhat["alpha"], nu = parhat["nu"]))
  inv_core <- max(abs(p_back[4:(length(p_back)-3)] - p_core))
  inv_ext  <- max(abs(p_back[c(1:3, (length(p_back)-2):length(p_back))] -
                       c(p_ext[1:3], p_ext[4:6])))

  # Trasformazione PIT (u = F(y; theta hat)); KS contro Uniforme(0,1) come check di adeguatezza.
  u    <- suppressWarnings(pst(y, xi = parhat["xi"], omega = parhat["omega"],
                               alpha = parhat["alpha"], nu = parhat["nu"]))
  ks_p <- suppressWarnings(ks.test(u, "punif")$p.value)  # UNIF: KS su U(0,1)

  # (D) Accumulo risultati
  # Raccoglie misure chiave per questo alpha: stime, loglik, errori d’inversione,
  # p-value KS, n e soglia nu usata; e logga i tempi di generazione e stima.
  all_fit[[length(all_fit)+1]] <- data.frame(
    engine="R", alpha_true=a,
    xi_hat=parhat["xi"], omega_hat=parhat["omega"],
    alpha_hat=parhat["alpha"], nu_hat=parhat["nu"],
    loglik=llhat, n=n,
    inv_core_maxerr=inv_core, inv_ext_maxerr=inv_ext, ks_pvalue=ks_p,
    nu_sem_threshold = SEM_NU_MIN
  )
  all_times[[length(all_times)+1]] <- data.frame(
    alpha_true=a, step="data_rst_generate", seconds=t_gen   # allineato al merge
  )
  all_times[[length(all_times)+1]] <- data.frame(
    alpha_true=a, step="fit_ML_LBFGSB", seconds=t_fit       # allineato al merge
  )
}

# Scrittura output aggregati 
fit_out   <- do.call(rbind, all_fit)
times_out <- do.call(rbind, all_times)
write.csv(fit_out,   file.path(dirs$tab, "st_uni_A_grid_R.csv"),           row.names=FALSE)
write.csv(times_out, file.path(dirs$log, "st_uni_A_grid_times_R.csv"),     row.names=FALSE)

# Messaggio di riepilogo a console con percorso dei file prodotti.
cat("✓ ST univariata — GRID alpha: scritto\n",
    "  • Tabelle :", file.path(dirs$tab, "st_uni_A_grid_R.csv"), "\n",
    "  • Tempi   :", file.path(dirs$log, "st_uni_A_grid_times_R.csv"), "\n", sep = "")
