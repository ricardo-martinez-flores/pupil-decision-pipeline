# =============================================================================
# pupil_analysis.R -- Statistical analysis of preprocessed pupil time series
# =============================================================================
#
# Generic, dataset-agnostic implementation of the statistical analysis
# described in:
#
#   Martinez-Flores, R., et al. Adapting pupillometry preprocessing and
#   analysis methods for older adults with mild cognitive impairment:
#   A decision-by-decision approach.
#
# Input: the long-format file written by pupil_pipeline.py
# (preprocessed_accepted_trials_long.csv; one row per sample of each accepted
# trial) and, optionally, its trial_ledger.csv for the confound check.
#
# The analysis answers three complementary questions about one two-level
# within-participant contrast (e.g. oddball Target vs. Distractor):
#
#   HOW does the effect unfold over time?   -> PART 1  GAMM (+ AR(1))
#   HOW LARGE is the effect on average?     -> PART 2  full time-series LMM
#   WHICH components of the response differ?-> PART 3  trial-level features
#
# Each model-based result is checked with resampling methods that do not rely
# on its distributional assumptions (the signal is autocorrelated and its
# residuals are not normal):
#
#   PART 0  Confound check: data quality (blink proxy) by condition
#   PART 1  GAMM without and with AR(1) residual correction; difference curve
#   PART 2  Full time-series LMM (single coefficient, Wald/Satterthwaite)
#     2b    Block permutation (trial labels shuffled within participant)
#     2c    Calibration of the block permutation under a simulated true null
#     2d    Cluster (participant) bootstrap, BCa 95% CI
#   PART 3  Trial-level features, LMM per feature + Holm correction
#     3b    Label permutation, cluster BCa bootstrap and max-t (Westfall & Young)
#   PART 4  Leave-one-subject-out (LOSO) sensitivity of the PART 2 estimate
#
# Permutation p-values use p = (b + 1) / (B + 1) (Phipson & Smyth, 2010), so a
# p-value is never exactly zero. Defaults: 5,000 permutations and 5,000
# bootstrap resamples.
#
# Usage:
#   Rscript pupil_analysis.R config_analysis.yaml
#
# Author: Ricardo Martinez-Flores
# License: see LICENSE in the repository root.
# =============================================================================

suppressMessages({
  library(yaml)
  library(dplyr)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(mgcv)
  library(ggplot2)
  library(parallel)
})

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_CONFIG <- list(
  input_csv    = "../python_preprocessing/outputs/preprocessed_accepted_trials_long.csv",
  trial_qc_csv = "../python_preprocessing/outputs/trial_ledger.csv",   # optional (PART 0)
  output_dir   = "outputs",
  csv_sep      = ",",
  csv_dec      = ".",
  columns = list(participant = "participant", trial = "trial", condition = "condition",
                 time = "time_ms", pupil = "pupil_bc"),
  qc_metrics   = c("pct_left", "pct_right", "coverage_ratio", "pct_interpolated"),
  conditions   = list(of_interest = "Target", reference = "Distractor"),
  gamm = list(k_condition = 20, k_participant = 10),
  lmm  = list(fixed_time_effect = FALSE, agreement_tolerance = 0.20),
  features = list(initial_window_ms = c(0, 500), late_window_ms = c(1000, Inf)),
  resampling = list(
    n_permutations = 5000,
    n_bootstrap = 5000,
    n_calibration_datasets = 100,
    n_calibration_permutations = 199,
    seed = 12345,
    n_cores = NULL,               # NULL = all available cores minus one
    timing_test = TRUE),
  calibration_simulation = list(
    n_participants = 15, n_trials_per_participant = 20, n_samples_per_trial = 40,
    prop_of_interest = 0.2, sample_step_ms = 30, noise_sd = 0.02, intercept_sd = 0.03,
    rho_if_gamm_skipped = 0.95),
  parts = list(confound = TRUE, gamm = TRUE, lmm = TRUE, lmm_permutation = TRUE,
               lmm_calibration = TRUE, lmm_bootstrap = TRUE, features = TRUE,
               features_resampling = TRUE, loso = TRUE),
  figure = list(font_family = "Times", width_in = 7, height_in = 4.3,
                accent_color = "#2166AC", ci_fill = "grey75")
)

deep_update <- function(base, new) {
  for (k in names(new)) {
    if (is.list(base[[k]]) && is.list(new[[k]]) && !is.null(names(new[[k]]))) {
      base[[k]] <- deep_update(base[[k]], new[[k]])
    } else {
      base[k] <- list(new[[k]])
    }
  }
  base
}

args <- commandArgs(trailingOnly = TRUE)
cfg <- DEFAULT_CONFIG
if (length(args) >= 1) cfg <- deep_update(cfg, yaml::read_yaml(args[1]))
rs <- cfg$resampling
if (is.null(rs$n_cores)) rs$n_cores <- max(1, parallel::detectCores() - 1)
set.seed(rs$seed)

OUT <- cfg$output_dir
DIRS <- c(p0 = "00_confound_check", p1 = "01_gamm", p2 = "02_time_series_lmm",
          p3 = "03_features", p4 = "04_loso")
for (d in DIRS) dir.create(file.path(OUT, d), recursive = TRUE, showWarnings = FALSE)
out_path <- function(part, file) file.path(OUT, DIRS[[part]], file)

REF  <- cfg$conditions$reference
INT  <- cfg$conditions$of_interest
TERM <- paste0("condition", INT)                 # name of the tested coefficient
LMER_CTRL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

SUMMARY <- character(0)                          # lines for analysis_summary.md
add_summary <- function(...) SUMMARY <<- c(SUMMARY, unlist(list(...)))  # one argument = one line

header <- function(txt) cat("\n", strrep("=", 78), "\n", txt, "\n", strrep("=", 78), "\n\n", sep = "")

# =============================================================================
# SHARED UTILITIES
# =============================================================================

# Permutation p-value (Phipson & Smyth, 2010): never exactly zero.
perm_p <- function(null_abs, obs) {
  v <- null_abs[!is.na(null_abs)]
  if (length(v) == 0 || is.na(obs)) return(NA_real_)
  (sum(v >= abs(obs)) + 1) / (length(v) + 1)
}

# Bias-corrected and accelerated (BCa) percentile interval.
# boot: bootstrap estimates; obs: full-sample estimate; jack: leave-one-
# participant-out estimates (acceleration constant).
bca_ci <- function(boot, obs, jack, level = 0.95) {
  boot <- boot[!is.na(boot)]; jack <- jack[!is.na(jack)]
  if (length(boot) < 10 || is.na(obs)) return(c(lo = NA_real_, hi = NA_real_, z0 = NA_real_, a = NA_real_))
  z0 <- qnorm(max(1e-6, min(1 - 1e-6, mean(boot < obs))))
  a <- 0
  if (length(jack) >= 3) {
    jm <- mean(jack)
    den <- 6 * (sum((jm - jack)^2))^(3 / 2)
    if (abs(den) > 1e-10) a <- sum((jm - jack)^3) / den
  }
  z <- qnorm(c((1 - level) / 2, 1 - (1 - level) / 2))
  alpha <- pnorm(z0 + (z0 + z) / (1 - a * (z0 + z)))
  alpha <- pmax(0.001, pmin(0.999, alpha))
  q <- quantile(boot, alpha, names = FALSE)
  c(lo = q[1], hi = q[2], z0 = z0, a = a)
}

# Run FUN(i) n times on the cluster, in batches, printing progress.
run_parallel <- function(cl, n, FUN, label) {
  if (n <= 0) return(list())
  batch <- max(rs$n_cores * 5, 1)
  out <- vector("list", n); done <- 0; t0 <- Sys.time()
  while (done < n) {
    idx <- (done + 1):min(n, done + batch)
    out[idx] <- parLapply(cl, idx, FUN)
    done <- max(idx)
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    cat(sprintf("  %s: %d / %d  (%.1f min elapsed, ~%.1f min remaining)\n",
                label, done, n, el, el / done * (n - done)))
  }
  out
}

# Optional timing test before a long resampling loop.
timing_test <- function(cl, FUN, n_target, label) {
  if (!isTRUE(rs$timing_test)) return(invisible(NULL))
  n_test <- min(rs$n_cores, n_target)
  t0 <- Sys.time(); invisible(parLapply(cl, seq_len(n_test), FUN))
  sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  [timing] %s: %.1f s for %d parallel iterations -> projected %.1f min for %d\n",
              label, sec, n_test, sec / n_test * n_target / 60, n_target))
}

# The manuscript's model has a random, but no fixed, slope for time. With a
# random slope and no fixed counterpart the random slopes are assumed to
# average zero; if the response rises (or falls) over the window differently
# by condition, this can bias the condition coefficient. The script therefore
# checks the coefficient against model-free and GAMM estimates (see PART 2)
# and `lmm: fixed_time_effect: true` adds the fixed slope for time.
lmm_formula_signal <- if (isTRUE(cfg$lmm$fixed_time_effect)) {
  pupil ~ condition + time_scaled + (1 + time_scaled + condition | participant)
} else {
  pupil ~ condition + (1 + time_scaled + condition | participant)
}

fit_signal_lmm <- function(d) {
  tryCatch(suppressMessages(suppressWarnings(lmer(lmm_formula_signal, data = d, REML = TRUE, control = LMER_CTRL))),
           error = function(e) NULL)
}

get_coef <- function(m, what = c("t", "beta")) {
  what <- match.arg(what)
  if (is.null(m)) return(NA_real_)
  tab <- tryCatch(summary(m)$coefficients, error = function(e) NULL)
  if (is.null(tab) || !(TERM %in% rownames(tab))) return(NA_real_)
  if (what == "t") tab[TERM, "t value"] else tab[TERM, "Estimate"]
}

# Resample participants with replacement; each draw becomes a new participant.
resample_participants <- function(split_list, idx) {
  d <- bind_rows(lapply(seq_along(idx), function(j) {
    x <- split_list[[idx[j]]]; x$participant <- j; x
  }))
  d$participant <- factor(d$participant)
  d
}

# =============================================================================
# LOAD DATA
# =============================================================================
header("Loading data")
if (!file.exists(cfg$input_csv)) stop("Input file not found: ", cfg$input_csv,
                                      " (set 'input_csv' in the config).")
cc <- cfg$columns
raw <- read.csv(cfg$input_csv, sep = cfg$csv_sep, dec = cfg$csv_dec, stringsAsFactors = FALSE)
miss <- setdiff(unlist(cc), names(raw))
if (length(miss)) stop("Missing columns (check 'columns' in the config): ", paste(miss, collapse = ", "))

df <- data.frame(participant = raw[[cc$participant]], trial = raw[[cc$trial]],
                 condition = as.character(raw[[cc$condition]]),
                 time_ms = as.numeric(raw[[cc$time]]), pupil = as.numeric(raw[[cc$pupil]]))
df <- df %>% filter(condition %in% c(REF, INT), !is.na(pupil)) %>%
  mutate(participant = factor(participant),
         condition = factor(condition, levels = c(REF, INT)),
         # z-scored for optimizer stability only; pupil stays in its own units
         time_scaled = as.numeric(scale(time_ms))) %>%
  arrange(participant, trial, time_ms) %>%
  group_by(participant, trial) %>% mutate(AR.start = row_number() == 1) %>% ungroup() %>%
  as.data.frame()

# unique integer key per trial, used for fast label permutation
df$trial_key <- as.integer(factor(paste(df$participant, df$trial, sep = "__")))
trial_lookup <- df %>% distinct(trial_key, participant, condition) %>% arrange(trial_key)
if (any(duplicated(trial_lookup$trial_key))) stop("Each trial must belong to one condition only.")

n_part <- nlevels(df$participant)
n_trials <- nrow(trial_lookup)
cat(sprintf("Samples: %d | Participants: %d | Trials: %d (%s: %d, %s: %d)\n",
            nrow(df), n_part, n_trials, INT, sum(trial_lookup$condition == INT),
            REF, sum(trial_lookup$condition == REF)))
add_summary("# Analysis summary", "", sprintf("- Input: `%s`", cfg$input_csv),
            sprintf("- %d samples, %d participants, %d trials (%s: %d; %s: %d)", nrow(df), n_part, n_trials,
                    INT, sum(trial_lookup$condition == INT), REF, sum(trial_lookup$condition == REF)),
            sprintf("- Resampling: %d permutations, %d bootstrap resamples, seed %d, %d cores",
                    rs$n_permutations, rs$n_bootstrap, rs$seed, rs$n_cores), "")

# One PSOCK cluster, reused across all resampling steps (portable across
# Windows/macOS/Linux). L'Ecuyer streams give each worker an independent,
# reproducible random-number stream.
cl <- makeCluster(rs$n_cores)
on.exit(try(stopCluster(cl), silent = TRUE), add = TRUE)
clusterEvalQ(cl, suppressMessages({ library(dplyr); library(lme4); library(lmerTest) }))
clusterSetRNGStream(cl, iseed = rs$seed)
clusterExport(cl, c("TERM", "LMER_CTRL", "lmm_formula_signal", "fit_signal_lmm", "get_coef",
                    "resample_participants"))

# =============================================================================
# PART 0 -- CONFOUND CHECK (data quality by condition)
# =============================================================================
# Missing data is a proxy for blink count/duration (Mathot & Vilotijevic,
# 2023). If data quality differed between conditions, a pupil difference could
# partly reflect it. Same model structure as PART 3: metric ~ condition +
# (1 + condition | participant), fit on single trials so the unequal trial
# counts per condition are properly weighted.
if (isTRUE(cfg$parts$confound)) {
  header("PART 0: Confound check (data quality by condition)")
  if (!is.null(cfg$trial_qc_csv) && file.exists(cfg$trial_qc_csv)) {
    qc <- read.csv(cfg$trial_qc_csv, stringsAsFactors = FALSE)
    if ("accepted" %in% names(qc)) qc <- qc[as.logical(qc$accepted), ]
    qc <- qc %>% rename(participant = !!cc$participant, trial = !!cc$trial, condition = !!cc$condition) %>%
      filter(condition %in% c(REF, INT)) %>%
      mutate(participant = factor(participant), condition = factor(condition, levels = c(REF, INT)))
    res0 <- bind_rows(lapply(intersect(cfg$qc_metrics, names(qc)), function(mtr) {
      d <- qc[!is.na(qc[[mtr]]), ]
      if (length(unique(d[[mtr]])) <= 1)
        return(data.frame(metric = mtr, note = "constant across trials; not fit"))
      m <- lmer(as.formula(paste0(mtr, " ~ condition + (1 + condition | participant)")),
                data = d, REML = TRUE, control = LMER_CTRL)
      broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE) %>% filter(term == TERM) %>%
        transmute(metric = mtr, estimate, std.error, statistic, df, p.value, conf.low, conf.high,
                  converged = length(m@optinfo$conv$lme4$messages) == 0,
                  singular_fit = isSingular(m, tol = 1e-4), note = "")
    }))
    print(res0, row.names = FALSE)
    write.csv(res0, out_path("p0", "confound_checks.csv"), row.names = FALSE)
    add_summary("## PART 0 -- Confound check", "",
                paste0("- ", res0$metric, ": beta = ", signif(res0$estimate, 3), ", p = ",
                       signif(res0$p.value, 3)), "")
  } else {
    cat("  trial_qc_csv not found; PART 0 skipped.\n")
  }
}

# =============================================================================
# PART 1 -- GAMM: shape of the response over time, with and without AR(1)
# =============================================================================
# pupil ~ condition + s(time, by = condition) + s(time, participant, bs = "fs")
# The by-condition smooths let the SHAPE (not only the level) differ between
# conditions; the factor-smooth gives each participant a non-linear trajectory.
# Model A (no AR1) is used to estimate rho, the lag-1 autocorrelation of its
# residuals computed strictly within trial. Model B refits with that rho
# (AR.start restarts the process at each trial). Both are reported.
# Note: in mgcv, residuals() of a bam fit are NOT whitened by rho, so residual
# autocorrelation after fitting is not a valid check of the correction; the
# comparison is made on fit statistics and smooth-term tables instead.
rho_hat <- NA_real_; beta_gamm <- NA_real_
if (isTRUE(cfg$parts$gamm)) {
  header("PART 1: GAMM (signal shape over time), without and with AR(1)")
  gamm_formula <- as.formula(sprintf(
    "pupil ~ condition + s(time_ms, by = condition, k = %d) + s(time_ms, participant, bs = 'fs', k = %d)",
    cfg$gamm$k_condition, cfg$gamm$k_participant))
  mA <- bam(gamm_formula, data = df, method = "fREML", discrete = TRUE)
  r <- residuals(mA)
  lagged <- c(NA, r[-length(r)]); lagged[df$AR.start] <- NA
  rho_hat <- cor(r, lagged, use = "complete.obs")
  cat(sprintf("  Within-trial lag-1 residual autocorrelation (Model A): rho = %.4f\n", rho_hat))
  mB <- bam(gamm_formula, data = df, method = "fREML", discrete = TRUE, rho = rho_hat, AR.start = df$AR.start)

  sA <- summary(mA); sB <- summary(mB)
  cmp <- data.frame(model = c("A: no AR1", "B: AR1"), rho = c(0, rho_hat),
                    dev_explained = c(sA$dev.expl, sB$dev.expl), r_sq_adj = c(sA$r.sq, sB$r.sq))
  st <- bind_rows(
    data.frame(model = "A: no AR1", term = rownames(sA$s.table), sA$s.table, check.names = FALSE),
    data.frame(model = "B: AR1", term = rownames(sB$s.table), sB$s.table, check.names = FALSE))
  pt <- data.frame(term = rownames(sB$p.table), sB$p.table, check.names = FALSE)
  beta_gamm <- sB$p.table[TERM, "Estimate"]
  print(cmp, row.names = FALSE); print(st, row.names = FALSE); print(pt, row.names = FALSE)
  write.csv(cmp, out_path("p1", "gamm_AR1_comparison.csv"), row.names = FALSE)
  write.csv(st, out_path("p1", "gamm_smooth_terms.csv"), row.names = FALSE)
  write.csv(pt, out_path("p1", "gamm_parametric_terms.csv"), row.names = FALSE)

  # Difference curve (of interest - reference), population smooths only, with
  # a 95% CI from the full coefficient covariance (lpmatrix approach).
  grid <- sort(unique(df$time_ms))
  nd <- function(lev) data.frame(time_ms = grid, participant = df$participant[1],
                                 condition = factor(lev, levels = c(REF, INT)))
  labs_s <- sapply(mB$smooth, function(s) s$label)
  excl <- labs_s[grepl("participant", labs_s)]
  Xd <- predict(mB, nd(INT), type = "lpmatrix", exclude = excl) -
        predict(mB, nd(REF), type = "lpmatrix", exclude = excl)
  fit <- as.numeric(Xd %*% coef(mB))
  se <- sqrt(rowSums((Xd %*% vcov(mB, unconditional = TRUE)) * Xd))
  diff_tab <- data.frame(time_ms = grid, diff = fit, ci_lo = fit - 1.96 * se, ci_hi = fit + 1.96 * se)
  diff_tab$ci_excludes_zero <- diff_tab$ci_lo > 0 | diff_tab$ci_hi < 0
  write.csv(diff_tab, out_path("p1", "gamm_difference_curve.csv"), row.names = FALSE)
  cat(sprintf("  Difference curve: CI excludes zero at %d/%d time points\n",
              sum(diff_tab$ci_excludes_zero), nrow(diff_tab)))

  f <- cfg$figure
  p <- ggplot(diff_tab, aes(time_ms, diff)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), fill = f$ci_fill, alpha = 0.6) +
    geom_line(color = f$accent_color, linewidth = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.3) +
    labs(x = "Time from stimulus onset (ms)", y = sprintf("%s − %s", INT, REF)) +
    theme_minimal(base_size = 11, base_family = f$font_family) +
    theme(panel.grid.minor = element_blank(), axis.line = element_line(color = "grey40", linewidth = 0.3))
  ggsave(out_path("p1", "fig_gamm_difference_curve.pdf"), p, width = f$width_in, height = f$height_in)
  ggsave(out_path("p1", "fig_gamm_difference_curve.png"), p, width = f$width_in, height = f$height_in, dpi = 300)

  add_summary("## PART 1 -- GAMM", "",
              sprintf("- Within-trial residual autocorrelation rho = %.3f", rho_hat),
              sprintf("- Deviance explained: %.3f (no AR1), %.3f (AR1)", sA$dev.expl, sB$dev.expl),
              sprintf("- Difference curve: CI excludes zero at %d of %d time points",
                      sum(diff_tab$ci_excludes_zero), nrow(diff_tab)),
              paste0("- edf (AR1 model): ", paste(sprintf("%s = %.2f", rownames(sB$s.table), sB$s.table[, "edf"]),
                                                  collapse = "; ")), "")
}

# =============================================================================
# PART 2 -- FULL TIME-SERIES LMM (one model, one coefficient)
# =============================================================================
# pupil ~ condition + (1 + time_scaled + condition | participant), REML.
# Every sample of every accepted trial enters one model. The random slope for
# condition is the one required by Barr et al. (2013) for the tested effect;
# the random slope for time is a nuisance term that absorbs each participant's
# temporal trend. The model has no fixed effect of time, so its single
# coefficient is the average condition difference over the whole window. Its
# Wald/Satterthwaite inference assumes independent residuals, which the
# signal violates; 2b-2d check it with methods that do not.
beta_lmm <- NA_real_; t_lmm <- NA_real_
if (isTRUE(cfg$parts$lmm)) {
  header("PART 2: Full time-series LMM")
  m2 <- lmer(lmm_formula_signal, data = df, REML = TRUE, control = LMER_CTRL)
  conv <- length(m2@optinfo$conv$lme4$messages) == 0
  res2 <- broom.mixed::tidy(m2, effects = "fixed", conf.int = TRUE) %>%
    mutate(converged = conv, singular_fit = isSingular(m2, tol = 1e-4),
           n_samples = nrow(df), n_participants = n_part, n_trials = n_trials)
  print(res2, row.names = FALSE)
  write.csv(res2, out_path("p2", "lmm_fixed_effects.csv"), row.names = FALSE)
  beta_lmm <- res2$estimate[res2$term == TERM]; t_lmm <- res2$statistic[res2$term == TERM]

  # Agreement check: the coefficient should match the average condition
  # difference estimated without this model's assumptions (mean of the
  # participant-level differences) and the GAMM's parametric term.
  pdiff <- df %>% group_by(participant, condition) %>% summarise(m = mean(pupil), .groups = "drop") %>%
    group_by(participant) %>% filter(n() == 2) %>%
    summarise(d = m[condition == INT] - m[condition == REF], .groups = "drop")
  beta_raw <- mean(pdiff$d)
  tol <- cfg$lmm$agreement_tolerance
  rel_raw <- abs(beta_lmm - beta_raw) / abs(beta_raw)
  agree <- data.frame(estimate = c("LMM coefficient", "Mean participant-level difference", "GAMM parametric term"),
                      value = c(beta_lmm, beta_raw, beta_gamm))
  print(agree, row.names = FALSE)
  write.csv(agree, out_path("p2", "lmm_agreement_check.csv"), row.names = FALSE)
  agree_ok <- is.finite(rel_raw) && rel_raw <= tol
  if (!agree_ok) {
    msg <- sprintf(paste0("WARNING: the LMM coefficient differs by %.0f%% from the mean participant-level ",
                          "difference. The time course probably differs between conditions; set ",
                          "`lmm: fixed_time_effect: true` and re-run."), 100 * rel_raw)
    cat("  ", msg, "\n")
  }
  add_summary("## PART 2 -- Full time-series LMM", "",
              sprintf("- beta = %.4f, 95%% Wald CI [%.4f, %.4f], t = %.2f, p = %.3g (converged: %s, singular: %s)",
                      beta_lmm, res2$conf.low[res2$term == TERM], res2$conf.high[res2$term == TERM],
                      t_lmm, res2$p.value[res2$term == TERM], conv, isSingular(m2, tol = 1e-4)),
              sprintf("- Fixed effect of time in the model: %s", isTRUE(cfg$lmm$fixed_time_effect)),
              sprintf("- Agreement check: LMM %.4f vs. mean participant-level difference %.4f vs. GAMM %.4f -> %s",
                      beta_lmm, beta_raw, beta_gamm,
                      if (agree_ok) "consistent" else "INCONSISTENT: set lmm: fixed_time_effect: true"))

  # ---------------------------------------------------------------------------
  # 2b -- Block permutation. Condition is constant within a trial, so the
  # exchangeable unit is the whole trial: labels are shuffled across a
  # participant's trials (keeping that participant's number of trials per
  # condition), while every trial keeps its own sample sequence and hence its
  # autocorrelation. Permuting single samples would destroy that structure.
  # ---------------------------------------------------------------------------
  if (isTRUE(cfg$parts$lmm_permutation)) {
    header("PART 2b: Block permutation (trial labels shuffled within participant)")
    perm_one_signal <- function(i) {
      lab <- ave(as.character(trial_lookup$condition), trial_lookup$participant, FUN = sample)
      d <- df; d$condition <- factor(lab[d$trial_key], levels = levels(df$condition))
      abs(get_coef(fit_signal_lmm(d), "t"))
    }
    clusterExport(cl, c("df", "trial_lookup"), envir = environment())
    timing_test(cl, perm_one_signal, rs$n_permutations, "block permutation")
    null2 <- unlist(run_parallel(cl, rs$n_permutations, perm_one_signal, "permutation"))
    p2b <- perm_p(null2, t_lmm)
    r2b <- data.frame(t_obs = t_lmm, p_wald = res2$p.value[res2$term == TERM], p_perm = p2b,
                      n_perm = rs$n_permutations, n_valid = sum(!is.na(null2)))
    print(r2b, row.names = FALSE)
    write.csv(r2b, out_path("p2", "block_permutation.csv"), row.names = FALSE)
    write.csv(data.frame(iteration = seq_along(null2), abs_t = null2),
              out_path("p2", "block_permutation_null.csv"), row.names = FALSE)
    add_summary(sprintf("- Block permutation (%d): p = %.4g (%d valid refits)",
                        rs$n_permutations, p2b, sum(!is.na(null2))))
  }

  # ---------------------------------------------------------------------------
  # 2c -- Calibration of the block permutation. Small datasets are simulated
  # under a TRUE null (no condition effect, AR(1) noise with the rho estimated
  # in PART 1, same unbalanced condition proportions) and each is tested with
  # the same permutation scheme. If the scheme is valid, the p-values are
  # ~Uniform(0, 1): ~5% below .05. Small data are used because nesting a
  # permutation test inside a simulation on the full dataset is prohibitive.
  # ---------------------------------------------------------------------------
  if (isTRUE(cfg$parts$lmm_calibration)) {
    header("PART 2c: Calibration of the block permutation under a simulated null")
    sim <- cfg$calibration_simulation
    rho_sim <- if (is.na(rho_hat)) sim$rho_if_gamm_skipped else rho_hat
    calib_one <- function(i) {
      rows <- list(); k <- 1
      for (p in seq_len(sim$n_participants)) {
        n_int <- round(sim$n_trials_per_participant * sim$prop_of_interest)
        labs <- sample(c(rep(INT, n_int), rep(REF, sim$n_trials_per_participant - n_int)))
        icpt <- rnorm(1, 0, sim$intercept_sd)
        for (tr in seq_len(sim$n_trials_per_participant)) {
          noise <- as.numeric(arima.sim(list(ar = rho_sim), n = sim$n_samples_per_trial, sd = sim$noise_sd))
          rows[[k]] <- data.frame(participant = paste0("S", p), trial = tr, condition = labs[tr],
                                  time_ms = seq(0, by = sim$sample_step_ms, length.out = sim$n_samples_per_trial),
                                  pupil = icpt + noise)          # no condition effect: true null
          k <- k + 1
        }
      }
      d <- bind_rows(rows)
      d$participant <- factor(d$participant); d$condition <- factor(d$condition, levels = c(REF, INT))
      d$time_scaled <- as.numeric(scale(d$time_ms))
      d$trial_key <- as.integer(factor(paste(d$participant, d$trial)))
      lk <- d %>% distinct(trial_key, participant, condition) %>% arrange(trial_key)
      t_obs <- get_coef(fit_signal_lmm(d), "t")
      if (is.na(t_obs)) return(NA_real_)
      nul <- replicate(rs$n_calibration_permutations, {
        lab <- ave(as.character(lk$condition), lk$participant, FUN = sample)
        dd <- d; dd$condition <- factor(lab[dd$trial_key], levels = c(REF, INT))
        abs(get_coef(fit_signal_lmm(dd), "t"))
      })
      if (sum(!is.na(nul)) < rs$n_calibration_permutations / 2) return(NA_real_)
      (sum(nul >= abs(t_obs), na.rm = TRUE) + 1) / (sum(!is.na(nul)) + 1)
    }
    clusterExport(cl, c("sim", "rho_sim", "REF", "INT", "rs"), envir = environment())
    p_cal <- unlist(run_parallel(cl, rs$n_calibration_datasets, calib_one, "calibration dataset"))
    v <- p_cal[!is.na(p_cal)]
    ks <- tryCatch(suppressWarnings(ks.test(v, "punif")), error = function(e) NULL)
    r2c <- data.frame(n_datasets = rs$n_calibration_datasets, n_valid = length(v),
                      n_perm_each = rs$n_calibration_permutations, rho_used = rho_sim,
                      pct_p_below_05 = 100 * mean(v < 0.05),
                      ks_D = if (!is.null(ks)) unname(ks$statistic) else NA,
                      ks_p = if (!is.null(ks)) ks$p.value else NA)
    print(r2c, row.names = FALSE)
    cat("  Calibrated if ~5% of p < .05 and the KS test does not reject uniformity.\n")
    write.csv(r2c, out_path("p2", "calibration_summary.csv"), row.names = FALSE)
    write.csv(data.frame(dataset = seq_along(p_cal), p_perm = p_cal),
              out_path("p2", "calibration_p_values.csv"), row.names = FALSE)
    add_summary(sprintf("- Calibration (%d null datasets x %d permutations, rho = %.3f): %.1f%% of p < .05 (nominal 5%%); KS p = %.3g",
                        length(v), rs$n_calibration_permutations, rho_sim, r2c$pct_p_below_05, r2c$ks_p))
  }

  # ---------------------------------------------------------------------------
  # 2d -- Cluster bootstrap. Participants (with all their data) are resampled
  # with replacement; the model is refit and the coefficient stored. BCa
  # interval with jackknife (leave-one-participant-out) acceleration. The
  # permutation tests "is there an effect?"; the bootstrap describes how much
  # the estimate would vary across samples of participants.
  # ---------------------------------------------------------------------------
  if (isTRUE(cfg$parts$lmm_bootstrap)) {
    header("PART 2d: Cluster bootstrap BCa CI for the LMM coefficient")
    split_df <- split(df, df$participant, drop = TRUE)
    boot_one_signal <- function(i) {
      d <- resample_participants(split_df, sample(seq_along(split_df), length(split_df), replace = TRUE))
      get_coef(fit_signal_lmm(d), "beta")
    }
    jack_one_signal <- function(j) get_coef(fit_signal_lmm(droplevels(bind_rows(split_df[-j]))), "beta")
    clusterExport(cl, c("split_df"), envir = environment())
    timing_test(cl, boot_one_signal, rs$n_bootstrap, "cluster bootstrap")
    boot2 <- unlist(run_parallel(cl, rs$n_bootstrap, boot_one_signal, "bootstrap"))
    jack2 <- unlist(run_parallel(cl, length(split_df), jack_one_signal, "jackknife"))
    ci <- bca_ci(boot2, beta_lmm, jack2)
    r2d <- data.frame(beta_obs = beta_lmm, ci_bca_lo = ci[["lo"]], ci_bca_hi = ci[["hi"]],
                      z0 = ci[["z0"]], a = ci[["a"]], n_boot = rs$n_bootstrap,
                      n_boot_valid = sum(!is.na(boot2)), n_jack_valid = sum(!is.na(jack2)))
    print(r2d, row.names = FALSE)
    write.csv(r2d, out_path("p2", "bootstrap_bca.csv"), row.names = FALSE)
    write.csv(data.frame(iteration = seq_along(boot2), beta = boot2),
              out_path("p2", "bootstrap_draws.csv"), row.names = FALSE)
    add_summary(sprintf("- Cluster bootstrap (%d): BCa 95%% CI [%.4f, %.4f]",
                        rs$n_bootstrap, ci[["lo"]], ci[["hi"]]))
  }
  add_summary("")
}

# =============================================================================
# PART 3 -- TRIAL-LEVEL FEATURES
# =============================================================================
# Six scalar features per trial (units follow the pupil signal, e.g. mm):
#   slope_initial  slope (per second) in the initial window
#   slope_global   slope over the whole response window
#   slope_late     slope in the late window
#   auc            area under the curve (trapezoidal, signal x ms)
#   time_to_peak   time (ms) of the largest absolute deviation from baseline
#   peak_power     largest absolute deviation from baseline
# Feature windows must be fixed a priori (e.g. from prior literature), not
# chosen after inspecting the tested contrast.
# Each feature: feature ~ condition + (1 + condition | participant), REML,
# Wald/Satterthwaite inference, Holm correction across the six features.
# Trials missing any feature are dropped from all features so that the
# resampling in 3b uses the same trials for every feature.
FEATURES <- c("slope_initial", "slope_global", "slope_late", "auc", "time_to_peak", "peak_power")
if (isTRUE(cfg$parts$features)) {
  header("PART 3: Trial-level features")
  fw <- cfg$features
  slope <- function(t, s, win) {
    k <- t >= win[1] & t <= win[2] & !is.na(s)
    if (sum(k) < 2) return(NA_real_)
    unname(coef(lm(s[k] ~ I(t[k] / 1000)))[2])
  }
  auc <- function(t, s) { k <- !is.na(s); t <- t[k]; s <- s[k]
    if (length(s) < 2) NA_real_ else sum(diff(t) * (head(s, -1) + tail(s, -1)) / 2) }
  df_feat <- df %>% group_by(participant, trial, condition) %>%
    summarise(slope_initial = slope(time_ms, pupil, fw$initial_window_ms),
              slope_global  = slope(time_ms, pupil, c(-Inf, Inf)),
              slope_late    = slope(time_ms, pupil, fw$late_window_ms),
              auc           = auc(time_ms, pupil),
              time_to_peak  = if (all(is.na(pupil))) NA_real_ else time_ms[which.max(abs(pupil))],
              peak_power    = if (all(is.na(pupil))) NA_real_ else max(abs(pupil), na.rm = TRUE),
              .groups = "drop")
  n_before <- nrow(df_feat)
  df_feat <- df_feat %>% filter(if_all(all_of(FEATURES), ~ !is.na(.x))) %>% droplevels() %>% as.data.frame()
  cat(sprintf("  %d trials with all features (%d dropped)\n", nrow(df_feat), n_before - nrow(df_feat)))
  write.csv(df_feat, out_path("p3", "features_per_trial.csv"), row.names = FALSE)

  res3 <- bind_rows(lapply(FEATURES, function(ft) {
    m <- lmer(as.formula(paste0(ft, " ~ condition + (1 + condition | participant)")),
              data = df_feat, REML = TRUE, control = LMER_CTRL)
    broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE) %>% filter(term == TERM) %>%
      transmute(feature = ft, beta = estimate, se = std.error, t = statistic, df, p_wald = p.value,
                ci_wald_lo = conf.low, ci_wald_hi = conf.high, singular_fit = isSingular(m, tol = 1e-4))
  })) %>% mutate(p_holm = p.adjust(p_wald, method = "holm"))
  print(res3, row.names = FALSE)
  write.csv(res3, out_path("p3", "features_lmm_holm.csv"), row.names = FALSE)

  # ---------------------------------------------------------------------------
  # 3b -- Resampling for the features (ML fits).
  # Permutation: trial labels are shuffled within participant and the model is
  #   refit; the observed feature values are never modified. (A residual-based
  #   Freedman-Lane scheme with a reduced model that keeps the random slope for
  #   condition must be avoided: fitted() of that reduced model contains each
  #   participant's BLUP for the condition slope, so the effect leaks back into
  #   the "null" data and the test loses power.) The same shuffle is applied to
  #   all six features in each iteration, so the maximum |t| across features
  #   gives the max-t (Westfall & Young, 1993) family-wise adjusted p-values,
  #   which account for the correlation between features.
  # Bootstrap: participants resampled with replacement, BCa with jackknife
  #   acceleration; one resample is used for all six features.
  # ---------------------------------------------------------------------------
  if (isTRUE(cfg$parts$features_resampling)) {
    header("PART 3b: Label permutation + max-t, cluster bootstrap BCa (features)")
    fit_feat_ml <- function(ft, d) {
      f <- as.formula(paste0(ft, " ~ condition + (1 + condition | participant)"))
      m <- tryCatch(suppressMessages(suppressWarnings(lmer(f, data = d, REML = FALSE, control = LMER_CTRL))),
                    error = function(e) NULL)
      if (is.null(m)) m <- tryCatch(suppressMessages(suppressWarnings(lmer(f, data = d, REML = FALSE,
                                     control = lmerControl(optimizer = "Nelder_Mead", optCtrl = list(maxfun = 1e6))))),
                                     error = function(e) NULL)
      m
    }
    all_feat <- function(d, what) sapply(FEATURES, function(ft) get_coef(fit_feat_ml(ft, d), what))
    t_obs3 <- all_feat(df_feat, "t"); beta_obs3 <- all_feat(df_feat, "beta")

    perm_one_feat <- function(i) {
      d <- df_feat
      d$condition <- factor(ave(as.character(d$condition), d$participant, FUN = sample), levels = c(REF, INT))
      abs(all_feat(d, "t"))
    }
    split_feat <- split(df_feat, df_feat$participant, drop = TRUE)
    boot_one_feat <- function(i) all_feat(resample_participants(split_feat, sample(seq_along(split_feat),
                                                         length(split_feat), replace = TRUE)), "beta")
    jack_one_feat <- function(j) all_feat(droplevels(bind_rows(split_feat[-j])), "beta")
    clusterExport(cl, c("df_feat", "split_feat", "FEATURES", "fit_feat_ml", "all_feat", "REF", "INT"),
                  envir = environment())
    timing_test(cl, perm_one_feat, rs$n_permutations, "feature permutation (6 refits each)")
    null3 <- do.call(rbind, run_parallel(cl, rs$n_permutations, perm_one_feat, "permutation"))
    boot3 <- do.call(rbind, run_parallel(cl, rs$n_bootstrap, boot_one_feat, "bootstrap"))
    jack3 <- do.call(rbind, run_parallel(cl, length(split_feat), jack_one_feat, "jackknife"))

    max_null <- apply(null3, 1, function(r) if (all(is.na(r))) NA_real_ else max(r, na.rm = TRUE))
    res3b <- bind_rows(lapply(FEATURES, function(ft) {
      ci <- bca_ci(boot3[, ft], beta_obs3[[ft]], jack3[, ft])
      data.frame(feature = ft, beta_ml = beta_obs3[[ft]], t_obs_ml = t_obs3[[ft]],
                 p_perm = perm_p(null3[, ft], t_obs3[[ft]]), p_maxt = perm_p(max_null, t_obs3[[ft]]),
                 ci_bca_lo = ci[["lo"]], ci_bca_hi = ci[["hi"]],
                 n_perm_valid = sum(!is.na(null3[, ft])), n_boot_valid = sum(!is.na(boot3[, ft])))
    }))
    res3b <- left_join(res3 %>% select(feature, beta_reml = beta, p_wald, p_holm, ci_wald_lo, ci_wald_hi),
                       res3b, by = "feature")
    print(res3b, row.names = FALSE)
    write.csv(res3b, out_path("p3", "features_robust_inference.csv"), row.names = FALSE)
    write.csv(data.frame(iteration = seq_len(nrow(null3)), null3), out_path("p3", "features_permutation_null.csv"),
              row.names = FALSE)
    res3 <- res3b
  }
  add_summary("## PART 3 -- Trial-level features", "", "| feature | beta | p (Wald) | p (Holm) |",
              "|---|---|---|---|",
              sprintf("| %s | %.4g | %.3g | %.3g |", res3$feature, if ("beta" %in% names(res3)) res3$beta else res3$beta_reml,
                      res3$p_wald, res3$p_holm), "")
  if ("p_maxt" %in% names(res3))
    add_summary(sprintf("- %s: p_perm = %.3g, p_max-t = %.3g, BCa [%.4g, %.4g]", res3$feature, res3$p_perm,
                        res3$p_maxt, res3$ci_bca_lo, res3$ci_bca_hi), "")
}

# =============================================================================
# PART 4 -- LEAVE-ONE-SUBJECT-OUT (LOSO) SENSITIVITY OF THE PART 2 ESTIMATE
# =============================================================================
# The PART 2 model is refit once per participant, holding that participant
# out. Checks whether the pooled estimate depends on any single participant
# (e.g. participants with very few trials in one condition). Refits are
# identified by iteration number only, not by participant ID.
if (isTRUE(cfg$parts$loso) && isTRUE(cfg$parts$lmm)) {
  header("PART 4: Leave-one-subject-out (LOSO)")
  ids <- levels(df$participant)
  loso_one <- function(i) {
    m <- fit_signal_lmm(droplevels(df[df$participant != ids[i], ]))
    if (is.null(m)) return(data.frame(iteration = i, estimate = NA, std.error = NA, p.value = NA,
                                      converged = FALSE, singular_fit = NA))
    s <- summary(m)$coefficients
    data.frame(iteration = i, estimate = s[TERM, "Estimate"], std.error = s[TERM, "Std. Error"],
               p.value = s[TERM, "Pr(>|t|)"], converged = length(m@optinfo$conv$lme4$messages) == 0,
               singular_fit = isSingular(m, tol = 1e-4))
  }
  clusterExport(cl, c("df", "ids"), envir = environment())
  loso <- bind_rows(run_parallel(cl, length(ids), loso_one, "LOSO refit"))
  r4 <- data.frame(n_refits = nrow(loso), beta_full = beta_lmm,
                   beta_min = min(loso$estimate, na.rm = TRUE), beta_max = max(loso$estimate, na.rm = TRUE),
                   sign_ever_flipped = any(sign(loso$estimate) != sign(beta_lmm), na.rm = TRUE),
                   n_p_above_05 = sum(loso$p.value >= 0.05, na.rm = TRUE),
                   n_singular = sum(loso$singular_fit, na.rm = TRUE), n_not_converged = sum(!loso$converged))
  print(r4, row.names = FALSE)
  write.csv(loso, out_path("p4", "loso_refits.csv"), row.names = FALSE)
  write.csv(r4, out_path("p4", "loso_summary.csv"), row.names = FALSE)
  add_summary("## PART 4 -- LOSO", "",
              sprintf("- beta range across %d refits: [%.4f, %.4f] (full sample %.4f); sign flipped: %s; refits with p >= .05: %d",
                      r4$n_refits, r4$beta_min, r4$beta_max, beta_lmm, r4$sign_ever_flipped, r4$n_p_above_05), "")
}

stopCluster(cl)
writeLines(SUMMARY, file.path(OUT, "analysis_summary.md"))
writeLines(capture.output(sessionInfo()), file.path(OUT, "session_info.txt"))
write_yaml(cfg, file.path(OUT, "config_used.yaml"))
cat("\nDone. Summary:", file.path(OUT, "analysis_summary.md"), "\n")
