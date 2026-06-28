import numpy as np
import pandas as pd
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42        # Fixed seed — ensures full reproducibility
N    = 20        # 10 Women + 10 Men
OUT  = "output"  # Output directory

rng = np.random.default_rng(SEED)
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("  SYNTHETIC DATA GENERATOR -- 3-Channel Multimodal Framework")
print(f"  N={N} (10F/10M)  |  seed={SEED}  |  Trials={N*20}")
print("=" * 70)


# ==============================================================================
# SECTION 1 -- PARTICIPANT TRAITS (BFI-2-S)
# ==============================================================================
# Normative means and SDs from Soto & John (2017) -- general population:
#   Neuroticism  (N): M=2.90, SD~0.65
#   Extraversion (E): M=3.40, SD~0.65
#   Conscientiousness (C): M=3.30, SD~0.65
#   Agreeableness (A): M=3.70, SD~0.65
#   Openness (O): M=3.70, SD~0.65
#
# Neuroticism-Extraversion negative correlation (r~-.25) is preserved via
# a shared latent factor (ne_factor).
# ==============================================================================
print("\n[1/6] Generating participant traits (BFI-2-S) ...")

pid    = [f"P{str(i+1).zfill(2)}" for i in range(N)]
gender = ["Woman"] * 10 + ["Man"] * 10
rng.shuffle(gender)

# Shared latent factor capturing Neuroticism-Extraversion covariance
ne_factor = rng.normal(0, 1, N)

# BFI-2-S domain scores -- clipped to valid Likert range [1, 5]
bfi_N = np.clip(2.90 + 0.55 * ne_factor + rng.normal(0, 0.40, N), 1, 5)
bfi_E = np.clip(3.40 - 0.20 * ne_factor + rng.normal(0, 0.55, N), 1, 5)
bfi_C = np.clip(3.30                     + rng.normal(0, 0.65, N), 1, 5)
bfi_A = np.clip(3.70                     + rng.normal(0, 0.55, N), 1, 5)
bfi_O = np.clip(3.70                     + rng.normal(0, 0.60, N), 1, 5)

# Within-sample z-scores for channel generation (preserves rank order)
Nz_all = (bfi_N - bfi_N.mean()) / bfi_N.std()
Ez_all = (bfi_E - bfi_E.mean()) / bfi_E.std()
Cz_all = (bfi_C - bfi_C.mean()) / bfi_C.std()

participants = pd.DataFrame({
    "participant_id":          pid,
    "gender":                  gender,
    "age_range":               rng.choice(
                                   ["18-21", "22-25", "26-30", "31-35", "36-40"],
                                   N, p=[.30, .30, .20, .10, .10]),
    "bfi2s_neuroticism":       bfi_N.round(2),
    "bfi2s_extraversion":      bfi_E.round(2),
    "bfi2s_conscientiousness": bfi_C.round(2),
    "bfi2s_agreeableness":     bfi_A.round(2),
    "bfi2s_openness":          bfi_O.round(2),
    # Internal z-scores (removed before saving)
    "_Nz": Nz_all,
    "_Ez": Ez_all,
    "_Cz": Cz_all,
})

print(f"  Gender: {gender.count('Woman')}F / {gender.count('Man')}M  OK")
print(f"  N: M={bfi_N.mean():.2f} SD={bfi_N.std():.2f}  "
      f"E: M={bfi_E.mean():.2f}  C: M={bfi_C.mean():.2f}")


# ==============================================================================
# SECTION 2 -- STIMULUS SET (NAPS BE, Riegel et al., 2016)
# ==============================================================================
# 20 NAPS BE images covering 8 Plutchik emotion categories + Neutral.
# true_valence: NAPS published norm (-1 to +1)
# true_arousal: NAPS published norm (0 to 1)
# ==============================================================================
print("\n[2/6] Loading NAPS BE stimulus set (Riegel et al., 2016) ...")

#              image_id             emotion        valence_cat  intensity  true_v  true_a
stimuli = [
    ("Landscapes_121_h",  "Joy",          "Positive",  "Medium",  +0.55, 0.40),
    ("Animals_166_v",     "Joy",          "Positive",  "Medium",  +0.50, 0.38),
    ("Faces_172_h",       "Sadness",      "Negative",  "Medium",  -0.45, 0.50),
    ("Animals_074_h",     "Sadness",      "Negative",  "Medium",  -0.40, 0.46),
    ("Animals_030_h",     "Fear",         "Negative",  "HIGH",    -0.72, 0.82),
    ("Animals_011_h",     "Fear",         "Negative",  "Medium",  -0.48, 0.58),
    ("People_220_h",      "Disgust",      "Negative",  "HIGH",    -0.78, 0.85),
    ("People_198_h",      "Disgust",      "Negative",  "HIGH",    -0.76, 0.80),
    ("Landscapes_139_h",  "Anger",        "Negative",  "Medium",  -0.46, 0.60),
    ("Landscapes_026_h",  "Anger",        "Negative",  "Medium",  -0.42, 0.56),
    ("People_237_h",      "Surprise",     "Negative",  "Medium",  -0.28, 0.52),
    ("Animals_008_v",     "Surprise",     "Neutral",   "Low",     -0.04, 0.28),
    ("Objects_282_h",     "Neutral",      "Neutral",   "Low",     +0.02, 0.12),
    ("Landscapes_024_v",  "Neutral",      "Neutral",   "Low",     +0.03, 0.10),
    ("Faces_198_h",       "Neutral",      "Neutral",   "Low",     +0.01, 0.09),
    ("Objects_177_h",     "Neutral",      "Neutral",   "Low",     -0.01, 0.11),
    ("Faces_270_h",       "Anticipation", "Negative",  "Medium",  -0.26, 0.46),
    ("Animals_087_h",     "Anticipation", "Negative",  "Medium",  -0.30, 0.50),
    ("Landscapes_005_h",  "Anticipation", "Negative",  "Medium",  -0.24, 0.44),
    ("Objects_148_h",     "Anticipation", "Negative",  "Medium",  -0.20, 0.40),
]

# Plutchik (1980): Ring 1=HIGH intensity, Ring 2=Medium, Ring 3=Low
INTENSITY_RING = {"HIGH": 1, "Medium": 2, "Low": 3}

# NAPS published norms per emotion category (Riegel et al., 2016)
# Used for Fig 5 (Russell vs NAPS norms validation)
NAPS_NORMS = {
    "Joy":          {"valence": +0.52, "arousal": 0.39},
    "Anticipation": {"valence": -0.25, "arousal": 0.45},
    "Surprise":     {"valence": -0.16, "arousal": 0.40},
    "Fear":         {"valence": -0.60, "arousal": 0.70},
    "Anger":        {"valence": -0.44, "arousal": 0.58},
    "Disgust":      {"valence": -0.77, "arousal": 0.83},
    "Sadness":      {"valence": -0.43, "arousal": 0.48},
    "Neutral":      {"valence": +0.01, "arousal": 0.11},
}

print(f"  {len(stimuli)} stimuli loaded | "
      f"Emotions: {len(set(s[1] for s in stimuli))} categories  OK")


# ==============================================================================
# SECTION 3 -- TRIAL-LEVEL DATA GENERATION
# ==============================================================================
# Effect sizes calibrated to published literature ranges:
#   HR-personality correlations    : r = .25-.45  (Appelhans & Luecken, 2006)
#   Mouse-personality correlations : r = .20-.40  (Yamauchi & Xiao, 2018)
#   Keyboard-personality corrs     : r = .25-.45  (Epp et al., 2011)
#
# Person-level nuisance variance added independently of BFI-2-S to produce
# realistic effect sizes without artificial inflation.
# ==============================================================================
print("\n[3/6] Generating 400 trial-level rows ...")

rows = []

for _, p in participants.iterrows():
    Nz = p["_Nz"]
    Ez = p["_Ez"]
    Cz = p["_Cz"]

    # Resting HR: Task Force (1996) -- range 60-100 bpm
    hrf_base = rng.normal(72, 6)

    # Person-level nuisance baselines (idiosyncratic style not explained by personality)
    p_mouse_style   = rng.normal(0,   4.5)
    p_mouse_speed   = rng.normal(0,  42.0)
    p_kb_speed      = rng.normal(0, 120.0)
    p_mouse_lin_off = rng.normal(0,   0.14)

    for trial_num, (img, emo, vcat, intensity, tv, ta) in enumerate(stimuli, 1):

        is_neg = 1.0 if vcat == "Negative" else 0.0

        # -- HR REACTIVITY (H1, H6, H10) ------------------------------------
        # Stimulus arousal drives HR; Neuroticism amplifies response
        # sigma=4.5 bpm reflects Moofit HW401 wrist-sensor measurement noise
        hrf_r = (
            4.5 * ta
            + 1.8 * Nz * ta
            + rng.normal(0, 4.5)
        )
        hrf_stim = hrf_base + hrf_r

        # -- MOUSE TRAJECTORY VARIANCE (H2, H7) -----------------------------
        # Neuroticism x negative stimulus -> erratic movement
        # Conscientiousness -> smoother paths
        mouse_var = np.clip(
            18.0
            + 2.8 * Nz * is_neg
            + 1.5 * Nz * ta
            - 1.8 * Cz
            + p_mouse_style
            + rng.normal(0, 4.5),
            2, None)

        # -- MOUSE VELOCITY (H4) --------------------------------------------
        # Extraversion -> faster cursor movement (~22 px/s per SD)
        mouse_vel = np.clip(
            420.0
            + 22.0 * Ez
            + p_mouse_speed
            + rng.normal(0, 28.0),
            80, None)

        # -- MOUSE PATH LINEARITY (H5) --------------------------------------
        # Conscientiousness -> more regulated, straighter paths
        mouse_lin = np.clip(
            0.62
            + 0.12 * Cz
            - 0.03 * Nz * is_neg
            + p_mouse_lin_off
            + rng.normal(0, 0.10),
            0, 1)

        # -- MOUSE HESITATION (H8) ------------------------------------------
        # High arousal -> longer pause; Extraversion -> shorter hesitation
        mouse_hes = np.clip(
            300
            - 20.0 * Ez
            + 30.0 * ta
            + rng.normal(0, 40),
            50, None)

        # -- KEYBOARD LATENCY (H3, H8) --------------------------------------
        # Extraversion -> shorter RT (behavioural activation)
        # Neuroticism -> slight penalty on negative stimuli
        kb_lat = np.clip(
            1450.0
            - 90.0 * Ez
            + 25.0 * Nz * is_neg
            + p_kb_speed
            + rng.normal(0, 90.0),
            300, None)

        kb_dwell  = np.clip(105.0 + rng.normal(0, 10),  40, None)
        kb_lat_sd = np.clip(175.0 + 15.0 * Nz + rng.normal(0, 15), 20, None)

        # -- RUSSELL RATINGS (H9, H13) --------------------------------------
        # Valence tracks NAPS norm (Russell, 1980; Marchewka et al., 2014)
        # Arousal = scaled true_arousal + HR coupling (Neuroticism moderates)
        r_val = np.clip(tv * 0.85 + rng.normal(0, 0.18), -1, 1)

        coupling = 0.20 + 0.08 * Nz   # H13: Neuroticism tightens coupling
        r_aro = np.clip(
            (ta * 2 - 1)              # scale [0,1] -> [-1,+1]
            + coupling * (hrf_r / 8)  # H9: HR drives arousal rating
            + rng.normal(0, 0.18),
            -1, 1)

        # -- PLUTCHIK INTENSITY RING (H11) ----------------------------------
        plut_ring = INTENSITY_RING[intensity]

        rows.append({
            "participant_id":            p["participant_id"],
            "gender":                    p["gender"],
            "age_range":                 p["age_range"],
            "trial_num":                 trial_num,
            "image_id":                  img,
            "stimulus_emotion":          emo,
            "stimulus_valence":          vcat,
            "stimulus_intensity":        intensity,
            "true_valence":              round(tv,       3),
            "true_arousal":              round(ta,       3),
            "bfi2s_neuroticism":         p["bfi2s_neuroticism"],
            "bfi2s_extraversion":        p["bfi2s_extraversion"],
            "bfi2s_conscientiousness":   p["bfi2s_conscientiousness"],
            "bfi2s_agreeableness":       p["bfi2s_agreeableness"],
            "bfi2s_openness":            p["bfi2s_openness"],
            "hrf_baseline_mean":         round(hrf_base, 2),
            "hrf_stimulus_mean":         round(hrf_stim, 2),
            "hrf_reactivity":            round(hrf_r,    2),
            "mouse_trajectory_variance": round(mouse_var, 2),
            "mouse_velocity_mean":       round(mouse_vel, 1),
            "mouse_path_linearity":      round(mouse_lin, 3),
            "mouse_hesitation_duration": round(mouse_hes, 1),
            "keyboard_latency_mean":     round(kb_lat,    1),
            "keyboard_dwell_mean":       round(kb_dwell,  1),
            "keyboard_latency_sd":       round(kb_lat_sd, 1),
            "russell_valence":           round(r_val, 3),
            "russell_arousal":           round(r_aro, 3),
            "plutchik_emotion_label":    emo,
            "plutchik_intensity_ring":   plut_ring,
        })

trials = pd.DataFrame(rows)
print(f"  Generated {len(trials)} rows x {len(trials.columns)} columns  OK")


# ==============================================================================
# SECTION 4 -- 3-CHANNEL COMPOSITE Z-SCORE (H14)
# ==============================================================================
# Equal-weight composite of three z-scored channels.
# Keyboard latency sign inverted: shorter = more activated.
# ==============================================================================
print("\n[4/6] Computing 3-channel composite z-score (H14) ...")

for col in ["hrf_reactivity", "mouse_trajectory_variance", "keyboard_latency_mean"]:
    trials[f"_z_{col}"] = (trials[col] - trials[col].mean()) / trials[col].std()

trials["_z_keyboard_latency_mean"] *= -1   # invert: shorter latency = higher activation

trials["composite_3ch_z"] = (
    trials[["_z_hrf_reactivity",
            "_z_mouse_trajectory_variance",
            "_z_keyboard_latency_mean"]]
    .mean(axis=1)
    .round(3)
)

trials.drop(columns=[c for c in trials.columns if c.startswith("_z_")], inplace=True)
print("  Composite z-score added  OK")


# ==============================================================================
# SECTION 5 -- PARTICIPANT-LEVEL SUMMARY
# ==============================================================================
print("\n[5/6] Aggregating to participant-level summary ...")

summary = trials.groupby("participant_id").agg(
    gender                  = ("gender",                    "first"),
    bfi2s_neuroticism       = ("bfi2s_neuroticism",         "first"),
    bfi2s_extraversion      = ("bfi2s_extraversion",        "first"),
    bfi2s_conscientiousness = ("bfi2s_conscientiousness",   "first"),
    bfi2s_agreeableness     = ("bfi2s_agreeableness",       "first"),
    bfi2s_openness          = ("bfi2s_openness",            "first"),
    mean_hrf_reactivity     = ("hrf_reactivity",            "mean"),
    mean_mouse_variance     = ("mouse_trajectory_variance", "mean"),
    mean_mouse_velocity     = ("mouse_velocity_mean",       "mean"),
    mean_mouse_linearity    = ("mouse_path_linearity",      "mean"),
    mean_keyboard_latency   = ("keyboard_latency_mean",     "mean"),
    mean_russell_valence    = ("russell_valence",           "mean"),
    mean_russell_arousal    = ("russell_arousal",           "mean"),
    mean_composite_z        = ("composite_3ch_z",           "mean"),
).reset_index()

# Negative-trial means for H1 and H2
neg_summary = (
    trials[trials["stimulus_valence"] == "Negative"]
    .groupby("participant_id")
    .agg(
        mean_hrf_neg       = ("hrf_reactivity",            "mean"),
        mean_mouse_var_neg = ("mouse_trajectory_variance", "mean"),
    )
    .reset_index()
)
summary = summary.merge(neg_summary, on="participant_id")
print(f"  Summary: {len(summary)} participants x {len(summary.columns)} columns  OK")


# ==============================================================================
# SECTION 6 -- SAVE OUTPUT FILES
# ==============================================================================
print("\n[6/6] Saving CSV files ...")

trials.to_csv(f"{OUT}/trial_level_N20.csv", index=False)
print(f"  OK {OUT}/trial_level_N20.csv  ({len(trials)} rows)")

participants.drop(
    columns=[c for c in participants.columns if c.startswith("_")]
).to_csv(f"{OUT}/participant_level_N20.csv", index=False)
print(f"  OK {OUT}/participant_level_N20.csv  ({len(participants)} rows)")

summary.to_csv(f"{OUT}/participant_summary_N20.csv", index=False)
print(f"  OK {OUT}/participant_summary_N20.csv  ({len(summary)} rows)")


# ==============================================================================
# DISTRIBUTIONS AUDIT
# ==============================================================================
print("\n" + "=" * 70)
print("  DISTRIBUTIONS AUDIT")
print("=" * 70)

print("\nBFI-2-S (N=20, participant level):")
print(f"  {'Trait':<28} {'Mean':>6}  {'SD':>5}  {'Min':>5}  {'Max':>5}")
for trait, label in [
    ("bfi2s_neuroticism",       "Neuroticism"),
    ("bfi2s_extraversion",      "Extraversion"),
    ("bfi2s_conscientiousness", "Conscientiousness"),
    ("bfi2s_agreeableness",     "Agreeableness"),
    ("bfi2s_openness",          "Openness"),
]:
    col = summary[trait]
    print(f"  {label:<28} {col.mean():>6.2f}  {col.std():>5.2f}  "
          f"{col.min():>5.2f}  {col.max():>5.2f}")

print("\nChannels (n=400 trials):")
print(f"  {'Metric':<33} {'Mean':>8}  {'SD':>7}  Unit")
for col, unit in [
    ("hrf_reactivity",            "bpm above baseline"),
    ("hrf_baseline_mean",         "bpm resting"),
    ("mouse_trajectory_variance", "px2"),
    ("mouse_velocity_mean",       "px/s"),
    ("mouse_path_linearity",      "0-1"),
    ("mouse_hesitation_duration", "ms"),
    ("keyboard_latency_mean",     "ms"),
    ("russell_arousal",           "-1 to +1"),
    ("russell_valence",           "-1 to +1"),
]:
    print(f"  {col:<33} {trials[col].mean():>8.2f}  "
          f"{trials[col].std():>7.2f}  {unit}")

print(f"\nGender: {gender.count('Woman')}F / {gender.count('Man')}M / Total={N}")
print("\n" + "=" * 70)
print("  Synthetic data generation complete.")
print(f"  Output directory: {OUT}/")
print("=" * 70 + "\n")
