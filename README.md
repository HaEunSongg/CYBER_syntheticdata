# MultiModal Behavioral Observation during Psychometric Testing using IoT & AI
### First-Order Pipeline Verification : 3-Channel Framework × BFI-2-S Personality

**Programme:** Erasmus Mundus CYBER (Cyberspace Behavior and E-Therapy)  
**Institutions:** Université Paris Cité 
**Lab:** Centre Borelli (UMR9010), in collaboration with Thales AVS  
---

## Project Overview

This repository contains the Python source code for the master's project
**"MultiModal Behavioral Observation during Psychometric Testing using IoT & AI"**.

The framework combines three IoT channels:
- **HR** : Moofit HW401 wristband (heart rate, bpm)
- **Mouse dynamics** :trajectory, velocity, hesitation, path linearity (Thales platform)
- **Keyboard interaction** : inter-key latency, response time (Thales platform)

with the **BFI-2-S** personality instrument (30-item Big Five) and **NAPS BE** affective
image stimuli rated via **Russell's Circumplex Model** (valence/arousal) and
**Plutchik's Wheel of Emotions** (categorical emotion + intensity ring).

> ⚠️ **Important:** All analyses use a **synthetic dataset (N=20, 10F/10M, seed=42)**
> generated from literature-informed effect sizes for **first-order pipeline
> verification only** (Morris, White & Crowther, 2019). Results are not empirical
> findings about human participants.

---

## Repository Structure
---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/[yourname]/multimodal-behavioral-observation-psychometric.git
cd multimodal-behavioral-observation-psychometric
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate the synthetic dataset**
```bash
python 1_generate_synthetic_data.py
```

**4. Run the analysis and generate figures**
```bash
python 2_run_analysis_revised.py
```

Figures (Figures 1–10 from the project report) will be saved to the output directory.

---

## Dependencies

- Python 3.x
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn

All listed in `requirements.txt`.

---

## Key Results (Pipeline Verification)

| Hypothesis | Test | Result |
|---|---|---|
| H1: Neuroticism → HR reactivity (neg. stimuli) | Pearson r | r = +0.73, p < .001 ✓ |
| H2: Neuroticism → Mouse variance (neg. stimuli) | Pearson r | r = +0.52, p < .05 ✓ |
| H3: Extraversion → Keyboard latency | Pearson r | r = −0.56, p < .05 ✓ |
| H4: Extraversion → Mouse velocity | Pearson r | r = +0.50, p < .05 ✓ |
| H5: Conscientiousness → Mouse path linearity | Pearson r | r = +0.64, p < .01 ✓ |
| H14: 3-channel RidgeCV composite vs single channels | LOO-CV R² | CV R² = +0.21 ✓ |

All five directional hypotheses confirmed. Generated r values recovered exactly (Δ = 0.000).

---

## Citation

If you use this code, please cite:

> Song, H. (2026). *MultiModal Behavioral Observation during Psychometric Testing
> using IoT & AI — First-Order Pipeline Verification*. Master's project,
> Erasmus Mundus CYBER, Université Paris Cité / Universidade Lusófona,
> Centre Borelli (UMR9010).

---

## References

- Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies
  to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102.
- Appelhans, B. M., & Luecken, L. J. (2006). Heart rate variability as an index
  of regulated emotional responding. *Review of General Psychology, 10*(3), 229–240.
- Marchewka, A., et al. (2014). The Nencki Affective Picture System (NAPS).
  *Behavior Research Methods, 46*(2), 596–610.
- Soto, C. J., & John, O. P. (2017). Short and extra-short forms of the BFI-2.
  *Journal of Research in Personality, 68*, 69–81.

---

## License

MIT License — see `LICENSE` file.
