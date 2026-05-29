# PDAF: Probabilistic Diffusion Aggregation Fusion

## 1. Problem Statement

**Domain Generalization (DG)**: Train on source domain, generalize to unseen target domains.

For retinal vessel segmentation:
- Source: CHASE_DB1 dataset (or other available retinal dataset)
- Target: Unknown different dataset (different imaging conditions, devices)

**Key Challenge**: At inference, we only have target image — no source image, no target labels.

## 2. Core Idea: Latent Domain Prior

The paper proposes learning a **latent domain prior** `z` that captures domain shift information.

```
Source Image x_s → [Teacher] → features → [LPE] → z_tilde (optimal prior)
Target Image x_t → [Student] → features → [DPE] → z_hat (deployable prior)
                   ↓                              ↓
              teacher_logits               student_logits
```

**Two types of priors**:
- `z_tilde`: "Optimal" prior from LPE, uses paired (source, target) info
- `z_hat`: "Deployable" prior from DPE, uses only target info

## 3. Architecture Components

### 3.1 Teacher (Frozen Pretrained)

- Frozen U-Net pretrained on source domain
- Provides stable feature extraction
- Teacher features → LPE to generate `z_tilde`

### 3.2 Student (Trainable)

- U-Net with same architecture as teacher
- Student features → DPE to generate `z_hat`
- Both teacher and student share encoder/decoder architecture

### 3.3 LPE (Latent Prior Extractor) - Fig.3(a)

```
Input: teacher(source_features) || teacher(target_features)
Output: z_tilde (latent domain prior), mu, logvar (for KL)

z_tilde = LPE(source_cond, target_cond)
```

**Role**: Learns to extract domain shift info from paired source-target features.

### 3.4 DCM (Domain Compensation Module) - Fig.3(b)

```
Input: student_encoder_features, latent_prior z
Output: modulated features

 modulated_feature = feature * (1 + gamma(z)) + beta(z)
```

**Role**: Modulates student features using latent prior to compensate domain shift.

### 3.5 DPE (Diffusion Prior Estimator) - Fig.3(c)

```
Training: z_tilde (clean) → add noise → denoise → predict z_tilde
Inference: pure Gaussian noise → denoise → z_hat

condition: student(target_features)  # same as inference!
```

**Role**: Learns to predict domain shift from student features alone (mimics inference).

**Key Design**: DPE is conditioned on student features (not teacher) to match inference.

## 4. Training Flow

### Stage 1: Pretrain Teacher
```
teacher = pretrained U-Net on source
freeze teacher
```

### Stage 2: Train PDAF Modules

For each batch:
```
1. source_x, pseudo_target_x = Aug(source_x)
2. teacher_source_logits = teacher(source_x)
3. teacher_target_logits = teacher(pseudo_target_x)

4. z_tilde, mu, logvar = LPE(teacher(source), teacher(pseudo_target))

5. logits_tilde = student(pseudo_target_x) with DCM(z_tilde)
   # This is "ideal" prediction using optimal prior

6. student_cond = student.encode(pseudo_target_x)
   # DPE conditions on student features from pseudo-target (matches inference!)

7. z_hat = DPE(student_cond, target_prior=z_tilde)
   # DPE learns to predict z_tilde from student features

8. logits_hat = student(pseudo_target_x) with DCM(z_hat)
   # This is "deployable" prediction

9. Losses:
   - task_loss: 0.5 * (seg(logits_tilde, y) + seg(logits_hat, y))
   - sc_loss: MSE(sigmoid(logits_hat), sigmoid(teacher_source_logits))
   - kl_loss: KL(mu, logvar)  # regularize LPE
   - prior_loss: L1(tanh(z_hat), tanh(z_tilde))  # align z_hat and z_tilde
   - distill_loss: MSE(sigmoid(logits_hat), sigmoid(logits_tilde))
```

## 5. DPE Dual-Mode Training

DPE must learn two modes:

**Mode 1: Conditional Denoising** (standard DDPM)
```python
z_noisy = q_sample(z_tilde, step)  # add noise to z_tilde
z_hat = DPE(cond, target_prior=z_tilde)  # denoise back
loss = L1(tanh(z_hat), tanh(z_tilde))
```

**Mode 2: Pure Gaussian Generation** (inference mode)
```python
if random() < pure_gaussian_prob:
    z_gaussian = N(0,1)  # pure noise
    z_hat = DPE(cond, target_prior=None)  # generate from noise only
    loss = L1(tanh(z_hat), tanh(z_tilde))
```

This ensures DPE can generate meaningful priors from noise alone at inference.

## 6. Inference Flow

```
target_image x_t
  → student.encode(x_t) → features
  → prepare_condition(features) → cond
  → DPE(cond, target_prior=None) → z_hat (pure Gaussian start)
  → DCM(features, z_hat) → modulated features
  → student.decode(modulated_features) → logits
  → sigmoid → prediction
```

**No teacher, no LPE at inference!**

## 7. Domain Generalization Principles

### 7.1 Train-Test Alignment

| Component | Training | Inference |
|-----------|----------|-----------|
| DPE condition | `student(pseudo_target_features)` | `student(target_features)` |
| DPE start | `z_tilde` + noise OR pure Gaussian | pure Gaussian |
| DPE output | Predict `z_tilde` from condition | Predict `z_hat` from noise |

**Key**: DPE always sees student features, both train and inference.

### 7.2 Pseudo-Target vs Real Target

During training, pseudo-target is `Aug(source)` — same image, different style.
At inference, target is a completely new domain image.

The DPE learns: "given student features of a target-like image, predict the domain shift prior."

## 8. Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| task_loss | 0.5 | Train student to segment both with z_tilde and z_hat |
| sc_loss | 0.1 | Semantic consistency: logits_hat ≈ teacher(source) |
| kl_loss | 0.1 | Regularize LPE latent space |
| prior_loss | 0.2 (warmup) | Align z_hat to z_tilde |
| distill_loss | 0.3 (warmup) | logits_hat ≈ logits_tilde |

**Warmup**: prior_loss and distill_loss warm up over `prior_warmup_epochs` to avoid shock.

## 9. Code Organization

```
├── models/
│   └── unet_pdaf.py          # PDAFUNet, UNetBackbone, LPE, DCM, DPE
├── train_unet_pdaf_v2.py     # Training script
├── hyps/
│   └── unet_v2.yaml          # Hyperparameters
└── docs/
    └── PDAF_design_notes.md   # This file
```

## 10. Key Hyperparameters

```yaml
pdaf:
  lambda_task: 0.5       # Task loss weight
  lambda_sc: 0.3         # Semantic consistency weight
  lambda_kl: 0.05        # KL divergence weight
  lambda_prior: 0.2       # Prior alignment weight
  lambda_distill: 0.3     # Distillation weight
  prior_warmup_epochs: 15 # Warmup for prior/distill losses
  dpe_start_epoch: 5      # Start DPE training after student warms up
  dpe_dual_mode: false    # Enable pure Gaussian branch (SHOULD BE TRUE)
  dpe_pure_gaussian_prob: 0.4  # 40% chance of Gaussian branch per batch

model:
  dcm_modulation_scale: 0.1  # Domain compensation strength (SHOULD BE HIGHER)
```

## 11. Common Mistakes to Avoid

### Wrong: DPE uses teacher features in training
```python
# WRONG - causes train/test mismatch
dpe_cond = teacher_source_cond  # inference uses student features!
```

### Correct: DPE uses student features in training
```python
# CORRECT - matches inference
dpe_cond = student.encode(pseudo_target_x)  # same as inference
```

### Wrong: sc_loss too high
```python
# WRONG - competes with task_loss
lambda_sc: 0.3
```

### Correct: sc_loss lower than task_loss
```python
# CORRECT - task loss dominates
lambda_sc: 0.1, lambda_task: 0.5
```

### Wrong: DCM modulation scale too low
```python
# WRONG - domain compensation is too weak
dcm_modulation_scale: 0.1
```

### Correct: DCM modulation scale
```python
# RECOMMENDED - stronger domain compensation
dcm_modulation_scale: 0.5
```

## 12. Inference Without Source

**Critical DG requirement**: At inference, we don't have source image.

```
train: DPE(student(pseudo_target_features)) → predict z_hat
inference: DPE(student(target_features)) → predict z_hat

Both use student features → correct DG design
```

LPE is NOT used at inference. It's only for training to generate `z_tilde` supervision.

## 13. Domain Shift Representation

```
source domain → different imaging conditions (illumination, contrast, artifacts)
    ↓
teacher extracts domain-aware features
    ↓
LPE learns: "source looks like X, target looks like Y, shift is z"
    ↓
DPE learns: "given student features, predict the shift z"
    ↓
At inference: given target student features → predict shift → compensate
```

The latent prior `z` captures WHAT changed between domains (illumination, contrast, etc.)