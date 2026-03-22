# Phase 5 Plan: Heterogeneous Experts & Routing Specialization

## 1. Phase 4 Takeaways & Residual Verdict

### 1.1 Residual Analysis
**Finding**: Residual mechanism does NOT beat plain MoE (base) on KuaiRec.

| Variation | Test MRR@20 | Delta vs Base | Verdict |
| --- | ---: | ---: | --- |
| `base` (plain MoE) | 0.1622 | — | ✅ Best, tuned reference |
| `shared_moe_warmup` | 0.1616 | -0.0006 | ⚠️ Competitive but -0.04% |
| `shared_moe_global` | 0.1615 | -0.0007 | ⚠️ Competitive but -0.04% |
| `shared_moe_stage` | 0.1613 | -0.0009 | ⚠️ Slightly worse |
| `shared_moe_fixed05` | 0.1614 | -0.0008 | ⚠️ Fixed worse than learnable |
| `shared_moe_fixed03` | 0.1609 | -0.0013 | ❌ Clearly hurt |
| `shared_only` | 0.1589 | -0.0033 | ❌❌ 2% loss, no MoE |

**Interpretation**:
- **Residual helps shared_only avoid collapse** ($-0.33\%$ → $-0.20\%$ vs base) but doesn't improve upon base
- **Hypothesis**: KuaiRec's current base model is already well-tuned for direct mix of shared + MoE. Adding learnable blending (residual) either (a) didn't find better alpha than 1.0, or (b) regularization hurt generalization
- **Not a tuning artifact**: All 7 residuals run with same hyperparams; base still wins
- **Recommendation**: Discontinue residual axis. Focus on improving base itself via **expert heterogeneity**.

### 1.2 K Axis Progress
- Current best: `12expert_dense`, `group_dense` ≈ 0.1618 avg
- **Still below base** (0.1622); no top-K combination has surpassed residual baseline
- Likely issue: flat experts lack specialization; top-K routing may overwhelm undifferentiated learning

### 1.3 Routing Insights (Phase 4 Diag)
**Router behavior is NOT "more balanced = better"**:

| Metric | Correlation w/ Test MRR@20 | Interpretation |
| --- | ---: | --- |
| `session_jitter` | -0.3656 | Unstable routing **hurts** (predictability matters) |
| `cv_usage` (imbalance) | +0.2917 | Some specialization **helps** (not all experts equal) |
| `top1_max_frac` | -0.0928 | Concentration slightly bad, but weak |
| `n_eff` | -0.1046 | "Balanced" routing not always best |

**→ Design implication**: Heterogeneous experts can naturally reduce jitter by making certain expert-item pairs more stable despite learning.

---

## 2. Heterogeneous Experts: Motivation & Design

### 2.1 Why Heterogeneous Experts?

**Current architecture**: All experts are identical deep FFN (MLP).
- Router learns which expert handles which item
- But all experts process item/user features identically
- No structural specialization → router must learn everything through weights alone
- Leads to: (a) harder routing learning, (b) less stable routing, (c) potential redundancy

**Heterogeneous design** (papers: ST-MoE, Mixtral, DeepSeek-MoE patterns):
- Different expert types naturally encode different inductive biases
- Router can leverage **feature characteristics** (popularity, session length, item type) to pre-assign
- Reduces routing jitter by constraining search space
- Better use of parameter budget (specialists vs. generalists)

### 2.2 Proposed Expert Heterogeneity Options

#### A. **Depth variation** (simple)
```
Expert pool:
- Type A (shallow): 1-2 layers, high-dim intermediate (quick patterns)
- Type B (medium): 3 layers, medium-dim (balanced)
- Type C (deep): 5 layers, low-dim intermediate (slow patterns)
```
**Pro**: Easy to implement, interpretable (shallow=speed, deep=nuance)  
**Con**: May not capture domain-specific patterns (popularity bias, session structure)

#### B. **Architecture variation** (medium complexity)
```
Expert pool:
- Type A (FFN/MLP): Dense GELU-activated, good for dense features
- Type B (SSM-like): Linear recurrent (gating + fast decay), good for temporal/sequential patterns
- Type C (Weak Attention): Small num heads, single query (light-weight specialization on item context)
- Type D (GRU-like): Gated recurrent, captures item-to-item correlations
```
**Pro**: Functionally diverse; can specialize by data domain (item vs. session patterns)  
**Con**: Higher implementation complexity; need careful initialization; harder to debug

#### C. **Parametric structure** (our sweet spot)
```
Expert pool:
- Type A (general): Standard MLP (current)
- Type B (popularity-aware): MLP with popularity-bucketed embeddings
- Type C (sequence-aware): MLP with session-length-weighted gates
- Type D (dynamic): MLP with dynamic feature importance (attention to feature subset)
```
**Pro**: Leverages domain knowledge; parameters still interpretable  
**Con**: May overfit to KuaiRec specifics; harder to generalize

### 2.3 **Recommended Design**: Hybrid architecture-depth approach

**Expert family (7-12 experts total)**:
- 3-4 baseline FFN (depths 2/3/4)
- 2-3 SSM-like (fast decay constants: τ=2, 5, 10)
- 1-2 weak attention (1-2 heads, small key dim)
- Optional: 1 GRU-like for sequences

**Rationale**:
- SSM naturally suits recommendation (item dynamics, recency bias)
- Weak attention can capture cross-item dependencies without full attention cost
- Depth variation provides generalization spectrum
- Still flat routing (no group, keeps stable)
- ~1.5K-2K params per expert vs. 3K-5K for full attention

---

## 3. Router Learning Strategy

**Challenge**: How do we guide router to learn heterogeneous assignment effectively?

### 3.1 Option A: Feature-based Prior (probability shaping)

```
For sample (user, item, features):
  popularity_score = item_popularity / max_popularity  # 0-1
  session_length_score = min(session_len, 20) / 20      # 0-1
  
  logit_bias = {
    'shallow_ffn': log(1 - popularity_score),           # likes dense/common items
    'deep_ffn': log(popularity_score),                  # likes rare/cold items
    'ssm_fast': log(recency_score),                     # likes recent patterns
    'ssm_slow': log(1 - recency_score),                 # likes long-term patterns
    'weak_attention': log(session_length_score),        # likes long sessions
    'gru_like': log(1 - session_length_score),          # likes short sessions
  }
  
  logits = router_hidden @ expert_params + logit_bias  # Add to learned routing
```

**Pro**: Interpretable, data-driven, leverages domain knowledge  
**Con**: Hand-crafted; may under/over-weight certain biases

### 3.2 Option B: KL-based Specialization Regularization

```
loss = main_loss + λ_kl * KL(learned_routing || prior_routing)

where prior_routing = normalized(softmax(logit_bias))
```

**Pro**: Soft constraint, router can override if needed, learnable balance  
**Con**: Adds hyperparameter tuning; KL may suppress exploration

### 3.3 Option C: Uncertainty/Difficulty Gate

```
For each sample, estimate "routing difficulty":
  difficulty = |hidden_variance| or max(routing_logits) - min(routing_logits)
  
If difficulty <= threshold:
  Use feature_based_prior (confident assignment to specialization)
Else:
  Use learned_routing (fallback to learned, when pattern unclear)
```

**Pro**: Adaptive; confident cases stay specialized, ambiguous cases learn  
**Con**: Adds conditional logic; may create training instability

### 3.4 **Recommended Strategy**: A + (soft B)

- **Primary**: Feature-based prior as logit bias (interpretable, fast)
- **Secondary**: Lightweight KL regularization (λ_kl = 0.01-0.05) if overfitting occurs
- **Validation**: Monitor routing assignment entropy per expert type; target specialization ratio 40-60% instead of uniform 1/K

---

## 4. Phase 5 Experimental Design

### 4.1 Validation Foundation (build on Phase 4 best)

**Use Phase 4 settings as baseline**:
- KuaiRecLargeStrictPosV2_0.2, FMoEN3 architecture
- Optimizer/LR/regularization from residual base best (stored in phase4 config)
- Combo lanes: **focus on C1 + C2** (best on base) + validate C3/C4 if heterogeneous helps factored

**Combo expansion**:
- Phase 4: 28 runs = 7 residuals × 4 combos
- **Phase 5 validation**: 12-16 combos = 3 heterogeneous variants × 4-5 combo lanes (total ~12-16 runs for quick turnaround)
- Then full sweep if promising

### 4.2 Core Phase 5 Experiments

**Axis V (Heterogeneous Variant): 4 variants (W axis replaces old R axis)**

| Variant | Expert Pool | Router Strategy | Rationale |
| --- | --- | --- | --- |
| `W_baseline_flat` | 7× identical FFN (depth 3) | Vanilla learned | Control: can't improve routing |
| `W_depth_only` | 3× FFN (depth 2/3/4), 4× SSM (τ=2,5,10,∞) | Vanilla learned | Test if architecture diversity helps blind routing |
| `W_depth_prior` | 3× FFN (depth 2/3/4), 4× SSM (τ=2,5,10,∞) | Feature prior + learned | Can router learn what prior suggests? |
| `W_depth_prior_kl` | 3× FFN (depth 2/3/4), 4× SSM (τ=2,5,10,∞) | Feature prior + learned + KL(λ=0.02) | Can soft constraint improve specialization? |

**Expected progression**:
1. `W_baseline_flat`: Should match Phase 4 base (sanity check)
2. `W_depth_only`: May hurt (no guidance, harder routing) but validate hypothesis
3. `W_depth_prior`: Should improve (inductive bias helps)
4. `W_depth_prior_kl`: May further improve or plateau (regularization safety)

### 4.3 Combination Strategy

**Axis Z (Combo): C1, C2, C3, C4** (same as Phase 4)

**Full grid**: W × Z = 4 variants × 4 combos = **16 runs** (Phase 5a)
- If all promising: extend to seed variation (3 seeds) → 48 runs (Phase 5b)
- If top performers: keep them for K-axis replication (Phase 5c)

### 4.4 Quick Validation Experiments (before full Phase 5)

**Option: 2-run quick check** (1-2 hours)
1. `W_depth_prior + C1` (most resource-efficient based on phase 4 best)
2. `W_baseline_flat + C1` (control, should match phase 4 base+C1)

If Delta > +0.0005 on `test_mrr20`, full sweep justified.

---

## 5. Implementation & Learning Specification

### 5.1 Expert Architecture Details

#### Baseline FFN (current, for reference)
```
FFN(hidden_dim=256):
  linear1(D_feat) → 256
  gelu
  linear2(256) → 256
  output(256) → orig_dim
```

#### Shallow FFN (depth 2)
```
linear(D_feat) → 512  # Wider to compensate
relu
linear(512) → orig_dim
```

#### Deep FFN (depth 4)
```
linear(D_feat) → 256
gelu
linear(256) → 256
gelu
linear(256) → 256
gelu
linear(256) → orig_dim
```

#### SSM-like (Linear Recurrent)
```
# For item sequence in session:
# Apply to pooled item features (mean across session)

class SSMExpert:
  def __init__(tau):
    self.decay = exp(-1/tau)  # τ=2 → 0.606 decay, τ=10 → 0.905
    self.gate_proj = linear(D_feat) → D_feat
    self.out_proj = linear(D_feat) → orig_dim
  
  def forward(x):
    gate = sigmoid(gate_proj(x))           # Control information flow
    decayed = gate * x + (1-gate) * 0      # Residual + memory decay
    return out_proj(decayed)
```

#### Weak Attention (light)
```
class WeakAttentionExpert:
  def __init__(num_heads=2):
    # Attention over item sequence, but only 2 heads, small key_dim
    self.mha = MultiheadAttention(
      embed_dim=D_feat, 
      num_heads=2, 
      head_dim=32,  # Lightweight
      dropout=0.1
    )
    self.ff = FFN_2layer(D_feat)
  
  def forward(item_seq, user_feat):
    # Add user feature as query bias
    attended = mha(item_seq, item_seq, item_seq)  # Self-attend items
    attended += user_feat.unsqueeze(0)             # Broadcast user context
    return ff(attended.mean(dim=0))                # Pool to orig_dim
```

### 5.2 Feature-based Prior Implementation

```python
class HeterogeneousRouter:
  def __init__(self, D_expert=7):
    self.router_mlp = MLPRouter(...)
    self.expert_types = [
      'shallow_ffn', 'mid_ffn', 'deep_ffn',
      'ssm_tau2', 'ssm_tau5', 'ssm_tau10', 'ssm_inf'
    ]
  
  def compute_logit_bias(batch_features):
    """
    batch_features: {
      'item_popularity': [B],       # 0-1 normalized
      'session_length': [B],         # 1-20+ normalized
      'recency_score': [B],          # how recent in session
    }
    """
    pop = batch_features['item_popularity']
    sess_len = batch_features['session_length']
    recency = batch_features['recency_score']
    
    bias = {
      'shallow_ffn': torch.log(pop + 0.1),              # Prefer freq items
      'mid_ffn': torch.log(1.0 - pop + 0.1),           # Balanced
      'deep_ffn': torch.log(1.0 - pop + 0.1) * 0.5,    # Slightly rare
      'ssm_tau2': torch.log(recency + 0.1),            # Favor recent
      'ssm_tau5': torch.log(recency + 0.1) * 0.5,      # Medium recency
      'ssm_tau10': torch.log(1.0 - recency + 0.1),     # Longer-term
      'ssm_inf': torch.zeros_like(pop),                # No preference
    }
    return bias  # shape [B, D_expert]
  
  def forward(hidden, features, **kwargs):
    logits = self.router_mlp(hidden)  # [B, D_expert]
    bias = self.compute_logit_bias(features)
    logits_with_bias = logits + bias * 0.5  # Scale prior influence
    routing = torch.softmax(logits_with_bias, dim=-1)
    return routing
```

### 5.3 Learning & Loss Design

```python
class HeterogeneousMoELoss:
  def __init__(self, lambda_kl=0.02):
    self.lambda_kl = lambda_kl
  
  def forward(main_loss, routing, learned_logits, bias_logits):
    # Main recommendation loss (unchanged)
    loss = main_loss
    
    # Optional: KL regularization to encourage prior-learning agreement
    if self.lambda_kl > 0:
      prior_routing = torch.softmax(bias_logits * 0.5, dim=-1)
      learned_routing = torch.softmax(learned_logits + bias_logits * 0.5, dim=-1)
      kl_div = F.kl_div(
        torch.log(learned_routing + 1e-8),
        prior_routing,
        reduction='batchmean'
      )
      loss += self.lambda_kl * kl_div
    
    return loss
```

### 5.4 Hyperparameter Space

**Fixed (from Phase 4 base best)**:
- Optimizer, LR schedule, batch size, regularization
- Number of stages, attention heads, etc.

**Tuning (Phase 5)**:
- Heterogeneous variant (W: 4 variants)
- Combo lane (Z: 4 lanes, secondary)
- **If promising, secondary sweep**:
  - Prior influence scale: 0.3, 0.5, 0.7 (instead of fixed 0.5)
  - KL weight: 0.01, 0.02, 0.05 (if enabling KL)
  - Expert initialization: standard vs. Xavier vs. special init per type

---

## 6. Success Criteria & Metrics

### 6.1 Phase 5a Target (16 runs, W×Z grid)
| Metric | Target | Rationale |
| --- | --- | --- |
| Test MRR@20 | ≥ 0.1625 | Exceed residual base (0.1622) + 0.0003 margin |
| Valid/Test gap | < 0.081 | Better generalization than Phase 4 avg (0.0805) |
| Routing entropy per expert | 0.5-0.7 (not 1.0) | Evidence of specialization |
| Jitter (stability) | < 0.35 | Lower than shared_only (0.565), comparable to base (0.323) |

**Go/No-Go Decision**:
- ✅ **Go to Phase 5b (seed variation)** if any variant hits `test_mrr20 ≥ 0.1623` and jitter < 0.34
- ⚠️ **Partial continue** if 0.1620-0.1622 range (competitive but not beating base)
- ❌ **Pivot to Phase 5c (K-axis only)** if all variants < 0.1620

### 6.2 Phase 5b Target (48 runs, 3 seeds × best 4 variants)
| Metric | Target |
| --- | --- |
| Mean test MRR@20 | ≥ 0.1625 |
| Std deviation | < 0.0008 (seed stability) |
| Consistency | Top variant wins all 3 seeds |

### 6.3 Phase 5c: K-axis replication (best heterogeneous variant)
- Apply best W variant to K-axis (top-K routing)
- Target: K + heterogeneous experts → test_mrr20 ≥ 0.1625 (both axes together)

---

## 7. Fallback & Pivot Strategies

### 7.1 If Heterogeneous Experts Don't Help
**Likely reasons**:
1. Feature prior is misaligned with learned routing (experts not specializing as intended)
2. Extra initialization variance hurts optimization
3. KuaiRec implicitly doesn't benefit from architectural diversity (uniform task)

**Fallback options**:
1. **Option A: Return to K-axis with base model** (no W axis)
   - Focus on top-K routing + larger expert count
   - Target: combine best residual+KL idea with K variants
2. **Option B: Routing-specific improvements** (no expert change)
   - Explicit routing stability loss (penalize jitter)
   - Learned routing temperature per layer
   - Expert affinity learning (item-expert co-occurrence matrix)
3. **Option C: Feature engineering approach**
   - Explicit item/session embeddings → concatenate to routing input
   - Latent factor routing (learn user/item latent → route via dot product)

### 7.2 If Heterogeneous Experts Partially Help (< 0.0003 delta)
- Worth exploring secondary tuning (prior scale, KL weight)
- May be marginal gain, not worth full K-axis replication
- Consider stopping in Phase 5a, consolidate findings

---

## 8. Implementation Roadmap

### Phase 5a: Heterogeneous Design + 16-run Validation (2-3 weeks)
1. **Week 1**:
   - Implement 4 expert architectures (FFN depths + SSM)
   - Implement feature-based prior computation
   - Add W axis to config/training loop
   - Dry-run sanity checks (W_baseline_flat should match Phase 4 base)

2. **Week 2**:
   - Launch 4-run quick validation (W_baseline_flat+C1, W_depth_prior+C1, + 2 alternates)
   - Parallel: 16-run full grid if quick validation promising
   - Monitor: routing entropy, jitter, expert load balance

3. **Week 3**:
   - Summarize 16-run results
   - Decide go/no-go for Phase 5b
   - Document patterns (which expert types dominate, priority among features)

### Phase 5b: Seed Stability (if 5a successful, 1-2 weeks)
- 3 seeds × 4 best variants = 12 runs
- Parallel training, target 1 week
- Consolidate variance estimates

### Phase 5c: K-axis Integration (if 5a/5b successful, 2-3 weeks)
- Apply best W to existing K variants (top-1, top-2, top-3, top-6)
- 4-6 K variants × best W = 4-6 runs
- Overlap with Phase 5b if possible

### Integration into shared_moe_warmup (if overall successful)
- Combine best heterogeneous variant + residual warmup
- Optional Phase 5 extension: 2-4 runs for combo validation

---

## 9. Monitoring & Diagnostics

### Per-run Outputs to Track:
1. **Routing metrics** (target: specialization without instability)
   - Assignment concentration per expert type (% of samples)
   - Entropy per feature bucket (popularity, session length)
   - Cross-epoch jitter

2. **Expert load balance** (target: healthy imbalance)
   - Load fraction per expert type (sum over samples)
   - Load variance (should be non-zero, unlike phase 4 base)

3. **Metric breakdown** (target: consistent improvement across all splits)
   - Test MRR@20 by popularity bin (cold/mid/hot)
   - Test MRR@20 by session length bin
   - Valid/test gap (should not widen)

4. **Training stability** (target: smooth learning curves)
   - Loss per epoch (main + KL, if enabled)
   - Validation MRR@20 trajectory
   - Routing entropy over epochs (should stabilize post-warmup)

### Comparison Baselines:
- Phase 4 base best: test_mrr20 0.1622, jitter 0.323, entropy ~0.95
- Phase 4 warmup best: test_mrr20 0.1616, jitter 0.336, entropy ~0.94
- Phase 4 K best: test_mrr20 0.1618, entropy ~0.92 (slightly lower, more specialized)

---

## 10. Related Work & Motivation Snippets

### Why Heterogeneous Experts Work (literature basis):
1. **ST-MoE (Lepikhin et al.)**: Different expert types (e.g., FFN vs. shared) in same MoE pool improves capacity utilization
2. **Mixtral-8x7B**: Heterogeneous parameter efficiency; different experts designed for different modalities
3. **DeepSeek-MoE**: Experts with separate dimensions for shared/special routing; implicit specialization
4. **Recommendation systems**: Different experts for different user segments (long-tail vs. head) improve robustness

### Routing Specialization (why prior helps):
- Bandit theory: UCB routing with side information (features) reduces regret
- Cold-start mitigation: feature-based prior guides router when data scarce
- Interpretability: expert assignment correlates with understood item/session properties

---

## 11. Success Story (Hypothetical Best Case)

**Scenario**: Heterogeneous experts + feature prior achieve 0.1626+ test MRR@20
```
Outcome:
  W_depth_prior + C1: 0.1626 (+0.0004 vs base)
  W_depth_prior + C2: 0.1625
  Routing specialization: 45% samples to depth-based experts, 55% to SSM
  Jitter: 0.329 (improved over base 0.323, stable)
  Interpretation:
    - Shallow FFN captures frequent items (popularity > 50)
    - Deep FFN captures cold items (popularity < 10)
    - SSM experts fine-grained on temporal patterns
    - Feature prior guided router without forcing (KL λ=0.02 only weak constraint)
  
Next: Combine with top-K routing (12exp_dense + heterogeneous) → aim for 0.1630+
```

---

## 12. Documentation Artifacts

**Output files** (to be created after Phase 5):
```
experiments/run/fmoe_n3/docs/
  phase5_results.md              # 16-run summary + routing analysis
  phase5_routing_assignments.csv # Per-expert assignment patterns
  phase5_feature_prior_analysis.png # Feature bias effectiveness
```

**Code artifacts** (to be created in Phase 5a):
```
experiments/models/
  fmoe_n3_heterogeneous.py       # Main architecture with W axis
  routing_strategies.py           # Feature prior, KL loss
experiments/configs/
  model/                          # W axis config templates
  tune_het_*.yaml                 # Phase 5 tuning configs
```

**Experiment tracking**:
- Phase 5a: 16 runs with common prefix `phase5_het_v1_`
- Phase 5b: 12 runs prefix `phase5_het_seed_`
- Phase 5c: 4-6 runs prefix `phase5_het_topk_`
