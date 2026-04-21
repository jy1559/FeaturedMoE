# Method Section Revision Plan
*Working draft — treat as a blueprint before touching the tex.*

---

## 0. What the Method section must do

The Method section has one job: resolve the three tensions stated in §Problem. It should not re-motivate the problem, re-introduce notation already in §Problem Setup, or explain why behavioral cues matter (that was §Introduction). Every paragraph opens with the tension it is resolving or the constraint it is working within, then states what the design does.

The section must also connect cleanly to the figure. Figure 2 already shows three panels: (a) cue extraction, (b) serial backbone layout, (c) hierarchical sparse routing. The text should map onto those three panels in the same order that the subsections appear — no mismatch between figure and prose ordering.

---

## 1. Structural problems in the current draft

### 1.1 Overview is doing the wrong work

The current Overview opens correctly (figure ref, two-path design), but then introduces a full routing equation with flat expert index `e`:

```
π^(s)_t = softmax(Router_s(f^(s)_t) / τ_s)
h̃^(s)_t = Σ_e π^(s)_{t,e} · Expert^(s)_e(h^in,(s)_t)
```

Two problems with this:

**Problem A — wrong place.** DETAIL.md §2.5 is explicit: the stage-routing equation belongs in §Problem and Challenges, not Method Overview. Method Overview should be lean: figure reference, two-path recap, component mapping. No new equations.

**Problem B — wrong index.** The flat index `e` introduced here is inconsistent with the `(g,c)` grouped index introduced later in §Hierarchical. The reader absorbs `Expert_e` in Overview, then encounters `Expert_{g,c}` in §Hierarchical with no explanation of the relationship. Since `(g,c)` is the final design, there is no reason to introduce `e` first. Remove the Overview equations entirely; introduce the grouped index once, in §Hierarchical where it belongs.

After removing the equations, the Overview also has a mechanical roadmap paragraph ("Section X constructs... Section Y organizes...") that duplicates the final sentence of §Problem and adds section-number labels. CORE.md: do not write challenge–solution connections as explicit labels; let content make the connections. The one exception from DETAIL.md §2.4 is that Overview *may* state which component handles which challenge — so keep a single mapping sentence, but remove the per-section citation list.

The meta-commentary sentence ("We first present RouteRec as a design framework and refer to the main experimental configuration only when implementation details matter") has no argumentative content. Remove.

**Target Overview:** ~3 sentences. Reference Figure 2. Restate two-path design in one sentence. State the three components and which challenge each resolves. End. No equations.

---

### 1.2 Behavioral Cue Construction: three specific issues

**Issue A — subsection name.** Current name: "Route-Guiding Behavioral Cues." DETAIL.md §2.1 uses "Behavioral Cue Construction." The current name is longer and breaks parallelism with the other subsection names (both of which are noun phrases describing a construction or allocation process). Rename.

**Issue B — cue families redefined.** Introduction §RouteRec already introduced all four with parenthetical definitions:
> "four cue families: Tempo (interaction pace and interval dynamics), Focus (category concentration and switching), Memory (repetition and recurrence patterns), and Exposure (popularity level and drift)"

Current Method §Behavioral Cues then says: "The summary dimensions are grouped into four semantic families: Tempo, Focus, Memory, and Exposure. Tempo captures coverage, pace, and interval dynamics. Focus captures..."

CORE.md is explicit: never redefine a defined term. Method should name the cue families without re-glossing. One clean sentence that points to the structure and states the 16-dimensional count is enough. The per-family descriptions (Tempo captures X, Focus captures Y...) were necessary on first introduction in §Introduction; repeating them here is padding.

**Issue C — wrong term.** "Four semantic families" → must be "four cue families." This is in the CORE.md canonical table under *never use*.

**Issue D — "low-dimensional feature vector."** CORE.md: use "cue vector," never "feature vector." Current text: "maps each window to a low-dimensional feature vector." Change to "cue vector."

**Issue E — passive voice.** "the corresponding focus cues are handled by the fixed missing-signal rule" → "when the grouping field is absent, the Focus cues default to the fixed missing-signal value documented in Appendix~\ref{app:method-details}."

**Issue F — vague closing.** "The goal is not exhaustive feature engineering, but a small semantically legible cue bank that serves as a stable routing interface across datasets." Three problems: "semantically legible" is vague (legible to whom? in what sense?), "routing interface" was not defined, and the framing as "goal is not X but Y" is defensive. Replace: the paragraph should close by stating what the shared template achieves — that the same cue bank is applied at three temporal scopes, each exposing a different kind of behavioral evidence — without the negative framing.

---

### 1.3 Multi-Stage Routing: prose and equations are separated

Current structure:
1. Opening sentence (stages not redundant despite shared formula)
2. Serial layout diagram
3. Hidden-state notation paragraph: h^(0) through h^(3), ordering rationale
4. `\paragraph{Macro and mid (sequence-wise routing).}` — prose only
5. `\paragraph{Micro (position-wise routing).}` — prose only
6. "Formally, let f̄..." — standalone bridge paragraph
7. Sequence-wise equation block
8. Micro equation
9. Concluding paragraph

The problem: steps 4–5 explain the distinction in words, then steps 6–8 re-express it in equations after a "Formally" break. The equations look like an afterthought — something bolted on after the prose was already finished. In a well-organized methods section, each conceptual claim appears once: in the paragraph where it is argued, followed immediately by its equation. There should be no bridge paragraph that says "Formally, let..."; the formal statement should be part of the paragraph that motivated it.

**Fix:** Merge each paragraph with its equation. Delete the standalone "Formally, let..." paragraph. The macro/mid paragraph introduces `f̄^(s)_i` (the sequence-level cue summary), gives its equation, then states the routing distribution `π̄`. The micro paragraph introduces `f^(micro)_{i,t}` and gives its equation inline. The `f̄` notation needs to be introduced cleanly — one sentence stating it is the average of per-position cues over the sequence, with its definition either inline or as a brief display.

One additional point: the closing sentence of the current section ends with "the final model uses sparse route commitment so that only a small subset of stage-relevant groups and experts is activated at each decision." This is forward-referencing §Hierarchical, which is fine as a bridge, but "sparse route commitment" is awkward — *route commitment* is a property, not an action. Rewrite: "each stage therefore captures a distinct behavioral control view; Section~\ref{sec:hierarchical-routing} structures how expert selection enforces this commitment within each stage."

---

### 1.4 Hierarchical Sparse Expert Allocation: three issues

**Issue A — "feature-driven" in prose.** Current: "All three stages use the same feature-driven group-conditional router." CORE.md: never use "feature" as a synonym for "cue" in this paper. Change to "cue-driven group-conditional router."

**Issue B — "interpretable" vs. "inspectable".** Current closing: "The hierarchical factorization keeps that selection interpretable by organizing it around semantic control views." CORE.md uses "inspectable" throughout when referring to route choices that can be traced back to behavioral quantities. "Interpretable" has a broader meaning that invites XAI associations the paper does not claim. Change to "inspectable."

**Issue C — "followed by flattening (g,c) to expert index e".** After introducing the grouped product `π_{t,g,c} = p_{t,g} · r_{t,g,c}`, the text says "(g,c) [is] followed by flattening to expert index e." This reintroduces the flat index `e` that we are removing from Overview. If the final output equation uses `(g,c)` directly (as it does: `Σ_{g,c} π̂_{t,g,c} · Expert_{g,c}(h^in)`), then the flattening comment is unnecessary and confusing. Remove it. The output equation stands on its own with `(g,c)` index.

Similarly, "Eq. (X) is the same for macro, mid, and micro at the score level" is a clarifying note that can be embedded in prose more naturally: "This factored form applies identically at all three routing stages; stages differ only in which cue vector drives the group-level scores."

---

### 1.5 Training Objective: three issues

**Issue A — paragraph break missing.** The z-loss paragraph opens on the same line as the closing of the `L_cons` equation block. This is a formatting issue — insert a paragraph break between them.

**Issue B — flat index `e` in z-loss.** The z-loss is written over `a_{i,t,e}^s` (flat expert index `e`). If we remove `e` from the rest of the section and use only `(g,c)`, the z-loss should be written as `a_{i,t,g,c}^s` for consistency, or explained explicitly that `e` indexes the flattened `(g,c)` pair. The cleaner fix: use `(g,c)` throughout, including z-loss.

**Issue C — "gating numerics" is jargon.** "To stabilize gating numerics" — "gating" is in the never-use list (CORE.md: use "routing," not "gating"). Change to "To stabilize routing logits" or "To prevent the pre-softmax logits from growing unbounded."

---

## 2. Notation map (what is introduced where, and only there)

The current draft introduces symbols in inconsistent places. This is the target state:

| Symbol | Where introduced | Form |
|---|---|---|
| `s_m`, `x_{1:t}`, `x_{t+1}` | §Problem Setup | already correct |
| `H`, `w` (window sizes) | §Behavioral Cue Construction | inline with window definitions |
| `R^macro_t`, `R^mid_t`, `R^micro_t` | §Behavioral Cue Construction | three-line align equation |
| `A(·)` (summary operator) | §Behavioral Cue Construction | display equation `f^(s)_t = A(R^(s)_t)` |
| `f^(s)_t` (cue vector) | §Behavioral Cue Construction | via `A(·)`, defined here |
| `f̄^(s)_i` (sequence-level cue average) | §Multi-Stage Routing | introduced at first use in macro/mid paragraph, with inline definition |
| `h^(0)_t ... h^(3)_t` (hidden states at backbone positions) | §Multi-Stage Routing | introduced in the layout paragraph |
| `h^in,(s)_t` (hidden state entering stage s) | §Multi-Stage Routing | follows naturally from h^(0)..h^(3) notation; clarify which h maps to which stage |
| `G`, `C`, `E = GC` | §Hierarchical | first sentence of §Hierarchical |
| `p^(s)_{t,g}`, `r^(s)_{t,g,c}` | §Hierarchical | softmax definitions, with `z` and `u` as pre-softmax logits |
| `π^(s)_{t,g,c} = p · r` | §Hierarchical | product equation |
| `TopK(v, k)` | §Hierarchical | inline definition at first use |
| `p̄^(s)_{t,g}`, `r̄^(s)_{t,g,c}` | §Hierarchical | sparse masking equations |
| `π̂^(s)_{t,g,c}` (executed route) | §Hierarchical | product of sparse p̄ and r̄ |
| `h̃^(s)_t` (stage output) | §Hierarchical | final sum equation |
| `τ_s` (routing temperature) | §Hierarchical | inline, at first appearance in routing equations |
| `L_i` (valid sequence length) | §Training Objective | at first use in `π̄_i` definition |
| `a^s_{i,t,g,c}` (pre-softmax logits) | §Training Objective (z-loss) | use `(g,c)` index consistently |

**Symbols that should NOT appear:**
- Flat index `e` anywhere except inside `E = GC` (the count)
- "feature vector" — use "cue vector"
- `\mathcal{T}` as a routing-stage set in Overview — it is introduced later; do not reference it in Overview prose

Note on `h^in,(s)_t`: this symbol is introduced in Overview currently (with the equation) and then reused in §Hierarchical. After removing the Overview equation, it needs to be introduced in §Multi-Stage Routing when the h^(0)..h^(3) sequence is laid out. The mapping is: h^in,(macro)_t = h^(0)_t, h^in,(mid)_t = h^(1)_t, h^in,(micro)_t = h^(3)_t. This can be stated in one sentence after introducing the layout chain.

---

## 3. Paragraph-level draft: what each paragraph says

This is a blueprint, not final wording. Each bullet is one paragraph (or a paragraph + its equation).

### §4.1 RouteRec Overview

> **Para 1 (only paragraph, no equations).**
> Figure 2 shows RouteRec at the framework level: [brief panel description]. The model follows the two-path design from §Introduction — the sequential backbone processes the sequence for next-item prediction while a separate routing control path reads behavioral cues from interaction logs and uses them to select experts at each stage. Three components instantiate this design: Behavioral Cue Construction addresses the portability–informativeness tension, Coarse-to-Fine Multi-Stage Routing separates routing decisions by temporal scope, and Hierarchical Sparse Expert Allocation enforces route commitment. In the main configuration described below, variants and broader structural sweeps are in Appendix~\ref{app:method-details}.

*Length check: ~4 sentences. No equations. Component–challenge mapping is explicit per DETAIL.md §2.4 exception.*

---

### §4.2 Behavioral Cue Construction

> **Para 1: What signals are used, and why this is the right choice given C1.**
> The routing control path must read from interaction logs without requiring side information unavailable in many benchmark datasets. RouteRec uses four signal types present after standard sessionization: item identity, timestamp, optional coarse grouping label (category, genre, artist, venue type), and corpus-level popularity statistics computed from training data. These are not designed to be optimal predictive features; their role is to capture the observable structural character of a session — its pace, repetition, and focus patterns — in a form that is stable across heterogeneous datasets.

*Note: "optimal predictive features" framing is fine — it preempts the confusion between routing cues and representation inputs. Keep this distinction.*

> **Para 2: Three windows — definition + equations.**
> For session $s_m$ and target position $t$, RouteRec defines three temporal windows:
> [three-line align equation for R^macro, R^mid, R^micro]
> where H = 5 and w = 5 in the main configuration; Appendix~\ref{app:stage-layout} reports ablations on these sizes. A shared summary operator A(·) maps each window to a cue vector:
> [display: f^(s)_t = A(R^(s)_t), s ∈ T]

*Note: keep the current equations — they are already clean. Just change "low-dimensional feature vector" to "cue vector."*

> **Para 3: Four cue families — named, not re-defined.**
> The cue vector is organized into the four cue families introduced in §1: Tempo, Focus, Memory, and Exposure. In the main configuration, this yields a 16-dimensional cue vector per stage. Focus cues rely on the coarse grouping field; when that field is absent, we apply the fixed missing-signal default documented in Appendix~\ref{app:method-details}. Representative cue definitions appear in Table~\ref{tab:appendix-feature-families}.

*Note: do not repeat Tempo/Focus/Memory/Exposure descriptions. "Introduced in §1" is the correct pointer.*

> **Para 4: What the shared template achieves across stages.**
> The same template is applied at all three scopes, but each window exposes a different kind of evidence: macro reflects persistent cross-session tendency, mid tracks evolving within-session dynamics, and micro captures short-horizon transitions near the prediction target. The result is a small, inspectable cue bank that provides a consistent routing interface across datasets without requiring dataset-specific feature engineering.

*Note: removes "goal is not exhaustive feature engineering" negative framing. "Inspectable" per CORE. Last sentence closes C1 without labeling it.*

---

### §4.3 Coarse-to-Fine Multi-Stage Routing

> **Para 1: Problem statement (C2) and the main answer — stage placement.**
> To address the scope-alignment tension, RouteRec does not use a single routing view but places three routing stages at different points along the backbone, each operating on a cue vector computed from its corresponding temporal window. In the main configuration, the serial layout is:
> [layout chain diagram: self-attn → macro → mid → self-attn → micro]
> The first attention block contextualizes token interactions before the macro and mid routing decisions; the second refreshes those interactions immediately before the local micro stage.

*Note: current opening "To keep routing aligned with temporal scope, the stages are not redundant even though the router formula is shared" is slightly awkward ("not redundant" is a double negative). Reframe positively: stages are placed at different backbone positions, not just differently labeled.*

> **Para 2: Hidden-state sequence and the stage-input mapping.**
> Let h^(0)_t denote the hidden state after the first attention block, h^(1)_t after macro routing, h^(2)_t after mid routing, and h^(3)_t after the second attention block. Macro, mid, and micro consume h^(0)_t, h^(1)_t, and h^(3)_t as their respective inputs h^in,(s)_t. Appendix~\ref{app:stage-layout} reports ablations on alternative layout orderings.

*Note: this is exactly what the current text says; it is already clean. Keep as-is with minor trimming.*

> **Para 3: Macro and mid — sequence-wise routing + equation.**
> Macro and mid each produce one routing distribution per sequence, broadcast over all valid positions. Macro draws on cross-session history from R^macro; mid draws on the within-session prefix R^mid. One distribution per sequence is appropriate for slower-changing behavioral context: position-wise routing at these stages added variance without improving control resolution in our ablations. Let f̄^(s)_i = (1/L_i) Σ_t f^(s)_{i,t} denote the sequence-level cue summary. The routing distribution is:
> [equation: π̄^(s)_i = softmax(Router_s(f̄^(s)_i) / τ_s), s ∈ {macro, mid}]
> [equation: π^(s)_{i,t} = π̄^(s)_i  (broadcast)]

*Note: f̄ is introduced inline with its definition before the equation. No standalone "Formally, let..." paragraph.*

> **Para 4: Micro — position-wise routing + equation.**
> Micro computes a separate routing distribution at each position using short-horizon cues from R^micro. This allows the route to respond to immediate transition patterns without altering the router's structure:
> [equation: π^(micro)_{i,t} = softmax(Router_micro(f^(micro)_{i,t}) / τ_micro)]

> **Para 5: Closing — what "coarse-to-fine" means and transition to §Hierarchical.**
> The coarse-to-fine character of RouteRec comes from evidence scope and update frequency — macro and mid commit to one route per sequence, micro adapts per position — while the router formula is shared across all stages. Each stage captures a distinct behavioral control view. Section~\ref{sec:hierarchical-routing} specifies how expert selection within each stage enforces route commitment.

*Note: removes the current sentence "the final model uses sparse route commitment so that only a small subset of stage-relevant groups and experts is activated at each decision" — that sentence is a preview that undercuts §Hierarchical's opening. Replace with the clean transition above.*

---

### §4.4 Hierarchical Sparse Expert Allocation

The current structure here is mostly correct. The equations are right. The issues are: (a) redundant flat-`e` reference, (b) "feature-driven" → "cue-driven," (c) "interpretable" → "inspectable," (d) the two-paragraph split between "soft score structure" and "sparse masking" creates a pause that isn't needed.

> **Para 1: Problem statement (C3) and the structural answer.**
> To enforce route commitment, expert selection must concentrate on a small focused subset rather than spreading weight across all experts. RouteRec partitions the E = GC experts into G groups of C experts each, and uses a factored router that makes a group-level decision before making a within-group decision. This structure links the group-level choice to a semantic control view — in the main configuration, groups are aligned with the four cue families — while allowing multiple experts within each group to expand capacity without collapsing all selection into a single score. Each stage uses the same hierarchical router.

> **Para 2: Group-level and within-group distributions + product.**
> For stage s and position t, the router produces a group-level distribution and a conditional within-group distribution:
> [align: p^(s)_{t,g} = softmax_g(z^(s)_{t,g}), r^(s)_{t,g,c} = softmax_c(u^(s)_{t,g,c})]
> where z and u are learned linear projections of the cue vector f^(s)_t. The joint expert weight is their product:
> [equation: π^(s)_{t,g,c} = p^(s)_{t,g} · r^(s)_{t,g,c}]
> This factored form applies at all three stages; stages differ only in which cue vector drives the group-level scores.

*Note: removes "followed by flattening (g,c) to expert index e" and the separate "Eq. (X) is the same for macro, mid, and micro" sentence — both are now handled by the final prose sentence here.*

> **Para 3: Sparse masking — TopK at both levels + executed route + output.**
> Route commitment requires that only a small subset of groups and experts is active at each decision. Let TopK(v, k) denote the index set of the k largest entries of v. We retain the top-k_g groups:
> [equation: p̄^(s)_{t,g} = sparse-normalized p, masked to top-k_g]
> and, inside each retained group, the top-k_e experts:
> [equation: r̄^(s)_{t,g,c} = sparse-normalized r, masked to top-k_e within group]
> The executed route is their product:
> [equation: π̂^(s)_{t,g,c} = p̄^(s)_{t,g} · r̄^(s)_{t,g,c}]
> and the stage output aggregates only the active experts:
> [equation: h̃^(s)_t = Σ_{g,c} π̂^(s)_{t,g,c} · Expert^(s)_{g,c}(h^in,(s)_t)]
> In the main configuration, G = 4 and C = 3, with k_g = 3 and k_e = 2, so each routing decision activates 6 of 12 experts. The group prior remains soft at the score level; the executed computation path is explicitly sparse.

*Note: this merges the current two separate paragraphs ("soft score structure" and "sparse masking") into one continuous argument: here is the factored score → here is how we mask it → here is what executes. This flow is cleaner.*

> **Para 4: Closing — why sparse, not just efficient.**
> Sparse activation is not primarily a computational optimization. A dense soft mixture, regardless of how the scores are computed, reduces route differences across sessions because every expert contributes to every output. Sparsity enforces the commitment that makes routing semantically meaningful: a repeat-heavy session and a fast-exploratory one activate different expert subsets. The hierarchical factorization keeps that selection inspectable — the group-level choice corresponds to a cue family, the within-group choice to a specific capacity expansion — while the experts themselves remain fully learned. Dense mixing is evaluated as an ablation in Q3.

*Note: "interpretable" → "inspectable." "Feature-driven" removed. The current closing sentence "Dense soft mixing is treated as an ablation variant rather than the final choice" is fine but too brief; the expanded version above explains *why* dense fails, which is more useful.*

---

### §4.5 Training Objective

The current structure is correct. Three targeted fixes:

> **Fix 1:** Add a paragraph break before "To stabilize gating numerics..."

> **Fix 2:** "gating numerics" → "pre-softmax routing logits from growing unbounded" (remove "gating," CORE.md never-use list).

> **Fix 3:** z-loss uses flat index `e` in `a_{i,t,e}^s`. Change to `a_{i,t,g,c}^s` to match `(g,c)` notation throughout, or write `a_{i,t}^{s,(g,c)}` — pick one and apply consistently.

Everything else in §Training Objective is correct. The L_cons motivation ("encourages neighboring behavioral states to produce compatible routing distributions") is precise and should not be changed.

---

## 4. Cross-section consistency checks

These are things to verify after making the above changes — they touch multiple sections simultaneously.

| Check | Where to verify |
|---|---|
| "behavioral cues" used everywhere; "routing features," "routing signals," "behavioral features" absent | all of §Method |
| "cue families" used (not "semantic families," "feature families") | §4.2, §4.4 |
| "cue vector" used (not "feature vector") | §4.2 |
| "inspectable" used when referring to route traceability (not "interpretable") | §4.4 |
| "routing" used (not "gating," "dispatching") | §4.5 |
| Flat index `e` appears only in `E = GC` | §4.4, §4.5 |
| `f̄^(s)_i` introduced with its definition before first equation use | §4.3 |
| `h^in,(s)_t` introduced in §Multi-Stage after removing from Overview | §4.3 |
| C1/C2/C3 labels appear only in §Problem and in §Overview component mapping | §4.1 |
| "route commitment" used as a noun property (not "sparse route commitment" as an action) | §4.3 closing, §4.4 |
| Figure 2 panel ordering (a)→(b)→(c) matches §4.2→§4.3→§4.4 subsection order | §4.1 figure caption mention |

---

## 5. What does NOT change

- The three window equations (R^macro, R^mid, R^micro) — already correct and clean
- The serial layout chain diagram — already correct
- The sparse TopK equations (p̄, r̄) — already correct
- The product equations π = p·r and π̂ = p̄·r̄ — already correct
- The final output equation — already correct (uses `(g,c)`)
- L_CE, L_cons (structure and motivation), L combined — already correct
- Appendix pointer strategy — correct, keep
- The h^(0)..h^(3) notation in §Multi-Stage — already clean, keep
- "In our ablations" first-person phrasing in §Multi-Stage — acceptable by convention
