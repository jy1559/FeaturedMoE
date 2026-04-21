# RouteRec Writing Core
# Give this file to the LLM at the start of every writing session.

These rules take precedence over your general writing instincts. When in doubt, prefer precision over elegance.

---

## What Is RouteRec?

RouteRec is a sequential recommendation model that replaces shared feed-forward blocks in a SASRec-style backbone with sparse, behavior-guided expert routing. It extracts compact scalar summaries — **behavioral cues** — directly from interaction logs (item IDs, timestamps, optional category labels), and uses these cues to select which expert subset to activate at each of three routing stages (macro, mid, micro).

Two paths are kept strictly separate: the **sequential backbone** encodes the sequence for next-item prediction; the **routing control path** computes behavioral cues and selects experts. Behavioral cues do not modify the sequence representation. Hidden states do not drive routing.

**RouteRec is not** a feature-enrichment model, a multi-task model, or a hidden-state-only MoE.

---

## Canonical Terminology

Never introduce a synonym for a defined term. If two terms appear in the same section and a reader might wonder if they mean the same thing, one must be eliminated.

| Concept | Use | Never Use |
|---|---|---|
| Observable signals from logs that drive routing | **behavioral cues** | "behavioral features", "behavioral signals", "routing features", "routing signals" |
| The scalar vector summarizing one temporal window | **cue vector** | "feature vector", "routing input" |
| The four behavioral dimensions (Tempo/Focus/Memory/Exposure) | **cue families** | "feature families", "feature groups", "cue groups" |
| One named dimension | **[Name] family** — e.g., "Tempo family" | "tempo group", "tempo features" |
| Mechanism selecting which experts activate | **routing** | "gating", "dispatching", "switching" |
| The expert subset activated for one input | **computation path** or **route** | "inference path", "execution path", "active path" |
| Activating exactly k_g groups × k_e experts | **sparse (expert) activation** | "sparse gating", "selective activation" |
| Activating all experts with soft softmax weights | **dense (soft) mixing** | "full mixing", "soft gating", "dense routing" |
| Macro, mid, micro collectively | **routing stages** | "routing layers", "routing levels", "routing blocks" |
| SASRec-style self-attention base | **sequential backbone** | "encoder", "base model", "backbone model" |
| C experts aligned to one cue family | **expert group** | "expert cluster", "expert block", "expert module" |
| Top-k_g groups then top-k_e experts within each | **hierarchical sparse expert allocation** | "hierarchical routing", "two-level selection" |
| Cross-session history window | **macro scope** | "long-term scope", "global scope" |
| Within-session prefix window | **mid scope** | "within-session scope", "session-level scope" |
| Last-w-positions window | **micro scope** | "short-term scope", "local scope" |
| MoE in general | **Mixture of Experts (MoE)** first use; **MoE** after | "mixture model" |

**MoE-specific:** In RouteRec, *routing* = selecting which expert subset activates via behavioral cues (not token dispatch). *Expert* = a small FFN block (not a large sub-network or task head). *Expert group* = C experts per cue family (not a multi-task tower). *Sparse activation* = exactly k_g × k_e active; zero weight elsewhere (not approximately sparse).

---

## Terms to Define on First Use

These are not standard vocabulary for all readers. Define each before using it technically. Vary the form — relative clause, appositive, colon, parenthetical — don't use the same pattern twice on one page.

- **routing** — "a router that selects which subset of expert modules processes each input, rather than passing all inputs through the same shared transformation"
- **computation path / route** — "The specific expert modules activated for a given prefix form its *computation path*, or route."
- **behavioral regime** — "the behavioral regime of a session — its observable structural character, such as repeat-heavy consumption or fast exploratory browsing"
- **route commitment** — "route commitment: routing that concentrates on a small, consistent expert subset rather than diffusing weight broadly"
- **conditional computation** — "(executing only a data-dependent subset of model parameters, rather than the full model for every input)"
- **expert** — "Each expert is a small feed-forward network. RouteRec organizes experts into groups, each aligned with one cue family."
- **cue families** — "four cue families — Tempo (pace and interval dynamics), Focus (category concentration and switching), Memory (repetition and recurrence), and Exposure (popularity level and drift)"

---

## Sentence-Level Rules

**Problem before solution.** Every paragraph introducing a design choice states the problem first, then the solution. If a sentence says *what* and *why* simultaneously, split it.

**One technical claim per sentence.** Do not chain two distinct claims with "and" or "while."

**No vague qualifiers.** Replace: "often semantically opaque" → "does not expose which behavioral property triggered routing"; "may warrant" → "warrants"; "naturally" (as justification) → remove and state the reason explicitly; "somewhat" → remove or quantify.

**Active voice by default.** Passive only when the actor is unknown or irrelevant.

**Definitions as argument, not glossary.** Write definitions as part of the sentence they appear in, not as standalone dictionary entries.

---

## Never Do

- Alternate "behavioral cues" and "behavioral features" — use "behavioral cues" only.
- Use "routing signal" to mean the cue vector — use "behavioral cues" or "cue vector."
- Say "our model routes inputs" without specifying the destination.
- Assert "different behavior requires different computation paths" as self-evident — frame it as a design question.
- Describe routing stages as "layers."
- Write challenge–solution–experiment connections as explicit labels (e.g., "C1 is addressed by Section 3.2"). Let content make these connections.

---

## Experiment Rules

- **Name datasets explicitly** when claiming "dynamic" or "stable" settings: "(Foursquare, KuaiRec, LastFM)."
- **Three distinct concepts:** Protocol (sessionized 70/15/15, 30-min threshold), Evaluation (HR/NDCG/MRR at k∈{5,10,20}, seen-target primary), Selection (highest mean seen-target validation score over 9 metrics). Never conflate.
- **"Seen-target" qualifier** on every main result sentence. Unseen-target is supplementary only.

---

## Checklist Before Submitting Any Generated Text

- [ ] "behavioral features" → "behavioral cues" everywhere
- [ ] "routing signal" as cue vector → "behavioral cues" or "cue vector"
- [ ] "feature families/groups" → "cue families"; "expert cluster/block" → "expert group"
- [ ] All terms above defined in natural sentences before first technical use
- [ ] No paragraph opens with solution before problem
- [ ] No vague qualifiers: "often," "somewhat," "may," "naturally"
- [ ] No two technical claims chained in one sentence
- [ ] Dataset names explicit for "dynamic"/"stable" claims
- [ ] "seen-target" on every main result
- [ ] "Different behavior → different paths" treated as design question, not axiom
