# RouteRec Writing Detail
# Reference when: discussing motivation, planning section structure, or doing deep revision.
# Not required for routine writing tasks — WRITING_CORE.md is sufficient for those.

---

## 1. Motivation

> **The motivation is not finalized and may change.** The philosophy and requirements below are stable. The storyline in Section 1.3 is a working draft.

### 1.1 Structural Philosophy

Motivation moves from **observations any reader accepts** to **design decisions specific to RouteRec**. Each level must feel like the natural next question raised by the previous one — not a jump.

- **L1**: Broadly observable, undisputed phenomenon. Establish the setting and observation.
- **L2**: The question this raises. What existing approaches do and what gap they leave.
- **L3**: RouteRec's angle. Introduce MoE as the tool, then immediately raise the routing design problem it creates.
- **L4**: RouteRec's answer. Behavioral cues as explicit router inputs, separating representation from routing control.
- **L5**: Three design questions → C1, C2, C3.

Interpretability is a **supporting motivation**, not the primary premise. It should strengthen L2-L4 by explaining why explicit cue-driven routing is attractive beyond raw accuracy: the selected route becomes easier to inspect when the router reads observable behavioral cues instead of only hidden-state geometry.

If a level introduces a concept without grounding it in the previous one, rewrite the transition.

### 1.2 Requirements (apply to any version)

1. **Establish session-based seqrec as the setting early.** Short sessions make behavioral differences structurally sharper — this must appear at or near L1.
2. **MoE enters as a tool, not a premise.** Its motivation emerges from L2's question ("how do we apply different computations for different sessions"), not from assumption.
3. **Give one sentence of justification for why conditional computation is on the table at all.** Do not argue that every behavioral difference requires a different computation path. Do argue that once MoE makes data-dependent computation available, shared transformation versus routed transformation becomes a meaningful modeling choice.
4. **Do not claim "different behavior requires different paths" as self-evident.** This invites: "inputs already differ, so outputs already differ — why change the path?" Frame it instead as a design question: *given conditional computation, what should drive routing?*
5. **The core design choice must be explicit**: behavioral cues as the router input, separate from hidden states. This is the distinguishing claim.
6. **Interpretability should be framed as an additional advantage of this separation, not as a standalone paper claim.** The point is that cue-driven routing exposes a cleaner control interface and makes route choices more inspectable post hoc.
7. **End at three design questions** mapping 1:1 to C1/C2/C3.

### 1.3 Current Working Storyline *(subject to change)*

**L1 — Setting and observation**
In session-based sequential recommendation, each session is short. Behavioral differences across users and sessions become structurally pronounced: some sessions are repeat-heavy, others fast and exploratory, some users stay narrowly focused while others shift categories frequently. These differences — in pace, repetition, focus, popularity exposure — are not merely differences in which items were clicked, but differences in the structural character of the entire sequence. The shorter the session, the more sharply these structural differences stand out.

**L2 — What this raises**
The natural question is how a model should use cross-session history and behavioral pattern information. Existing approaches answer by enriching representations — time-aware encoding, session-aware attention, feature injection — improving *what the model represents*. Mixture of Experts (MoE) opens a second option: behavior can also influence *what computation is performed* through conditional computation, rather than only enriching the representation passed through a shared transformation.

This is also where a lighter interpretability motive can enter. If a model uses conditional computation, the selected route is itself part of the model's behavior. That makes the routing input consequential: if it comes from observable behavioral cues, the route can be inspected in terms a reader already understands from the interaction log.

**L3 — MoE as the tool; the routing design problem**
This does not mean the paper must claim that every behavioral difference warrants a different computation path. The narrower point is that once MoE introduces conditional computation, routing becomes a meaningful modeling choice rather than a fixed implementation detail. Applying MoE to sequential recommendation then immediately raises: *what should drive routing?* Hidden states are optimized to predict the next item, not to explicitly expose behavioral regime. In short sessions, the hidden state may not yet reflect cross-session behavioral patterns when routing decisions are made. Expert specialization then remains implicit, and the reason a route was chosen is harder to inspect because routing is entangled with latent hidden-state geometry.

**L4 — RouteRec's answer**
RouteRec defines router inputs explicitly: behavioral cues computed directly from interaction logs — statistics summarizing tempo, focus, memory, and exposure — without passing through the hidden state. Representation path and routing control path are kept separate. Cross-session history, within-session progression, and local transitions become behavioral cues at distinct temporal scopes. This separation does not make the model intrinsically interpretable in a strong sense, but it does make routing behavior more inspectable because route changes can be related back to explicit behavioral quantities.

**L5 — Three design questions**
- What behavioral cues to use and how to construct them from ordinary logs → C1
- At what temporal scopes routing decisions should be made → C2
- How expert selection should be structured to make routing explicit and committed → C3

### 1.4 Motivation Vocabulary

Use these consistently. Meanings are fixed for this paper.

| Term | Meaning |
|---|---|
| **behavioral heterogeneity** | Structural differences in interaction patterns — pace, repetition, focus, exposure — beyond the fact that different items were clicked |
| **behavioral regime** | Observable structural character of a session: e.g., repeat-heavy, fast exploratory, narrowly focused. Not a latent cluster — derived from log statistics |
| **structural character** | Aggregate pattern of a sequence captured by statistics (pace, repetition rate, category entropy), as opposed to the specific items |
| **conditional computation** | Executing only a data-dependent subset of model capacity, rather than the full model uniformly |
| **cue vector** | Input to the router. In RouteRec: a behavioral cue vector. In prior work: often a hidden state |
| **hidden-state-only routing** | Routing where the router input is the sequence hidden state, without explicit behavioral statistics. The contrast case |
| **representation path** | Channel responsible for encoding the sequence into hidden states for prediction |
| **routing control path** | Channel responsible for computing behavioral cues and selecting experts. Kept separate from the representation path |
| **inspectable routing** | Routing whose decisions can be related back post hoc to explicit behavioral cues rather than only to latent hidden-state geometry |
| **expert specialization** | Extent to which individual experts learn to handle specific behavioral patterns. Does not happen automatically with hidden-state-only gating |
| **route commitment** | Property that routing concentrates on a small focused expert subset rather than diffusing broadly. Enforced by sparse activation |
| **cross-session history** | Interaction history spanning multiple past sessions. Used as the macro-scope behavioral context |

---

## 2. Paper Structure

### 2.1 Section Map

```
Introduction
  → motivation: session brevity → structural heterogeneity → representation enrichment versus routed conditional computation
  → MoE as mechanism → routing-control problem → behavioral cues as router inputs
  → three design choices → RouteRec high-level answer
  → does NOT enumerate C1/C2/C3 — that belongs in the Problem section

Related Work  (3 subsections, ~1 column total)
  → Session-based and Sequential Recommendation
  → Behavioral Context and Router Inputs
  → Mixture of Experts in Recommendation
  Each subsection ends by positioning the gap RouteRec addresses on that axis.
  Do NOT end every paragraph with "RouteRec does X instead" — close the gap once per subsection.

Problem and Challenges  ← between Related Work and Method
  (section name candidates: "Problem and Challenges", "Setup and Challenges", "Setting and Design Challenges")
  → Task formulation: session-based next-item prediction
  → Notation: session s_m, prefix x_{1:t}, target x_{t+1}, available fields (item ID, timestamp, optional category)
  → Available behavioral information from ordinary logs — what can be extracted without side-information encoders
  → MoE two-path design (brief): representation path vs. routing control path, Eq. for stage interface
    - Keep this light: one equation for the stage router, factorized index (g,c) for hierarchical selection
    - This is the setup for the three challenges, not a method section
  → Three challenges (C1, C2, C3) introduced explicitly and precisely
    - C1: constructing behavioral cues from sparse logs
    - C2: routing across different temporal scopes
    - C3: structuring expert selection to enforce route commitment
  → Challenge labels first appear here — NOT in Introduction
  → Ends: "The following section addresses each challenge in order."

Method
  → Overview: two-path design figure, brief reminder of which component addresses which challenge
    (this is now lean — task/notation/equation already in Problem section)
  → Behavioral Cue Construction  (C1)
  → Coarse-to-Fine Multi-Stage Routing  (C2)
  → Hierarchical Sparse Expert Allocation  (C3)
  → Training Objective

Experiments
  → Q1: Ranking quality — does RouteRec improve?
  → Q2: Routing control — what should drive routing?
  → Q3: Design justification — why the three-part structure?
  → Q4: Practicality — is the sparse design cost-efficient?
  → Q5: Routing semantics — what does the router actually capture?
```

### 2.2 Section Responsibilities

Each section has one job. Do not let responsibilities bleed across sections.

- **Introduction**: motivates the problem and the high-level approach
- **Related Work**: establishes the gap on each of three design axes
- **Problem and Challenges**: defines task, notation, the MoE two-path setup, and the three challenges formally
- **Method**: solves the three challenges
- **Experiments**: evaluates the solution

### 2.3 Related Work Plan (detailed)

**Subsection 1 — Session-based and Sequential Recommendation**
- Cover the setting: session-based seqrec, next-item prediction from short temporally bounded sessions
- Representative models: GRU4Rec, SR-GNN, STAMP, NARM (session-based); SASRec, BERT4Rec (Transformer backbone)
- Note behavioral heterogeneity across sessions as a recognized challenge (cite surveys)
- Gap: all models apply shared computation regardless of session structural character

**Subsection 2 — Behavioral Context and Router Inputs**
- Models that incorporate richer behavioral context: time-aware (TiSASRec), cross-session (HRNN, SHAN, HGN), feature-aware (FDSA, DIF-SR), session-aware hybrids
- Multi-interest and intent-aware models (MIND, ComiRec, ICSRec) — show behavioral regime shifts are recognized
- These improve the representation, not the computation path
- Gap: none of these define an explicit behavioral-cue input for conditional computation

**Subsection 3 — Mixture of Experts in Recommendation**
- Classical MoE (Jacobs, Jordan), sparse MoE (Shazeer, Switch)
- Recommendation MoE: MMoE, PLE (multi-task, not behavioral routing); FAME, M3SRec, HM4SR (implicit routing semantics)
- MoE routing design techniques (expert choice, z-loss) — useful but don't specify what should drive the router
- Gap: prior work does not address what observable signal should drive routing in session-based seqrec

### 2.4 Challenge–Method–Experiment Connections

These connections should be **felt by the reader from the content**, not spelled out as labels. Do not write "C1 is addressed by Section 3.2 and verified by Q2."

The one exception: the **Method Overview** subsection may explicitly state which component addresses which challenge — this helps the reader navigate the method.

Method and experiment sections may reference each other directly (e.g., "as the ablation in Q3 confirms"). Motivation-to-method connections should remain implicit.

### 2.5 What Moves from Method Overview to Problem and Challenges

The current Method Overview (§3.1) contains material that belongs in the Problem section:
- Task formulation and notation (prefix x_{1:t}, target x_{t+1}, session s_m)
- Available interaction fields (item ID, timestamp, optional category)
- The stage-routing equation (Eq. 1-2) and factorized expert index (g,c)

These should move to Problem and Challenges. Method Overview then becomes lean:
it references the figure, briefly recalls the two-path design, and states which component handles which challenge.

---

## 3. Definition Style Guide

Definitions should read as part of the argument, not as a glossary. Vary the grammatical form — avoid repeating the same pattern more than once per page.

| Form | Example |
|---|---|
| Relative clause | "a router that selects which subset of expert modules processes each input" |
| Appositive | "the behavioral regime of a session — its observable structural character, such as repeat-heavy consumption" |
| Naming sentence | "The expert modules activated for a given prefix form its *computation path*, or route." |
| Colon within argument | "This enforces *route commitment*: routing that concentrates on a small expert subset." |
| Parenthetical gloss | "conditional computation (executing only a data-dependent subset of parameters)" |

Introduce all four cue families together on first use: "four cue families — Tempo (pace and interval dynamics), Focus (category concentration and switching), Memory (repetition and recurrence), and Exposure (popularity level and drift)."
