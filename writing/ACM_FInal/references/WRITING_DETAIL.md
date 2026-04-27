# RouteRec Writing Detail
# Use this file when revising motivation, section structure, or paper-level storyline.
# WRITING_CORE.md remains the mandatory base rule set.

---

## 1. Storyline Philosophy

The paper should now be written around a stronger and cleaner question:

> In sessionized sequential recommendation, once MoE-based conditional computation is available, which recurring behavioral axes should define expert allocation?

This is stronger than the old motivation because it does three things at once:

1. It gives MoE a clear reason to appear.
2. It connects RouteRec to prior heterogeneity-aligned routing literature.
3. It prevents the paper from looking like an ad hoc feature-engineering story.
4. It makes the paper look like a routing-design study for seqrec, not like a narrow session-length argument.

### What Changed From the Old Motivation

The old version leaned too quickly into "behavioral cues" and "separate control path." That made the introduction feel half-method, half-motivation.

The new version should move in this order:

1. SeqRec has recurring behavioral variation in raw session logs.
2. Prior seqrec mostly uses that variation to enrich representation.
3. MoE adds a new modeling question: what should define expert specialization?
4. Prior MoE literature says routing works best when aligned with the source of heterogeneity.
5. In seqrec under a sessionized interaction view, raw logs suggest a small set of router-relevant behavioral axes.
6. RouteRec operationalizes those axes as lightweight behavioral cues.

This keeps the motivation at the **axis level** and lets the method introduce cues as the **operational answer**.

---

## 2. Core Framing

### 2.1 Main Claim

Use this framing repeatedly:

> RouteRec is not mainly a claim that sequential recommendation universally needs radically different computation paths. It is a claim that, once MoE-based conditional computation is introduced, expert allocation should be guided by routing signals aligned with recurring behavioral axes in sessionized interaction logs, rather than by prediction-optimized hidden states alone.

### 2.2 Headline Terms

Prefer these terms:

- **behavioral routing demand**
- **recurring behavioral axes**
- **behavioral cues**
- **context availability**
- **routing headroom**

Use **behavioral heterogeneity** only as a supporting term. Do not let it become the headline concept.

### 2.3 Stable Distinctions

- **behavioral routing demand**: the motivation-level framing
- **recurring behavioral axes**: the dataset-level descriptors that reveal routing-relevant structure
- **behavioral cues**: the sample-level operationalization used by the router
- **context availability**: a support condition that explains why routing headroom differs by dataset

If a paragraph mixes these without distinction, rewrite it.

---

## 3. Recommended Motivation Structure

### L1. Sessionized SeqRec Is Not Behaviorally Uniform

Start with sequential recommendation, then bring in sessionization only as the interaction view that makes recurring behavioral structure observable and usable.

The first observation should be simple and defensible:

- Interaction logs do not vary only in item identity.
- They also vary in structural character.
- That variation recurs along a small number of axes.

Good examples:

- pace or temporal regime
- ambiguity in local transitions
- repetition and carryover structure
- popularity exposure regime

Do not mention cue families yet.

### L2. Existing SeqRec Mostly Improves Representation Under Shared Computation

The next step is not "therefore MoE."

First say what prior work already does:

- time-aware modeling
- cross-session modeling
- side-information fusion
- intent or multi-interest modeling

Then state the gap:

> these methods improve what the model encodes, but they do not directly ask which computation path should be applied

### L3. MoE Introduces the Routing-Design Question

Now bring in MoE.

The key sentence should be close to this:

> Once conditional computation becomes available, expert allocation is no longer a fixed implementation detail. It becomes a modeling choice.

Then immediately pivot to the MoE literature:

- MMoE and PLE align experts with task relations
- AdaSparse and PEPNet align routing with domain or personalized priors
- HM4SR, M3SRec, and related work align expert usage with modality or temporal structure

The point is not "MoE is strong."

The point is:

> prior MoE work suggests that routing is most meaningful when specialization is aligned with the source of heterogeneity

### L4. SeqRec Needs Its Own Routing Axes

Now ask the paper's central question:

> in sessionized sequential recommendation, which recurring behavioral axes are router-aware enough to define expert allocation?

This is where the raw-log reading matters.

State that raw session logs repeatedly reveal:

1. **Tempo / regime diversity**
2. **Transition ambiguity**
3. **Memory regime**
4. **Exposure regime**
5. **Context availability** as a support condition

Important:

- The first four are routing-relevant axes.
- The fifth is not a cue family. It explains macro-level support and routing headroom.

### L5. RouteRec Operationalizes These Axes

Only here should the introduction descend into the RouteRec answer.

Use language like:

> RouteRec does not begin from an arbitrary bank of routing features. It starts from recurring sources of routing demand in raw session logs and operationalizes them as lightweight behavioral cues.

Then connect axis to cue family:

- Tempo / regime diversity -> Tempo family
- Transition ambiguity -> Focus family
- Memory regime -> Memory family
- Exposure regime -> Exposure family

Mention macro, mid, and micro scopes only after the axis story is clear.

### L6. Result Preview

The intro can end with a restrained preview:

> later experiments show the clearest gains where local branching is strong and repeated-session context is sufficiently available, while the margin narrows where routing headroom is limited

Do not put score tables, correlation coefficients, or appendix-style analysis here.

---

## 4. Recommended Introduction Paragraph Plan

Use six paragraphs unless there is a strong reason to collapse two.

### Paragraph 1

Seqrec setting first. Then explain that a sessionized view of the logs reveals recurring variation in structural behavior, not only item identity.

### Paragraph 2

Prior seqrec mostly enriches representation under a shared path.

### Paragraph 3

MoE changes the question from only representation learning to expert allocation. Position this with heterogeneity-aligned MoE literature.

### Paragraph 4

State the recurring behavioral axes that appear router-relevant in sessionized logs.

### Paragraph 5

Present RouteRec as the operationalization of those axes through lightweight cues on a routing control path separated from the sequential backbone.

### Paragraph 6

Preview the empirical pattern and state contributions.

### What Must Not Happen

- Paragraph 1 must not start with MoE.
- Paragraph 3 must not jump directly to cue families.
- Paragraph 4 must not look like a handcrafted feature list.
- Paragraph 5 must not read like a full method subsection.

---

## 4.5 Abstract Blueprint

The abstract should not begin with behavioral cues. It should begin with the seqrec setting and the routing-design question.

Recommended order:

1. **Setting and gap**
   Seqrec already uses behavioral evidence, but mostly to improve shared representations.
2. **MoE question**
   Conditional computation introduces a second question: what should define expert allocation?
3. **MoE bridge**
   Prior MoE work is strongest when routing aligns with the source of heterogeneity.
4. **SeqRec observation**
   A sessionized view of interaction logs shows recurring behavioral axes relevant to routing.
5. **RouteRec answer**
   RouteRec operationalizes these axes as lightweight behavioral cues on a routing control path separate from the sequential backbone.
6. **Method sketch**
   Mention macro, mid, and micro scopes plus hierarchical sparse expert allocation.
7. **Result preview**
   Give the seen-target result and the high-level pattern of where gains are strongest.

### Abstract Tone Rules

- Use strong but defensible verbs: **observe**, **motivate**, **operationalize**, **show**, **improve**.
- Avoid overclaiming verbs such as **prove**, **establish**, **guarantee**.
- Keep numeric detail light. One aggregate comparison is enough.
- If one interpretive clause is included, make it qualitative rather than score-based.

### One-Sentence Abstract Core

If the abstract loses focus, reduce it to this core:

> We bring the MoE routing-design question to sequential recommendation and answer it by aligning expert allocation with recurring behavioral axes visible in sessionized interaction logs.

---

## 4.6 Introduction Hook Strategy

The introduction should open with a sharper hook than the old L1-to-L5 progression.

### Recommended Opening Move

Open with the routing-design question, not with a feature list and not with a method summary.

Good opening idea:

> In MoE models, the value of routing depends not only on sparsity, but on what the router is aligned to. This raises a concrete question in sequential recommendation: which recurring behavioral axes should define expert allocation?

This hook is useful because it immediately signals that the paper is about a modeling question, not about attaching MoE to seqrec as a module.

### After the Hook

Then move through these roles:

1. sequential recommendation already contains recurring structural variation, and a sessionized view makes it easier to read
2. prior seqrec mostly improves representations under shared computation
3. prior MoE work aligns routing with the source of heterogeneity
4. seqrec therefore needs its own routing axes
5. RouteRec operationalizes those axes
6. experiments later show the expected gain pattern

### What The Hook Must Avoid

- sounding like a survey opener
- sounding like a method contribution sentence
- sounding like "we found four features"
- sounding like seqrec universally requires different paths

---

## 5. Related Work Structure

Keep three subsections, but change the logic.

### 5.1 Session-based and Sequential Recommendation

Purpose:

- establish the setting
- show that short-session modeling and sequence modeling are mature
- highlight that these models still use shared computation

End this subsection with the gap:

> prior seqrec improves representation quality, but not expert allocation

### 5.2 Behavioral Context in Sequential Recommendation

Purpose:

- cover time-aware, cross-session, feature-aware, and intent-aware modeling
- show that behavioral variation is already recognized
- show that it is used mainly to enrich hidden representation

End this subsection with the gap:

> prior work uses behavioral evidence to improve representations, not to define router inputs for conditional computation

### 5.3 Mixture of Experts and Routing Alignment

Split the discussion conceptually into two classes even if it stays in one subsection:

1. **heterogeneity-aligned routing**
2. **router-mechanics-oriented MoE**

Heterogeneity-aligned routing should carry more weight because it supports the paper's logic.

Use examples:

- Local experts / HME
- MMoE
- PLE
- AdaSparse
- PEPNet
- HM4SR
- M3SRec
- FAME

Router-mechanics work such as Switch, DSelect-k, Expert Choice, and V-MoE can appear briefly. Their role is contrast:

> they improve how sparse routing is executed, but they do not answer what the router should read in sessionized seqrec

---

## 6. Problem Section Responsibilities

The paper benefits from keeping a dedicated problem or design-challenge section between related work and method.

Recommended jobs for that section:

1. define sessionized next-item prediction
2. define available log fields
3. define the two-path view at a high level
4. formalize the three design questions

### Keep This Section Lean

Do not over-expand architecture details here.

This section should set up:

- what information is available from ordinary logs
- why router design is nontrivial
- why the three challenges are the right abstraction

### Recommended Design Questions

- **C1. Axis-to-cue construction**
  How can recurring routing-relevant axes be operationalized from ordinary logs without dataset-specific engineering?

- **C2. Scope alignment**
  At which temporal scopes should routing decisions be made so that each type of behavioral evidence is read where it is actually reliable?

- **C3. Route commitment**
  How should expert allocation be structured so that behaviorally different sessions actually activate different expert subsets?

The old challenge statements can remain close to the current ones, but the framing should begin one level earlier from routing-relevant axes.

---

## 7. Method Structure

The method should now read as the answer to the motivation, not as the source of it.

### 7.1 Method Overview

Open with a sentence close to this:

> RouteRec does not start from an arbitrary feature bank. It starts from recurring sources of routing demand visible in raw session logs and instantiates them as lightweight behavioral cues for expert allocation.

Then explain the two-path design and name the three components:

- Behavioral Cue Construction
- Coarse-to-Fine Multi-Stage Routing
- Hierarchical Sparse Expert Allocation

### 7.2 Behavioral Cue Construction

This subsection should explicitly bridge:

- recurring behavioral axes
- operational cue families

Adding a small mapping table is strongly recommended:

| Raw-log axis | Operational cue family | Example signal |
|---|---|---|
| Tempo / regime diversity | Tempo | gap statistics, pace trend, valid-prefix ratio |
| Transition ambiguity | Focus | switch rate, concentration, suffix entropy |
| Memory regime | Memory | repeat rate, recurrence, carryover |
| Exposure regime | Exposure | popularity level, drift, concentration |
| Context availability | routing support | history validity, repeated-session support |

Context availability can be described in text if a fifth table row feels too close to a cue family. The wording must make the distinction explicit.

### 7.3 Multi-Stage Routing

This subsection should explain why scope separation follows the evidence:

- macro reads reusable cross-session structure
- mid reads evolving within-session behavior
- micro reads short-horizon local transitions

Avoid writing this as a purely architectural convenience. It is part of the paper's logic.

### 7.4 Hierarchical Sparse Expert Allocation

This subsection should connect directly to route commitment.

Use this idea:

> if routing spreads weight too broadly, the model falls back toward a soft shared mixture and the value of axis-aligned routing disappears

---

## 8. Experiment Storyline

Experiments should support the framing, not replace it.

### Main Reading of Results

Preferred summary sentence:

> RouteRec is most effective where local transition ambiguity is pronounced and repeated-session context is sufficiently available, while its margin narrows when routing headroom is limited by context scarcity or by strong shared-path suitability.

### Dataset Interpretation Guide

- **KuaiRec**: strong branching and strong context support; large routing headroom
- **LastFM**: strong repeated-session context and memory-related variation
- **Foursquare**: high volatility and useful context; gains can appear more in broader ranking quality than in earliest-hit precision
- **Beauty**: some behavioral variation but weaker context availability
- **ML-1M**: context exists, but strong shared-path encoders already fit much of the dominant regime

Keep these as interpretation aids, not as absolute claims.

### What Stays Out of the Main Motivation

Move the following to appendix-oriented discussion:

- composite score formulas
- rank correlation numbers
- full dataset descriptor table
- rejected auxiliary metric rationale

Use one rule:

> the main text should be axis-centered; appendix material can be score-centered

---

## 9. Readability Enforcement

These rules matter enough to repeat here.

### Paragraph-Level Tests

Reject a paragraph if:

- the reader cannot tell in one pass what the paragraph's main point is
- a sentence contains two different technical moves
- the paragraph mixes motivation, method, and experiment interpretation without transitions
- the paragraph sounds translated rather than argued

### Common Failure Patterns To Remove

- opening with an abstract noun instead of a concrete subject
- multiple "which" clauses in one sentence
- sentence-final citation pile after the main verb
- elegant variation that hides term consistency
- over-compressed contrast such as "while," "whereas," and "with" used to carry too much logic

### Recommended Writing Mode

When drafting, prefer:

1. short sentence
2. explicit transition
3. one defined term
4. one claim

Then compress only if the paragraph still reads clearly.

---

## 10. Suggested Boilerplate Sentences

These are good anchors for future writing.

### Motivation Anchor

> Existing seqrec work mostly uses behavioral evidence to improve what the model represents. MoE raises a different question: which behavioral axis should define expert allocation once conditional computation is available?

### Axis Anchor

> We do not begin from an arbitrary bank of routing features. Instead, we inspect raw session logs and identify a small set of recurring sources of behavioral routing demand.

### Method Anchor

> RouteRec operationalizes these axes as lightweight behavioral cues that stay separate from the hidden-state representation path.

### Result Anchor

> The clearest gains appear where local branching is strong and repeated-session context is sufficiently available.

---

## 11. Final Sanity Check

Before accepting any new section draft, verify all of the following:

- [ ] The text is built around the routing-design question, not a generic MoE claim
- [ ] Motivation stops at recurring behavioral axes and does not collapse into cue details too early
- [ ] Cue families appear as operationalizations of the axes
- [ ] Context availability is treated as support, not as a fifth cue family
- [ ] Related work emphasizes heterogeneity-aligned routing more than router mechanics
- [ ] Experiment interpretation uses routing headroom and context availability carefully
- [ ] The English reads cleanly enough for a non-native but paper-experienced reader
