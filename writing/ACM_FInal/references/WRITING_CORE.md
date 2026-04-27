# RouteRec Writing Core
# Give this file to the LLM at the start of every writing session.

These rules override general writing instincts. Prioritize consistency first, readability second, elegance third.

---

## What This Paper Claims

RouteRec is a session-based sequential recommendation model that introduces Mixture of Experts (MoE) into a SASRec-style backbone and asks a specific routing-design question:

> Once conditional computation is available, what should define expert allocation in sessionized sequential recommendation?

The paper's main claim is **not** that sequential recommendation universally requires radically different computation paths.

The paper's main claim **is** this:

> In sessionized sequential recommendation, expert allocation should be guided by routing signals aligned with recurring behavioral axes in interaction logs, rather than by prediction-optimized hidden states alone.

RouteRec operationalizes those axes as lightweight **behavioral cues** and uses them on a separate **routing control path**. The **sequential backbone** still handles next-item prediction. Hidden states do not drive routing.

---

## Priority Order

Every generated paragraph must satisfy this order:

1. **Terminology consistency**
2. **Readability for a Korean reader with moderate English proficiency**
3. **Precise technical meaning**
4. **Concision**
5. **Style**

If a sentence is technically correct but hard to parse, rewrite it.

---

## Canonical Terminology

Never introduce a new synonym for a defined term.

| Concept | Use | Never Use |
|---|---|---|
| Main framing of the paper | **behavioral routing demand** | "heterogeneity" alone, "behavioral diversity" |
| Repeated dataset-level sources of routing need | **recurring behavioral axes** | "feature axes", "behavioral dimensions" unless already defined |
| Whether repeated-session evidence exists strongly enough to support routing | **context availability** | "history richness", "context richness" |
| Whether routed models have room to improve over shared computation | **routing headroom** | "MoE suitability", "routing opportunity" |
| Router input derived from logs | **behavioral cues** | "behavioral features", "behavioral signals", "routing features", "routing signals" |
| Vector of cues for one window or scope | **cue vector** | "feature vector", "routing input" |
| Tempo / Focus / Memory / Exposure | **cue families** | "feature families", "cue groups", "feature groups" |
| One of the four families | **[Name] family** | "tempo group", "focus features" |
| Routing driven by hidden states | **hidden-state-driven routing** | "hidden-only gating", "latent routing" |
| Mechanism selecting experts | **routing** | "gating", "dispatching", "switching" |
| Activated experts for one input | **computation path** or **route** | "inference path", "execution path" |
| SASRec-style model path for prediction | **sequential backbone** | "encoder", "base model", "backbone model" |
| Separate path computing cues and selecting experts | **routing control path** | "control branch", "routing branch" |
| Macro, mid, micro collectively | **routing stages** | "routing layers", "routing levels", "routing blocks" |
| Cross-session history scope | **macro scope** | "long-term scope", "global scope" |
| Within-session prefix scope | **mid scope** | "session-level scope", "within-session level" |
| Short local window | **micro scope** | "short-term scope", "local level" |
| Group of experts aligned with one family | **expert group** | "expert cluster", "expert block" |
| Top-k_g families then top-k_e experts within them | **hierarchical sparse expert allocation** | "hierarchical routing", "two-level selection" |
| MoE general term | **Mixture of Experts (MoE)** first use; **MoE** after | "mixture model" |

### Fixed Distinctions

- **recurring behavioral axes** are the dataset-level or motivation-level descriptors.
- **behavioral cues** are the sample-level operationalization of those axes.
- **context availability** is a support condition, not a cue family.
- **behavioral routing demand** is the framing term for why routed conditional computation may help.

Do not collapse these four ideas into one term.

---

## Terms To Define On First Use

Define each term in a natural sentence before using it technically.

- **conditional computation**: executing only a data-dependent subset of model capacity rather than the full model for every input
- **routing**: a router selects which subset of experts processes a given input
- **computation path / route**: the specific expert subset activated for one prefix
- **behavioral routing demand**: the extent to which recurring behavioral variation creates a meaningful need for routed computation
- **recurring behavioral axes**: repeated structural axes in sessionized logs along which routing demand appears
- **behavioral cues**: lightweight scalar summaries derived from ordinary interaction logs
- **context availability**: whether reusable cross-session evidence is sufficiently available for macro-level routing
- **routing headroom**: the degree to which routed computation can improve on a strong shared-path model
- **route commitment**: routing that concentrates on a small, behavior-dependent expert subset rather than diffusing broadly

Introduce the four cue families together:

> four cue families: Tempo (pace and interval dynamics), Focus (concentration and switching), Memory (repetition and carryover), and Exposure (popularity level and drift)

---

## Readability Rules

This is the highest writing priority after terminology.

### Core Principle

Write so that a Korean researcher who reads English papers regularly can follow the paragraph without re-reading each sentence twice.

Keep the tone academically serious, but use simple words and clean sentence shapes whenever possible.

### Hard Rules

- Keep the grammatical subject clear. Do not hide the subject behind a long opening phrase.
- Prefer one main clause per sentence.
- If a sentence contains both a contrast and a justification, split it into two sentences.
- Avoid noun piles such as "behavioral routing control signal alignment question." Rewrite with verbs.
- Avoid vague pronouns such as "this," "these," or "they" when the referent could be ambiguous.
- Avoid "former/latter."
- Avoid long parenthetical interruptions.
- Put old information first and new information second when possible.
- End sentences on the key idea, not on a citation tail or a weak qualifier.
- If two adjacent sentences use the same structure, vary one of them.
- Keep the tone steady across the section. Do not mix plain spoken sentences with overly inflated academic phrasing.
- Prefer easier words when they preserve the same meaning. Simplicity is better than ornate wording.
- Use dashes sparingly. If a dash is necessary, write it as `---`.
- When defining terms, vary the sentence form. Do not define every term with the same pattern such as "`X` means ..." or "`X` refers to ...".

### Prefer

- short declarative sentences
- explicit causal links: "because", "therefore", "this leaves", "this raises"
- concrete nouns
- repeated canonical terms instead of elegant variation
- simple academic vocabulary over fancy synonyms
- smooth paragraph rhythm, with sentence lengths that vary slightly but remain easy to parse

### Avoid

- stacked subordinate clauses
- abstract filler: "in this regard", "from this perspective", "in such cases"
- vague emphasis: "clearly", "naturally", "indeed", "notably" unless actually needed
- inflated verbs: use "use", "show", "define", "align", "separate", "support"
- abrupt tone shifts
- overusing em-dash-style interruptions
- repeating the same definition template many times in a row

---

## Argument Rules

### Problem Before Solution

Each design paragraph must follow this order:

1. What is the problem?
2. Why do existing approaches leave it unresolved?
3. What is RouteRec's answer?

Do not open a paragraph with RouteRec unless the problem was established immediately before.

### One Claim Per Sentence

Do not combine two real claims with "and," "while," or a semicolon.

Bad:
"Hidden states are optimized for prediction and may hide behavioral regime, while cue-based routing improves interpretability."

Better:
"Hidden states are optimized for prediction. They do not explicitly expose which behavioral property triggered routing. Cue-based routing therefore gives the router a cleaner control signal."

### Treat the Main Question Correctly

Never write as if the paper proves:

> different behavior requires different computation paths

Write instead:

> once MoE introduces conditional computation, expert allocation becomes a modeling choice, and the key question is what should define that allocation

### Keep Motivation and Method Distinct

- **Motivation** should reach the axis-level question.
- **Method** should introduce cues as the operational answer.

It is acceptable for motivation and method to touch, but motivation must not read like a feature list.

---

## Motivation Rules

- Start from sessionized sequential recommendation, not MoE.
- Introduce MoE as a tool that creates a routing-design question.
- Use heterogeneity-aligned MoE literature to justify the question "what defines specialization?"
- In the introduction, emphasize **recurring behavioral axes**.
- In the method, emphasize **behavioral cues**.
- Treat interpretability as a supporting advantage of separating routing control from representation learning.
- Do not front-load dataset tables, composite scores, or correlation numbers in the introduction.

### Preferred Motivation Flow

1. Sessionized logs do not follow one uniform behavioral regime.
2. Prior seqrec mostly enriches representation under shared computation.
3. MoE changes the question from representation only to expert allocation.
4. Prior MoE work aligns routing with the source of heterogeneity.
5. In seqrec, raw logs suggest recurring behavioral axes that are router-relevant.
6. RouteRec operationalizes those axes as lightweight cues on a separate control path.

---

## Never Do

- Do not use "behavioral heterogeneity" as the main headline term. Use it only when needed and subordinate it to **behavioral routing demand**.
- Do not present cue families as if they were discovered first. The paper first identifies recurring axes, then operationalizes them as cues.
- Do not say "we start from a bank of routing features." Say the paper starts from recurring sources of routing demand in raw logs.
- Do not imply that cues are richer predictive side information.
- Do not oversell interpretability as a standalone paper claim.
- Do not blur dataset-level descriptors and sample-level router inputs.
- Do not call context availability a fifth cue family.
- Do not describe routing stages as layers.
- Do not say "hidden states are wrong." Say they are optimized for prediction and need not align with expert specialization.

---

## Experiment Language Rules

- Main result sentences must say **seen-target**.
- When claiming where RouteRec helps most, name the datasets explicitly.
- Use this pattern for interpretation:
  - strong gains where local branching is pronounced and repeated-session context is sufficiently available
  - smaller margins where context availability is limited or strong shared computation already matches the dominant regime
- Keep composite scores, directional correlation, and metric-selection rationale out of the main motivation. Put them in appendix-oriented discussion only.

---

## Final Checklist

- [ ] Canonical terms are used consistently
- [ ] "behavioral routing demand" and "behavioral cues" are not confused
- [ ] "recurring behavioral axes" appear before cue-family operationalization
- [ ] Every defined term is introduced in a readable sentence before technical use
- [ ] Sentences are short enough to parse on first read
- [ ] The tone stays uniform across the section
- [ ] Word choice is simple enough for a non-native reader without losing academic tone
- [ ] Dashes are used rarely, and only as `---` when needed
- [ ] Repeated term definitions do not all use the same sentence pattern
- [ ] No noun piles or stacked clause structures
- [ ] Paragraphs present problem before solution
- [ ] The paper's claim is framed as a routing-design question, not a universal claim about path diversity
- [ ] Main result sentences include the seen-target qualifier
