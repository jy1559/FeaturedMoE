# RouteRec Writing Improvement Notes

Scope: review of `writing/ACM_template/sample-sigconf.tex` from the abstract through `Problem Setting and Design Challenges` (`\begin{abstract}` to the end of Section 3), using `writing/WRITING_CORE.md` and `writing/WRITING_DETAIL.md` as the primary standards.

Also checked for consistency against the current experiment framing:
- `experiments/run/fmoe_n4/movielens_v4_session_fixed_portfolio_2_summary.md`
- `experiments/run/baseline_2/docs/baseline_2_best_valid_test_tables_v4.md`

The goal of this memo is not to rewrite the paper directly. It is to identify where the current front matter is already working, where it violates the writing rules, where a first-time reviewer would slow down or lose confidence, and how to revise it so the prose reads like careful human academic writing rather than polished-but-generic LLM output.

## Overall judgment

The current front matter is already much stronger than a typical rough draft. The high-level story is coherent:
- short sessions make behavioral differences more visible,
- most prior work uses that information to enrich representation,
- MoE turns the routing input into a real design choice,
- RouteRec answers that choice with explicit behavioral cues and a separate routing control path,
- the paper then decomposes the method into three design tensions.

That backbone works. A reviewer can follow the intended argument.

The main problem is not logical collapse. The main problem is density and rhetorical over-control. The text often says the same idea in slightly different forms, explains too many things before the reader has asked for them, and sometimes uses defensive or carefully balanced phrasing where a simpler statement would be stronger. That is exactly the kind of thing that makes writing feel LLM-generated: every sentence is polished, but the paragraph does not feel selective.

In short:
- the paper has a solid story,
- the opening sections are too eager to fully explain that story all at once,
- `Introduction` currently does too much work that should be left to `Problem Setting and Design Challenges`,
- several local wording choices violate `WRITING_CORE.md`,
- some claims are plausible but not yet phrased in the most reviewer-proof way.

## Best parts of the current draft

These are worth preserving during revision.

### 1. The opening problem frame is clear

The first paragraph of the introduction is effective. It establishes the setting early, names the kind of behavioral variation the paper cares about, and keeps the discussion tied to observable log structure rather than vague personalization language. That is aligned with `WRITING_DETAIL.md` Section 1.

### 2. The paper mostly respects the core conceptual separation

The distinction between the sequential backbone and the routing control path is stated clearly and repeatedly enough that a reader will not confuse RouteRec with feature enrichment or multi-task MoE. This is the most important conceptual burden of the paper, and the current draft does carry it.

### 3. The problem-to-method bridge is visible

The move from motivation to three design tensions is understandable. Even where the prose is redundant, the reader is unlikely to miss what C1, C2, and C3 are supposed to be.

### 4. The paper is already trying to avoid exaggerated interpretability claims

The text repeatedly treats inspectability as a benefit of the design separation rather than the main thesis. That is the right instinct and should be kept.

## Main weaknesses

### 1. `Introduction` is overloaded

The introduction currently performs all of the following:
- setting definition,
- behavioral observation,
- prior-work taxonomy,
- design question,
- MoE introduction,
- critique of hidden-state routing,
- two-path RouteRec explanation,
- cue-family definition,
- temporal-scope motivation,
- three design tensions,
- partial method preview,
- contributions,
- main result summary.

That is too much. A human reviewer can follow it, but it does not feel selective. It feels like the paper is trying to pre-answer every possible confusion before the formal sections begin. This reduces impact because the key claims do not land once; they arrive repeatedly in smaller variations.

### 2. The same idea is often restated instead of advanced

Several ideas recur too closely:
- "representation enrichment versus computation-path selection"
- "hidden states are optimized for prediction, not behavioral regime"
- "representation path versus routing control path"
- "behavioral cues are observable and therefore more inspectable"
- "three tensions / three components"

None of these repetitions are individually wrong. The issue is accumulation. The reader starts to feel that the text is circling the point rather than moving through it.

### 3. The prose is sometimes too self-protective

Examples of this tone:
- "We do not treat a different computation path as a necessary requirement."
- "an advantage of the separation, not an independent design claim"
- "rather than simply adding a routing step that leaves behavior largely unchanged"

This kind of hedging is intellectually careful, but too much of it creates drag. It reads as if the text is constantly defending itself against anticipated objections. Stronger writing usually states the design question plainly, makes the design choice plainly, and only qualifies where the qualification materially changes the claim.

### 4. Some sentence-level choices still read like generated academic prose

Common symptoms in this draft:
- highly symmetrical sentence construction,
- repeated contrast templates such as "not X but Y" and "rather than A, B",
- long sentences that contain both the core claim and its justification,
- nouns like "tension," "control view," "structural character," and "meaningful computation differences" appearing so often that they lose sharpness,
- frequent abstract meta-language instead of plain concrete statements.

The content is not empty. But the sentence texture sometimes feels over-smoothed.

## CORE rule compliance check

### Clear violations that should be fixed

1. `sample-sigconf.tex` uses `feature vector` in the method section.
- Current text: "maps each window to a low-dimensional feature vector"
- Required by CORE: `cue vector`

2. `sample-sigconf.tex` uses `semantic families`.
- Current text: "grouped into four semantic families"
- Required by CORE: `cue families`

3. `sample-sigconf.tex` uses `gating` in method wording.
- Current text: "To stabilize gating numerics"
- Required by CORE: `routing`

4. `sample-sigconf.tex` uses vague qualifiers that CORE explicitly warns against.
- `often` appears in figure description text.
- `naturally` appears later in the experiments section.
- `may` appears in the setup text.

Only the first three are inside the requested front-section scope plus nearby method spillover, but they are important because they show the draft is not yet fully rule-disciplined.

### Borderline issues, not outright violations

1. "four families" in the introduction is understandable, but `WRITING_CORE.md` prefers `cue families` whenever ambiguity is possible. In the current introduction, saying "four cue families" would be cleaner.

2. "structural character" is allowed by `WRITING_DETAIL.md`, but the paper uses it often enough that it risks sounding like a catch-all abstraction. It should appear when it adds meaning, not as a default synonym for session pattern.

3. "control view" is not forbidden, but it is not canonical terminology either. Use it sparingly. When possible, state exactly what differs: temporal scope, cue vector, or routing frequency.

## Section-by-section review

## Abstract

### What works

- The abstract starts with the correct setting: sessionized sequential recommendation with short sessions and available cross-session history.
- It frames the paper around a design question rather than a universal claim that different behavior must imply different computation paths. That is strongly aligned with `WRITING_DETAIL.md`.
- It states the central design choice clearly: route using behavioral cues computed from logs and keep routing control separate from the sequential backbone.

### What weakens it

1. It is too dense for an abstract.

The abstract currently tries to carry motivation, design choice, criticism of hidden-state routing, mechanism, temporal scopes, sparsity structure, result summary, and interpretability-style benefit in a very compact block. That makes it informative, but not maximally readable.

2. The result sentence is not as reviewer-friendly as it could be.

"best overall Avg.~Rank of 1.67, versus 3.56" is precise, but Avg. Rank is not intuitive on first contact. A reviewer reading only the abstract does not yet know the table structure or selection rule. The sentence would land better if it first stated the practical result in plain language and then optionally added the rank summary.

3. "diagnostics confirm more behavior-consistent expert selection" is too compressed.

That phrase asks the reader to accept a qualitative conclusion without any hint of what the diagnostic actually shows. In an abstract, that can feel slightly promotional.

### Suggested revision direction

- Keep the first sentence pair almost as-is.
- Shorten the hidden-state critique to one sentence.
- Mention only one method detail after the central design choice: either three temporal scopes or hierarchical sparse expert allocation, not both in full detail.
- Rewrite the result line so the first clause is easy to parse without knowing the tables.

### A better abstract rhythm

Use four compact moves:
1. setting and gap,
2. RouteRec design choice,
3. one-sentence mechanism summary,
4. plain-language main result with seen-target qualifier and dataset names.

Right now the abstract has the right ingredients but too many of them appear at full length.

## Introduction

### What works

1. The first paragraph is strong.

It establishes the session-based setting early, explains why short sessions matter, and names the behavioral dimensions in a way that feels concrete rather than speculative.

2. The introduction mostly follows the intended L1 -> L2 -> L3 -> L4 arc from `WRITING_DETAIL.md`.

The text moves from observable session heterogeneity to the question of whether behavior should affect computation, then introduces MoE, then raises the routing-input problem, then presents RouteRec's answer.

3. The hidden-state critique is conceptually sound.

The claim is not that hidden-state routing is invalid. The claim is that once MoE is introduced, routing from hidden states is only one option and it entangles routing control with representation. That is the correct level of argument.

### What should be improved

#### 1. The prior-work paragraph is one step too long

The paragraph beginning from the figure explanation and prior-work review is not bad, but it tries to compress too many model families before landing the key question. A reviewer will get the point faster if that paragraph is shorter and more selective.

Current effect:
- the paragraph first catalogues time-aware, contrastive, frequency-aware, feature-aware, and cross-session models,
- then returns to the paper's main question.

This slows the transition into the design question. The reader does not yet need a mini-related-work paragraph at that moment. They need a sharp statement of what existing work does and what it leaves open.

Revision principle:
- keep at most three representative categories in the introduction,
- leave the fuller taxonomy to `Related Work`.

#### 2. The introduction repeats the RouteRec answer before the problem section formalizes it

The paragraph beginning "A separate routing control path avoids this confound" is clear, but it carries too much detail at once:
- two-path separation,
- behavioral cues,
- four cue families,
- inspectability benefit,
- three temporal scopes.

Then the next paragraph introduces the three tensions. Then the next paragraph partially resolves them again. This is exactly where the introduction starts to feel over-written.

A cleaner structure would be:
- one paragraph that states RouteRec's answer at a high level,
- one paragraph that says this answer raises three design questions,
- then stop.

The introduction should not partially solve the tensions in prose before `Problem Setting and Design Challenges` and `Method` have done their jobs.

#### 3. The contributions list overlaps too much with the preceding prose

The contributions are individually reasonable, but as a block they do not add enough new information after the introduction's last three paragraphs. Bullet 1 and Bullet 2 are especially close:
- Bullet 1 says routing input is a substantive design decision.
- Bullet 2 says RouteRec separates backbone and routing control via behavioral cues.

That is fine conceptually, but the reader has just been told this in near-identical language.

Options:
- keep the contributions but make them shorter and more differentiated, or
- shorten the preceding paragraphs so the contributions feel like a crisp recap rather than a re-recap.

#### 4. Some sentences carry more than one technical job

This conflicts with `WRITING_CORE.md`.

Example pattern in the introduction:
- first clause states a design choice,
- second clause explains why hidden states are insufficient,
- third clause adds an inspectability benefit.

Even when the sentence remains grammatical, the reader has to unpack multiple argumentative moves at once. That is where the prose starts to feel heavy.

#### 5. The introduction is slightly too explicit about its own caution

Phrases like "not an independent design claim" and "we do not treat ... as a necessary requirement" are intellectually respectable, but the current draft uses this style often enough that the writing loses force.

Use this rule:
- if a disclaimer changes how the claim should be interpreted, keep it,
- if it merely shows that the authors are being careful, consider deleting it.

### Concrete revision targets for the introduction

1. Cut about 15 to 20 percent of total introduction length.

2. Make each paragraph do only one rhetorical job:
- Paragraph 1: setting and behavioral observation.
- Paragraph 2: what existing approaches do and what they leave open.
- Paragraph 3: why MoE makes routing input a design question.
- Paragraph 4: RouteRec's answer at high level.
- Paragraph 5: three tensions.
- Contributions.

3. Remove at least one full layer of repetition around:
- hidden-state critique,
- two-path separation,
- inspectability,
- three-component preview.

4. Keep the introduction focused on motivation and design logic. Let Section 3 carry the explicit C1/C2/C3 burden.

## Related Work

### What works

- The three-subsection structure matches `WRITING_DETAIL.md` well.
- Each subsection is readable and the gaps are generally well positioned.
- The paper does not repeatedly end every paragraph with "RouteRec does X instead," which is good.

### What weakens it

1. The first subsection is a bit too list-like.

The historical model list is accurate, but it is slightly longer than necessary for the paper's actual argumentative need. A reviewer in this area already knows the backbone families. The paragraph should serve the gap, not completeness.

2. The second subsection is strong in concept but still somewhat catalogue-heavy.

The central point is simple: prior work uses behavioral context to improve representation, not to define an explicit router input for conditional computation. That point should land earlier in the paragraph.

3. The tone occasionally becomes generic survey prose.

The paper is strongest when it makes a sharp comparative claim. It is weaker when it sounds like a broad literature summary.

### Suggested revision direction

- tighten each subsection by removing one or two examples that do not advance the gap,
- keep the closing gap sentence in each subsection,
- make sure the reader feels why this related work matters specifically for the routing-input question.

The related work is not the most urgent problem in the front matter. It is acceptable as-is, but it can be made faster and sharper.

## Problem Setting and Design Challenges

### What works

1. The problem setup is concise and concrete.

This section does what it should do: define sessions, prefix, target, available fields, and the practical constraints on what signals RouteRec uses.

2. The challenge statements are the cleanest part of the front matter.

Each of C1, C2, and C3 has a clear tension, a concrete failure mode, and a visible connection to the eventual method component. This section is already close to what the paper needs.

3. The section respects the design-question framing.

It does not overclaim that behaviorally different sessions inherently require different routes. It says that cue-driven routing must solve specific design tensions to matter.

### What weakens it

1. The introduction has already spent too much of this section's argumentative capital.

By the time the reader reaches C1/C2/C3, they have already seen:
- explicit route-control separation,
- macro/mid/micro scope motivation,
- sparse route commitment,
- a preview of the same three tensions.

That makes the problem section feel slightly less consequential than it should. The section itself is good; the issue is upstream redundancy.

2. The final transition sentence is a little mechanical.

"The following section addresses each tension ..." is serviceable, but because the paper already signaled this mapping multiple times, it lands like another piece of roadmap prose rather than a meaningful turn.

### Suggested revision direction

- keep the content of Section 3 mostly intact,
- move more of the explicit challenge burden here by trimming the corresponding preview material from the introduction,
- make the ending transition shorter and less procedural.

This section should feel like the point where the paper becomes formally structured, not the third time the reader hears the same roadmap.

## Reviewer-style concerns

These are the places where a careful reviewer might push back, even if the paper's logic is basically sound.

### 1. The introduction can feel over-argued

A reviewer may think: "I understand the point, but the paper is trying a bit too hard to choreograph my interpretation." That reaction often comes from repetition plus repeated micro-disclaimers, not from any one incorrect sentence.

### 2. Avg. Rank is convenient for summarization but not ideal for first-contact persuasion

This is not wrong, but the front matter currently foregrounds Avg. Rank quickly. A reviewer who has not yet seen the table may prefer a more direct summary of where RouteRec wins and where it is competitive.

### 3. The distinction between motivation and method preview is slightly blurred

The introduction names cue families and temporal scopes in a level of detail that begins to feel methodological. That weakens the crisp sectional separation requested in `WRITING_DETAIL.md`.

### 4. Some nouns risk sounding more precise than they are

Terms like "structural character," "control view," and "meaningful computation differences" are useful, but if repeated without enough concrete anchoring they can start to sound like polished abstractions rather than observations. This is a common LLM smell in technical writing.

## Most important revisions to make

If time is limited, do these first.

### Priority 1. Trim and de-duplicate the introduction

This is the single most important change. The paper's logic is good enough that trimming will improve it more than adding anything new.

Cut or compress:
- the mini-related-work inventory inside the introduction,
- one layer of hidden-state-routing critique,
- one layer of inspectability explanation,
- one of the two paragraphs that both preview and partially resolve the three tensions.

### Priority 2. Make Section 3 carry the formal problem burden

Let the introduction motivate. Let `Problem Setting and Design Challenges` define the design tensions explicitly. Right now the paper half-solves the problem before it formally states it.

### Priority 3. Fix canonical terminology violations everywhere

At minimum:
- `feature vector` -> `cue vector`
- `semantic families` -> `cue families`
- `gating numerics` -> `routing logits` or `routing stabilization`

Even if some of these are outside the exact requested scope, they matter because they weaken confidence that the paper is rule-disciplined.

### Priority 4. Replace defensive phrasing with cleaner assertions

Prefer:
- a plain design question,
- a plain design choice,
- a plain statement of the consequence.

Only keep caveats that materially change the claim.

### Priority 5. Reduce abstract nouns where a concrete phrase will do

Examples:
- instead of "meaningful computation differences," say "different expert subsets across sessions" if that is the actual point,
- instead of "control view," say "routing view at a given temporal scope" when scope is what matters,
- instead of repeatedly saying "structural character," occasionally say the actual pattern type: pace, repetition, category switching, or popularity drift.

## How to revise so it does not read like LLM writing

This matters because the current draft is already polished enough that the risk is no longer grammar. The risk is texture.

### 1. Make each paragraph more selective

A human writer usually leaves something unsaid until the next section. LLM prose often tries to make each paragraph self-sufficient. This draft sometimes has that second property.

When revising, ask:
- what is the one job of this paragraph?
- what can safely be left for the next section?

### 2. Cut mirrored contrast patterns

The draft often uses polished opposition structures:
- not X but Y,
- rather than A, B,
- what the model represents versus which computation path it applies.

These are useful, but overuse makes the prose feel templated. Keep the important version of each contrast once. Do not restate it three times with slightly different wording.

### 3. Use a few plainer sentences

Not every sentence should sound like it belongs in a polished abstract. Papers feel more human when some sentences are simply direct.

Examples of better texture:
- "This leaves the routing input underspecified."
- "Section 3 makes those choices explicit."
- "That detail belongs in the method, not here."

The current draft is sometimes too uniformly elevated.

### 4. Avoid generic balancing language

Phrases like:
- "carefully balances"
- "meaningfully differentiates"
- "semantically legible"
- "substantive design choice"

can be valid, but they often sound like polished placeholders unless followed immediately by a concrete explanation.

### 5. Prefer asymmetry over perfect rhythm

LLM writing often produces three equally weighted clauses, each with parallel grammar. Human academic prose is usually less symmetric because it gives more space to the most important point and less to the support point.

If three sentences in a row have the same length and structure, that is often a sign to revise.

### 6. Keep the strongest idea in the shortest sentence

The strongest idea in this paper is simple:
- RouteRec routes with explicit behavioral cues instead of hidden states.

That idea should appear at least once in very plain form. If every version is wrapped in multiple subordinate clauses, the paper loses punch.

## Recommended rewrite plan

### Step 1

Revise the abstract for compression, not for added nuance.

### Step 2

Rewrite the introduction paragraph-by-paragraph with a strict one-job rule.

### Step 3

Do a terminology sweep immediately after that rewrite.

### Step 4

Read only the front matter aloud from top to bottom and mark every point where you feel one of these reactions:
- "I already know this"
- "this sentence is doing too much"
- "this sounds polished but not sharp"
- "this belongs in the next section"

That pass will catch most remaining LLM-like residue.

## Bottom line

The front matter is conceptually solid and reviewer-readable, but it is not yet maximally persuasive. The paper's main issue is not that it is unclear what RouteRec does. The issue is that the writing explains the same core ideas too many times, too carefully, and sometimes with slightly over-abstract diction. Tightening the introduction, restoring cleaner section boundaries, and enforcing the canonical terminology more strictly would make the paper feel sharper, more confident, and much less like it was assembled by an LLM trying to be comprehensive.