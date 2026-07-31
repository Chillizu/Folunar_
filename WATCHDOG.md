# WATCHDOG.md — PEDA Paper Claim Validator

This file provides project-specific guidance for the oh-my-pi advisor.
The advisor reviews the primary agent's work each turn and may inject
advice at severity levels: blocker, concern, or nit.

The PEDA experimental phase is CLOSED. The project is writing a
negative-result paper. Ground truth is `PEDA_FINAL/PEDA_CONCLUSION.md`
(2026-07-31, FINAL): all three charter sub-questions answered **No**
under tested conditions; the one statistically significant result
(Phase 3, N=20, p=0.0043) is attributable to candidate-set engineering
and success caching, not epistemic prediction error.

**Every rule below is a mechanical check, not a slogan.** The advisor
runs the stated command/pass and reports what it finds. Organization is
by WORKFLOW STAGE — the 7/26 manuscript failure was writing running
ahead of data; a flat rule list cannot catch that pattern.

---

## Stage 1: Data Collection — Pre-Writing Gates

Runs BEFORE any paper sentence is drafted. All checks target results/
files and the git tree, not prose.

### 1.1 A1 — Evidence bundle required before drafting (Blocker)

**Check**: For every quantitative claim planned, a bundle
`(source_file:line, verbatim_quote)` must already exist. Example bundle
that must exist before the number may be drafted:
`(results/phase8_gpu_run_2026-07-31.md:32, "28/45 (62.2%)")`.
No number without a bundle. No drafting until the bundle table is
complete — the advisor blocks any first sentence of a Results section
if its numbers lack bundles.

### 1.2 A2 — No transcript-only numbers (Blocker)

**Check**: Every result number must exist in a persisted file under
`results/` or `PEDA_FINAL/`. Mechanical test: for each number in the
draft, `grep -rn "<number>" results/` must hit a file. A number that
lives only in this chat gets a Blocker: "write it to results/<file> and
commit before the paper may cite it."

### 1.3 Source-file completeness gate (Blocker)

**Check** before drafting starts:
- `PEDA_FINAL/PEDA_CONCLUSION.md` exists and is committed.
- `results/phase8_gpu_run_2026-07-31.md` exists and is committed.
- `results/phase*/` for all phases 1-8 exist and are committed.
- `git status` is clean for these paths.

Any source file missing, uncommitted, or modified after the experiment
it records → Blocker: drafting may not begin.

### 1.4 D1 — Train/test split documented at collection time (Concern)

**Check**: Each results/ file recording model accuracy must state
in-distribution vs held-out (e.g. Phase 2 row: "train v1, eval v1" is
in-distribution; "train v1, eval v2" is held-out). If a results file
lacks the label, flag it. In-distribution-only results MUST be labeled
"memorization, not generalization" in the paper (Phase 2's L1=1.000
pass was in-distribution only, per conclusion).

### 1.5 D3 — Noise dimensions excluded from training targets (Concern)

**Check**: Inspect training-target definitions in `scripts/phase*_*.py`
and the WM code. Targets must be structural deltas (cwd_changed,
new_cwd, exit, files_created) — NOT timestamps, PIDs, or raw stdout/file
contents. If noise dimensions are in the target, the paper MUST list
them as a known limitation (conclusion root cause #1).

### 1.6 D4 — Hardware/environment recorded per experiment (Concern)

**Check**: Every results/ file must record model, GPU/CPU, Docker image
tag, sandbox version, commit hash. The bar:
`results/phase8_gpu_run_2026-07-31.md:7-10` (Model, Docker images,
Code, Commit). Missing fields → flag; the paper may not state hardware
it cannot source from a results file.

### 1.7 F1/F2 — Statistical metadata recorded with results (Concern)

**Check**: Any p-value in results files must carry test name + sample
size + effect size, e.g. "Mann-Whitney, N=20, d=-1.01"
(conclusion:48). If a results file tests many hypotheses, it must state
a correction method or "uncorrected — risk acknowledged". Missing
metadata → flag; the paper may not print bare "p<0.05".

**Stage 1 gate**: NO drafting until every claim's source file is
committed. Writing that runs ahead of data is the 7/26 failure mode —
the advisor stops it.

---

## Stage 2: Writing — Claim Discipline

Checks applied to the draft, per section, as sections are written.

### 2.1 A1 — Every number cites a bundle (Blocker)

**Check**: Every number, %, p-value in the draft is followed by
`(source_file:line, verbatim_quote)`. Mechanical test: pick any number,
`grep -n "<number>" <source_file>` — the quote must match verbatim
(same digits, same rounding: 62.2 not 62.22). No bundle = claim
rejected; rewrite or delete the sentence.

### 2.2 A3 — Pilot vs confirmatory language (Blocker)

**Check**: For each experimental claim, look up N in its bundle.
N≤3/condition → only "directional signal". N≥10/condition → may use
"validated". Any "validated/confirmed/proven" with N<10 → Blocker.
Retroactive upgrades (paper says N=20 where the run log says N=5) →
Blocker, and flag as fabrication (inherits old B2).

### 2.3 A4 — All conditions reported (Blocker)

**Check**: Diff the condition set in the source results file against
the paper's method/table. If the source has 4 conditions and the paper
reports 2, the omitted 2 must be explicitly disclosed with their
results — hiding unfavorable conditions = rejection. Mechanical:
`grep` each condition key from the results file in the draft.

### 2.4 A5 — Metric field verification (Blocker)

**Check**: For every metric name in the draft (success rate, fht, L1/L2/
L3, DLR, SCR), open the file that DEFINES it and read the definition
line. Do not trust field names. If the field is constant-true (the
`success = SCR > 0` tautology from `phase3_sandbox_experiment.py:132`,
always true when the agent visits ≥2 dirs), flag as suspicious and the
paper must report the real metric (`fht >= 0`), not the tautology.

### 2.5 B3 — Self-references use relative paths (Blocker)

**Check**: Internal citations must be project-root-relative
(`results/phase8_gpu_run_2026-07-31.md`), never absolute
(`/home/chillizu/...`) and never bare filenames. Mechanical:
`grep -n "/home/" <draft>` and check bare-name citations resolve.

### 2.6 E1 — No hedging that contradicts data (Blocker)

**Check**: Scan the draft for banned phrases: "promising", "may work
with larger model", "future work will resolve", "shows potential".
Cross-check each hit against its bundle: if the source says FAIL
("JEPA toggle adds zero delta", phase8:62), the phrase is banned. The
17 JEPA experiments are not "promising" — they are "flat uncertainty at
~37x the cost of counting" (conclusion root cause #3). Hedge phrase
with a FAIL source = Blocker.

### 2.7 E2 — The one positive carries its caveat (Blocker)

**Check**: Every occurrence of the Phase 3 N=20 p=0.0043 result must,
within ±1 sentence, state the attribution: candidate-set engineering
(NovellyExplorer's candidate set contained `cat hello.txt`) + success
cache, NOT epistemic prediction error. Mechanical:
`grep -n "0.0043" <draft>` — each hit must be adjacent to "candidate"
or "cache". Bare mention without caveat = Blocker.

### 2.8 E3 — Unambiguous declaration (Blocker)

**Check**: The paper must contain a sentence equivalent to "The PEDA
hypothesis is DISPROVEN under tested conditions", matching the
conclusion's Declaration (conclusion:136). Mechanical:
`grep -n "DISPROVEN" <draft>` — zero hits = Blocker.

### 2.9 D2 — Measurement method stated (Concern)

**Check**: Each metric's computation must be stated in the paper as a
code reference (`scripts/phaseX_*.py:NN`) or pseudocode. Two metrics
claiming to measure the same thing must agree (fht vs the success
tautology do NOT — the paper must explain which is used where).
Number without a method reference → flag.

### 2.10 C3 — Abstract post-written, then verified (Blocker)

**Check**: Abstract section mtime must be later than the last
Results/Discussion edit. Abstract drafted first → Blocker. After the
abstract exists, every abstract number must appear in the body WITH its
bundle — run the 2.1 check against the abstract specifically.

### 2.11 Manuscript reuse restriction (Blocker)

**Check**: `PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md` is SUPERSEDED — its
results/conclusions contain disproven claims. The paper may reuse ONLY
its Theory (Section 2) and Architecture (Section 3) sections. Any
result number or conclusion sentence copied from the manuscript →
Blocker. Mechanical: diff manuscript-sourced paragraphs against
PEDA_CONCLUSION.md.

---

## Stage 3: Review — Consistency Verification

Applied after the full draft exists.

### 3.1 C1 — No contradiction with PEDA_CONCLUSION.md (Blocker)

**Check**: For each draft claim, look up the matching Phase row in the
conclusion's evidence table and its verdict. Conclusion says FAIL, paper
says "positive signal" → Blocker. Mechanical: sentence-by-sentence
polarity scan ("success"/"fail", "validated"/"no effect") against
conclusion:28-60.

### 3.2 C2 — Same number everywhere (Blocker)

**Check**: Pick each headline number (62.2%, 0.0043, d=-1.01, DLR
~0.996, L1=1.000) and grep the whole draft. Abstract, Results table,
and Discussion must show IDENTICAL digits; rounding drift (62.2 vs
62.22) = Blocker. Mechanical: `grep -n "62.2" <draft>` — all hits
identical.

### 3.3 B1 — Every citation resolvable (Blocker)

**Check**: Every `[Author, Year]` in the reference list must resolve to
a real publication: authors, venue, year, title all match the actual
record. Inherited citations from the 7/26 manuscript MUST be
re-verified, especially:
- DreamerV4 `[Hafner et al., 2025]` — "Training agents inside of
  scalable world models" (arXiv preprint).
- V-JEPA 2 `[Bhardwaj et al., 2025]` — "Internet video and robot data
  for zero-shot control" (arXiv preprint).

Mechanical: verify each against the publication record (web search /
publisher page). Authors+venue+year+title must match; mismatch =
Blocker.

### 3.4 B2 — No fabricated citations (Blocker)

**Check**: Any reference that cannot be located (no matching
publication record) is DELETED from the paper — never "fixed" by
inventing details. A citation with no verifiable record = paper
rejection-level defect. "Hallucinated" citation = Blocker; do not
include it.

### 3.5 C4 — Conclusion reverse-scan (Blocker, final gate)

**Check**: Reverse-scan the paper with PEDA_CONCLUSION.md as key.
Every factual sentence in the paper must either appear in the
conclusion, or be independently sourceable to a results/ file bundle.
Mechanical: sentence-by-sentence pass; any factual sentence with no
bundle and no conclusion match = Blocker. The paper does not ship until
this pass completes clean.

---

## Advisor Self-Governance (Nit)

### G1 — No repeated advice
Same issue → same advice once. One advisory per concrete issue per turn.

### G2 — Park after task complete
After the current stage's checks pass (or blockers are emitted), park.
No lingering.

---

## Superseded Rules (experimental-phase WATCHDOG — no longer enforced)

- **B1** phase advancement — moot, project concluding.
- **B3** module review gate — moot.
- **B4** document inflation — moot.
- **B7** env-model mismatch — documented in the conclusion.
- **B8** death spiral — moot.
- **C1-C8, C10, C12-C22** process rules — experimental phase over.
- **N1-N3** process nits — moot.
- **"3 Questions" framework** — was for experiments, not paper.
- **Rule changelog** — historical.

## Inheritance Map

| Old rule | New rule |
|----------|----------|
| B2 fabrication | A1 (2.1), A2 (1.2) |
| B5 sample size | A3 (2.2), A4 (2.3), F1 (1.7) |
| B6 cherry-picking | A4 (2.3), E2 (2.7) |
| B9 anomaly investigation | C1 (3.1), C4 (3.5) |
| C9 train/test split | D1 (1.4) |
| C11 measurement consistency | D2 (2.9) |
| C23 metric field meaning | A5 (2.4) |
| C24 noise in training | D3 (1.5) |
