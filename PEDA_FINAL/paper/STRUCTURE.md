# PEDA Final Manuscript — Structure & Contracts

## Paper: "Prediction-Error-Driven Exploration in LLM Agents: A Negative Result"

### Section Map

| Section | Source | Writer | Lines |
|---------|--------|--------|-------|
| Abstract | Written LAST, verified against all sections | Main | ~20 |
| 1. Introduction | Charter Q1-Q3, motivation | WriterA | ~60 |
| 2. Theoretical Framework | Old manuscript §2 (FEP, EFE, Active Inference) | WriterA | ~80 |
| 3. Architecture | Old manuscript §3 (7 modules, L1/L2/L3) | WriterA | ~80 |
| 4. Experimental Setup | All 11 evidence bundles: env, model, metrics | WriterB | ~60 |
| 5. Results | CLAIMS_VS_EVIDENCE.md — REPRODUCIBLE only | Diverse | |
| 5.1 Phase 1-2 | evidence/phase1.md, phase2.md | WriterC | ~80 |
| 5.2 Phase 3-4 | evidence/phase3.md, phase4.md | WriterD | ~80 |
| 5.3 Phase 5-7 | evidence/phase5.md, phase6.md, phase7.md | WriterE | ~100 |
| 5.4 Phase 8 | evidence/phase8.md | WriterF | ~60 |
| 6. Discussion + Root Causes | PEDA_CONCLUSION.md §5, CLAIMS_VS_EVIDENCE | WriterG | ~100 |
| 7. Conclusion | PEDA_CONCLUSION.md §8 Declaration | WriterG | ~40 |
| References | Verified citations only | Main | ~30 |

### Writing Rules (WATCHDOG Stage 2)
1. EVERY number: (source_file:line, quote_bundle)
2. Abstract: written AFTER all sections complete
3. CLAIMS_VS_EVIDENCE.md: only REPRODUCIBLE + PARTIAL(caveated) claims allowed
4. MISSING claims: excluded
5. CONFLICT claims: resolved before use
6. Hedge ban: no "promising," "may work," "future work will resolve"
7. Phase 3 positive: MUST carry candidate-engineering caveat every time
8. Old manuscript: Theory (§2) + Architecture (§3) ONLY

### Evidence Bundles (READ BEFORE WRITING)
- PEDA_FINAL/paper/evidence/theory.md (E01-E04 context)
- PEDA_FINAL/paper/evidence/phase1.md (E01-E03)
- PEDA_FINAL/paper/evidence/phase2.md (E05-E07)
- PEDA_FINAL/paper/evidence/phase3.md (E08-E09)
- PEDA_FINAL/paper/evidence/phase4.md (E10-E12)
- PEDA_FINAL/paper/evidence/phase5.md (E13-E14)
- PEDA_FINAL/paper/evidence/phase6.md (E15-E16)
- PEDA_FINAL/paper/evidence/phase7.md (E17)
- PEDA_FINAL/paper/evidence/phase8.md (E18-E19)
- PEDA_FINAL/paper/evidence/worklog.md (timeline, decisions)
- PEDA_FINAL/paper/CLAIMS_VS_EVIDENCE.md (canonical cross-reference)
- PEDA_FINAL/PEDA_RESEARCH_MANUSCRIPT.md (old draft — §2 Theory, §3 Architecture only)
- PEDA_FINAL/PEDA_CONCLUSION.md (structural guidance)

### Output path
PEDA_FINAL/paper/PEDA_FINAL_MANUSCRIPT.md
