ECC 2027 CONFERENCE PAPER -- FINAL DRAFT
=========================================

Generated: 29 July 2026

FILES
=====

conference_draft_ecc.tex
  - Source LaTeX file
  - Format: ieeeconf class, twocolumn, 10pt
  - Ready for compilation with actual ieeeconf.cls
  - Status: Body content complete; frontmatter TODOs remain (see below)

conference_draft_ecc.pdf
  - Compiled PDF (4 pages)
  - Generated with ieeeconf shim for testing
  - Layout and page count finalized
  - All technical content verified against project handoff
  - Ready to review; figures are placeholder boxes (6 TODO-FIGs total)

OUTSTANDING BEFORE SUBMISSION
==============================

1. Author Block (front matter)
   - Line 22-26: Fill in Author names and affiliations
   - Line 23: Set corresponding author email (TODO-EMAIL)

2. Tables (data regeneration required)
   - Table 1 (Section 3): Duty-cycle measurements for P1, P3, P4
     Expected: values matching (0.125, 0.500, 0.250, 0.125) to 3 decimals
     Source: code/asymptotics/duty_cycle.py
   
   - Table 2 (Section 5): Final errors for five step-size schedules
     Clarify: per-plant or averaged across P1/P3/P4
     Source: code/asymptotics/schedule_comparison.py

3. Section 5 (Local Structure)
   - Jacobian determinants: (1.2, 5.9, 4.3) for P1, P3, P4
   - Clarify units/normalization in reproduce script

4. Figures (4 required; currently placeholder boxes in PDF)
   - Figure 1: Move geometry in log-gain space (e, d₀, d₁, d₂)
   - Figure 2: Period-8 terminal orbit around triple point (P1, β=0.03)
   - Figure 3: Discontinuity jump (hard count vs. regularized index)
   - Figure 4: Schedule convergence curves (log scale, 5 curves)

5. Paper 1 Citation
   - Line 247: Update citation once paper 1 has DOI or acceptance status
   - Currently placeholder: "submitted to J. Process Control, 2026"

COMPILATION INSTRUCTIONS
========================

With actual ieeeconf.cls:
  1. Obtain ieeeconf.cls from IEEE (usually included in conference author kit)
  2. Place ieeeconf.cls in the same directory as conference_draft_ecc.tex
  3. Run: pdflatex conference_draft_ecc.tex
  4. Run: pdflatex conference_draft_ecc.tex (again, for references)

Expected output: conference_draft_ecc.pdf (~4-6 pages, twocolumn)

CONTENT VERIFICATION
====================

All numbers from the project handoff are present and verified:
  ✓ Thresholds: (0.5, 0.75, 1.0)
  ✓ Duty-cycle weights: (1/8, 1/2, 1/4, 1/8)
  ✓ Discontinuity jump: 0.05% gain change → 1.000 index jump
  ✓ Regularized index change: 7×10⁻⁴
  ✓ Triple-point errors: P1 0.010, P3 0.017, P4 0.041
  ✓ Schedule errors: const 0.074, n⁻¹/² 0.010, n⁻³/⁴ 0.001, n⁻¹ 0.197, reversal 1e-4
  ✓ Reversal-halving: 36–98 iterations to 1e-4 accuracy
  ✓ P2 complementarity reframe: matches companion paper exactly

TERMINOLOGY (consistent with paper 1)
======================================
  ✓ "turn index" N_k: used throughout body
  ✓ "winding number": keywords only + proof-route discussion (§7)
  ✓ "Pachner plots": NOT mentioned (reserved for paper 1)
  ✓ "SPIN": introduced once in intro
  ✓ "triangular rule": used for move table

SPINE (from project proposal)
=============================
  §1  Introduction: Context, three open questions, sign-based vs. gradient methods
  §2  The Rule as Switched System: Log-gain coordinates, four moves, partition
  §3  The Target (Triple Point): Proposition 1 (unique balance), Corollary 1 (target),
      duty-cycle measurements, boundary case (complementarity)
  §4  The Obstruction (Discontinuous Count): 0.05% jump, consequence for asymptotics
  §5  The Fix (Regularized Index): Definition eq(2), three properties, convergence
  §6  Step Schedules & Limit Cycles: Table 2, constant step, diminishing steps,
      Lemma 1 (logarithmic travel), reversal-halving
  §7  Local Structure & Proof Route: Jacobian analysis, Filippov dynamics,
      per-band acceleration remark
  §8  Conclusions: Summary, future work (Filippov proof)

STYLE NOTES
===========
  - Plain-English register matching companion paper 1
  - Propositions/corollaries stated simply, proofs omitted or sketched inline
  - Lemma 1 is the only new formal result stated outside a theorem
  - Rprop connection: 1 sentence at end of §6
  - Momentum note: 1 sentence in §7
  - All future work deferred to conclusions (no apologies for gaps)

PAGE COUNT
==========
Tested with ieeeconf shim (complete.cls): 4 pages twocolumn
Compiled PDF shown: 4 pages
Expected with actual ieeeconf.cls: 4–6 pages (depending on exact class spacing)
With figures added: expect final version 5–6 pages (well within 6-page limit)

NEXT STEPS
==========
1. Regenerate TODO-REGEN tables using reproduce scripts in code/asymptotics/
2. Generate four figures (see OUTSTANDING section)
3. Fill in author block and paper 1 citation
4. Obtain actual ieeeconf.cls and do final compile
5. Verify page count ≤ 6 pages
6. Submit to ECC 2027 or target IEEE conference


=====================================================================
REVISION 29 July 2026 -- REFRAMED AROUND THE LIMIT CYCLE
=====================================================================

New title: "Limit Cycles of Model-Free PID Tuning by Step-Response
Inspection" (echoes paper 1's title; no "regularization", no "triple
point" -- both are tools, not the contribution).

What changed relative to the previous draft:
  - Abstract and introduction now lead with the limit cycle as the
    object of study; three organizing questions are where / what shape
    and size / can it be shrunk.
  - Old Sec. 3 "The Target: a Triple Point" -> "The Limit Cycle and Its
    Center". Corollary restated as "Center of the Cycle" and now states
    explicitly that theta* depends only on plant + thresholds, not on
    step size or history. Period-8 realization promoted to a numbered
    Observation (frequencies are theorem, realization is measurement).
  - Old Sec. 4 (obstruction) + Sec. 5 (fix) merged into one section
    "A Continuous Index for the Analysis": regularization explicitly
    framed as an instrument that makes the cycle's center well posed,
    not as a change to the method. Diminishing-step convergence is
    labeled a theoretical device.
  - New Sec. 5 "Size of the Cycle at Fixed Step": constant-beta error
    0.074 presented as the orbit radius and as the guarantee supporting
    the companion paper's constant-step recommendation. Power-law
    schedules kept but explicitly labeled analysis instruments.
  - New Sec. 6 "Accelerated Tuning: Shrinking the Cycle": reversal-
    halving gets its own section, derived from the cycle picture
    (travel vs. orbit regimes distinguishable from the move sequence),
    with an honest paragraph on which parts are theorem vs. measured.
  - Sec. 7 proof route reworded: goal is existence/stability/O(h)
    radius of the cycle via Filippov averaging.

Unchanged: all numbers, all TODO-REGEN markers, terminology, the four
figure slots, bibliography, ieeeconf twocolumn format.

Compiled: 4 pages twocolumn (shim), 0 overfull. Expect 5-6 pp with the
real ieeeconf.cls and the four figures -- within the ECC limit.
