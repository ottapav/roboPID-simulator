ECC 2027 -- LIMIT CYCLES OF MODEL-FREE PID TUNING BY STEP-RESPONSE INSPECTION
============================================================================

conference_draft_ecc.tex / .pdf   the draft (ieeeconf; compile with the
                                  ieeeconf.cls of ../ECC27_certified_SPIN)
fig_index.pdf                     Fig. 1
lc_results.json                   every number in the draft
scripts/lc_setup.py               the setting: linear loop, derivative on the
                                  error, app defaults, core.tuning unmodified
scripts/lc_results.py             regenerates lc_results.json and fig_index.pdf
                                  (python docs/ECC27_limit_cycle/scripts/lc_results.py,
                                  about 10 minutes)
ORBIT_UNIQUENESS_NOTES.md         the investigation that led to the current
                                  draft (earlier settings; superseded where it
                                  disagrees with lc_results.json)

The draft relies on two corrections in the reference code: a blocked cut
passes to a lower band (core/tuning.py), and the winding number is the sum of
segment angles (core/features.py). The continuous count of Sec. V is
docs/scripts/regindex_soft.py.

The other .tex files and figures in this folder belong to earlier versions of
the draft and are not used.
