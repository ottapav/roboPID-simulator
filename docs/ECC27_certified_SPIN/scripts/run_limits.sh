#!/bin/bash
# Certified SPIN (gain box on) with other count limits, cautious and bold start.
# Results: lim<tag>_0.jsonl (cautious), lim<tag>b_0.jsonl (bold). Runs two jobs at a time.
# The published limits (0.5, 0.75, 1.0) are cbox_0.jsonl and cbox2_0.jsonl.
cd "$(dirname "$0")"
run() { SPIN_BOX=1 RULES_ONLY=cert-shape CERT_LIMITS=$1 START_SCALE=$2 OUT_PREFIX=$3 python3 designed.py 0 1 > $3.log 2>&1; }
run 0.5,1.0,1.0 1 lim101 & run 0.5,1.0,1.0 2 lim101b & wait
run 0.5,0.5,0.5 1 lim050 & run 0.5,0.5,0.5 2 lim050b & wait
echo ALLDONE > run_limits.done
