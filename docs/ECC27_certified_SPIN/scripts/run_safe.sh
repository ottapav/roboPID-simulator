#!/bin/bash
# Safety during tuning (Theorem 3): published SPIN, certified SPIN and safe SPIN with the gain box,
# a fixed sampling period, cautious (START_SCALE=1) and bold (START_SCALE=2) starts.
export SPIN_BOX=1 FIXED_TS=1 RULES_ONLY=original,cert-shape,cert-ahead
for sc in 1 2; do
  START_SCALE=$sc OUT_PREFIX=safe$sc python3 designed.py 0 2 > safe${sc}_0.log 2>&1 &
  START_SCALE=$sc OUT_PREFIX=safe$sc python3 designed.py 1 2 > safe${sc}_1.log 2>&1
  wait
done
