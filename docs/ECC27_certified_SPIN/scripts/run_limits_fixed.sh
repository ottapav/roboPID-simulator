#!/bin/bash
# Certified SPIN with every limit 1/2 (gain box, fixed T_s), both starts; then the count decisions of
# certified SPIN with the published limits (confirm_limits.py, fixed T_s).
export SPIN_BOX=1 FIXED_TS=1 RULES_ONLY=cert-shape CERT_LIMITS=0.5,0.5,0.5
for sc in 1 2; do
  START_SCALE=$sc OUT_PREFIX=limf$sc python3 designed.py 0 2 > /dev/null 2>&1 &
  START_SCALE=$sc OUT_PREFIX=limf$sc python3 designed.py 1 2 > /dev/null 2>&1
  wait
done
unset CERT_LIMITS RULES_ONLY
python3 confirm_limits.py > confirm_limits.log 2>&1
