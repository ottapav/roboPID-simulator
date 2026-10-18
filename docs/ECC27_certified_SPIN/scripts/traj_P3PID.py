"""M_s and gains (K_i, K_p, K_d) of the applied tuning in every iteration, P3 with PID, cautious
start, gain box, fixed T_s, for basic, certified and safe SPIN (Fig. 4). Writes traj_P3PID.json
({rule: [M_s per iteration]}, as before) and traj_P3PID_gains.json ({rule: [[Ki, Kp, Kd], ...]})."""
import os, json
os.environ.update(SPIN_BOX="1", FIXED_TS="1")
import designed as D
p = [q for q in D.PLANTS if q[0] == "P3 PID"][0]
res = {r: D.run(p[1], p[2], p[3], p[4], r)["traj"] for r in ("original", "cert-shape", "cert-ahead")}
out = {r: t["Ms_it"] for r, t in res.items()}
json.dump(out, open("traj_P3PID.json", "w"))
json.dump({r: t["gains_it"] for r, t in res.items()}, open("traj_P3PID_gains.json", "w"))
print({r: (sum(x > 1.7 for x in v), round(max(v), 3)) for r, v in out.items()})
