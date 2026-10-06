import numpy as np, math
import spin_run as S
def S_rec(e,dt,w):
    de=np.diff(e); t=(np.arange(len(de))+0.5)*dt; out=[]
    for wc in np.array_split(w,15): out.append(1+np.exp(-1j*np.outer(wc,t))@de)
    return np.concatenate(out)
if __name__ == "__main__":
    rows=[]
    for name in S.PLANTS:
        K,L,lags=S.PLANTS[name]
        for pid in [False,True]:
            ms,feas=S.spin(name,pid,(0.5,0.5,1.0),iters=200)
            if not feas: rows.append((name,pid,None,None)); continue
            Ki,Kp,Kd=feas[-1][3]; Mt=S.Ms_true(K,L,lags,Ki,Kp,Kd)[0]
            Ta=L+sum(lags); dt=Ta/300; e=S.sim(K,L,lags,Ki,Kp,Kd,dt,int(40*Ta/dt))
            Mr=np.max(abs(S_rec(e,dt,np.linspace(1e-3,6/L,2000))))
            rows.append((name,pid,Mt,Mr))
            print(f"{name} {'PID' if pid else 'PI '}  Ms true {Mt:.3f}   from its own step record {Mr:.3f}  ({100*(Mr-Mt)/Mt:+.1f}%)",flush=True)
    np.save('spin_cert.npy',np.array([[r[2] or np.nan, r[3] or np.nan] for r in rows]))
