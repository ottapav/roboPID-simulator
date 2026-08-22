import numpy as np, warnings, json, itertools, sys; warnings.filterwarnings('ignore')
import regindex; regindex.patch()
from core.features import standard_pid_features
from core.tuning import pid_tuning
from collections import Counter
Ts=1.0
PL={'P1':([10.,1.,1.,1.],1.0,1.0),'P2':([5.]*4,1.25,8.0),
    'P3':([2.,1.,1.,1.],1.0,10.0),'P4':([8.]*6,1.0,4.0)}
R2M={'none':'e','N0':'d0','N1':'d1','N2':'d2'}
MV={'e':np.array([1,1,1.]),'d0':np.array([-1,0,0.]),'d1':np.array([1,-1,0.]),'d2':np.array([1,1,-1.])}

def go(tau,K,L,beta,g,ni):
    d=standard_pid_features(); rows=[]
    pid_tuning(d,np.array(tau),K,L,Ts,g[0],g[1],g[2],dtype='y',
               n_iter=ni,beta=beta,on_iteration=lambda *a: rows.append(a[-1]))
    return rows

def orbit(rows,pmax=64):
    seq=[R2M.get(r,'X') for r in rows if r in R2M]
    for p in range(1,pmax+1):
        n=0
        for i in range(len(seq)-p-1,-1,-1):
            if seq[i]==seq[i+p]: n+=1
            else: break
        if n>=2*p:
            u=seq[-p:]; c=Counter(u)
            cnt=np.array([c.get(k,0) for k in ('e','d0','d1','d2')])
            x=np.zeros(3); pts=[x.copy()]
            for m in u: x=x+MV[m]; pts.append(x.copy())
            pts=np.array(pts)
            if not np.allclose(pts[0],pts[-1]): continue
            return p,cnt,(pts[:-1].max(0)-pts[:-1].min(0))/2,u
    return None,None,None,None

BOUND=np.array([2.,1.,0.5])
out=[]
STARTS={'A':(1.0,1.0,1.0),'B':(0.5,0.2,0.5)}
for nm,(tau,K,L) in PL.items():
    for beta in (0.10,0.07):
        ni=int(300/beta)          # scale budget with 1/beta
        for sn,g in STARTS.items():
            try: rows=go(tau,K,L,beta,g,ni)
            except Exception as ex: print(nm,beta,sn,'ERR',ex,flush=True); continue
            p,cnt,hs,u=orbit(rows)
            if p is None:
                print(f'{nm} b={beta} {sn}: no periodic orbit',flush=True); continue
            ne=int(cnt[0])
            ratio=cnt/cnt[0] if ne>0 else None
            ok_counts = ne>0 and np.allclose(cnt, ne*np.array([1,4,2,1]))
            ok_bound  = np.all(hs <= ne*BOUND+1e-9) if ne>0 else None
            out.append(dict(plant=nm,beta=beta,start=sn,period=p,ne=ne,
                            counts=cnt.tolist(),half=hs.tolist(),
                            ok_counts=bool(ok_counts),ok_bound=bool(ok_bound),
                            unit=' '.join(u)))
            print(f'{nm} b={beta:.2f} {sn}: per={p:2d} ne={ne} counts={cnt.tolist()} '
                  f'half={np.round(hs,2).tolist()} bound={np.round(ne*BOUND,2).tolist()} '
                  f'counts_ok={ok_counts} bound_ok={ok_bound}',flush=True)
json.dump(out,open('/home/claude/bound_test.json','w'),indent=1)
print('\nSUMMARY: runs with a periodic orbit =',len(out))
print(' counts match ne*(1,4,2,1):',sum(o['ok_counts'] for o in out),'/',len(out))
print(' half-span within ne*(2,1,0.5)h:',sum(o['ok_bound'] for o in out),'/',len(out))
