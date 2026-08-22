import numpy as np, core.features as F
_HARD=F.encirc
def encirc_reg(x,y,eps=F.EPSILON):
    x=np.asarray(x,float).ravel(); y=np.asarray(y,float).ravel()
    mx,my=np.max(np.abs(x)),np.max(np.abs(y))
    if mx<1e-12 or my<1e-12 or len(x)<3: return 0.0
    x,y=x/mx,y/my
    xm,ym=0.5*(x[:-1]+x[1:]),0.5*(y[:-1]+y[1:])
    dx,dy=np.diff(x),np.diff(y)
    return float(np.sum((xm*dy-ym*dx)/(xm*xm+ym*ym+eps*eps))/(2*np.pi))
def patch(on=True): F.encirc = encirc_reg if on else _HARD
