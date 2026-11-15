import sys,pickle,json,numpy as np,cv2,torch
sys.path.insert(0,'/workspace/fil/r11'); from common import *; from selfeval import *
C='/dev/shm/fil_r11/'
ell=lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
def finish(out,m,minA):
    out=out.copy(); out[m==0]=0
    u,c=np.unique(out,return_counts=True); keep=np.zeros(out.max()+1,np.int32); k=0
    for i_,n in zip(u,c):
        if i_>0 and n>=minA: k+=1; keep[i_]=k
    return keep[out],k
def labels_B(q,th=.5,minA=250,close=51,ht=.3):   # r5 config B, from cached 1024 probs (resized to 2048 as in r8 Q_arms)
    Q=cv2.resize(np.dstack([q[...,0],q[...,1],np.zeros_like(q[...,0])]),(2048,2048))[...,:2]
    m=(Q[...,0]>th*255).astype(np.uint8)
    lm=cv2.morphologyEx(m,cv2.MORPH_CLOSE,ell(close)); _,base=cv2.connectedComponents(lm,8)
    h=(Q[...,1]>ht*255).astype(np.uint8); nh,hl=cv2.connectedComponents(h,8)
    out=np.where(hl>0,hl,0).astype(np.int32); miss=(m>0)&(out==0); out[miss]=base[miss]+nh
    return finish(out,m,minA)
def dec_inst(inst):
    out=[]
    for c,rle in inst:
        m=mu.decode(rle)>0; ys=np.flatnonzero(m.any(1)); xs=np.flatnonzero(m.any(0))
        if len(ys): out.append((c,ys[0],ys[-1]+1,xs[0],xs[-1]+1,m[ys[0]:ys[-1]+1,xs[0]:xs[-1]+1]))
    return out
def labels_Y(di,th=.25,minA=250):
    lab=np.zeros((2048,2048),np.int32); k=0
    for c,y0,y1,x0,x1,m in di:
        if c<th: break
        sub=lab[y0:y1,x0:x1]; mm=m&(sub==0)
        if mm.sum()>=max(minA,1): k+=1; sub[mm]=k
    return lab,k
