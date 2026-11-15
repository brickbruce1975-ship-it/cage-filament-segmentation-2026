# Filament r12 (PREREG comps/filament/PREREG_r12_A1.md, base PREREG_r12.md): r11 agreement fusion + small-filament YOLO recall add. Model: Grok Bot executor.
import sys; sys.path.insert(0,'/workspace/fil/r11'); from analyze import *
import time
T0=time.time()
SM=[(None,None),(0.25,2000),(0.25,5000),(0.15,2000),(0.15,5000),(0.10,2000),(0.10,5000),(0.05,2000),(0.05,5000)]
GRID=[(t,c,s,a) for t in (0.1,0.2,0.3) for c in (None,0.5) for (s,a) in SM]
assert len(GRID)==54
def flat(m): return np.flatnonzero(m.ravel())
def prep_file(base_lab,kb,inst):
    ba=np.bincount(base_lab.ravel(),minlength=kb+1); Ys=[]
    for c,rle in inst:
        ix=flat(mu.decode(rle)>0)
        if len(ix): Ys.append((c,ix))
    agree=np.zeros(kb+1)
    for c,ix in Ys:
        if c<0.25: continue
        I=np.bincount(base_lab.ravel()[ix],minlength=kb+1).astype(float); U=ba+len(ix)-I; iou=I/np.maximum(U,1); iou[0]=0
        agree=np.maximum(agree,iou)
    return dict(base=base_lab,kb=kb,agree=agree,Ys=Ys)
def fuse(P,t,c,s,a):
    kb=P['kb']; keep=np.zeros(kb+1,np.int32); k=0
    for j in range(1,kb+1):
        if P['agree'][j]>=t: k+=1; keep[j]=k
    lab=keep[P['base']].ravel().copy()
    if c is not None:   # step 1: r11 adds (unchanged)
        kept=lab>0
        for cf,ix in P['Ys']:
            if cf<c: continue
            if kept[ix].sum()>=0.2*len(ix): continue
            free=ix[lab[ix]==0]
            if len(free)>=250: k+=1; lab[free]=k
    if s is not None:   # step 2: THE r12 change, small YOLO adds (Ys already in descending conf)
        for cf,ix in P['Ys']:
            if cf<s or len(ix)>=a: continue
            if (lab[ix]>0).sum()>=0.2*len(ix): continue
            free=ix[lab[ix]==0]
            if len(free)>=100: k+=1; lab[free]=k
    return lab,k
GT={}
def gt_flat(i):
    if i not in GT: GT[i]=[flat(g>0) for g in gt_layers(i)]
    return GT[i]
def score_row(i,lab,k):
    G=gt_flat(i); ng=len(G)
    if ng==0 or k==0: iou=torch.zeros(ng,k)
    else:
        pa=np.bincount(lab,minlength=k+1)[1:].astype(float); I=np.stack([np.bincount(lab[g],minlength=k+1)[1:] for g in G]).astype(float)
        ga=np.array([len(g) for g in G],float)[:,None]; U=ga+pa[None]-I; iou=torch.tensor(np.where(U==0,0.,I/np.maximum(U,1e-9)),dtype=torch.float32)
    _,t=get_pq_score([(iou,ng,k)]); den=t['tp']+.5*t['fp']+.5*t['fn']
    return (iou,ng,k),dict(id=i,**t,pq=(t['sum_iou']/den if den>0 else 1.0))
if __name__=='__main__':
    PF={}
    for f in sorted(HFILES):
        D=pickle.load(open(C+f+'.pkl','rb')); bl,kb=labels_B(D['q']); PF[f]=prep_file(bl,kb,D['inst'])
    print('prepped',len(PF),round(time.time()-T0),flush=True)
    ROWS={}
    for g in GRID:
        ROWS[g]={}
        for f in sorted(HFILES):
            lab,k=fuse(PF[f],*g)
            for i in grp[f]: ROWS[g][i]=score_row(i,lab,k)
        print('cfg',g,round(get_pq_score([ROWS[g][i][0] for i in H])[0],5),round(time.time()-T0),flush=True)
    # incumbent r5 replay = r11's (0,None) config: no filtering, no adds
    INC={}
    for f in sorted(HFILES):
        P=PF[f]; lab=P['base'].ravel().copy(); k=P['kb']
        for i in grp[f]: INC[i]=score_row(i,lab,k)
    pooled=lambda g,L: get_pq_score([ROWS[g][i][0] for i in L])[0]
    FS=sorted(HFILES); perm=np.random.RandomState(11).permutation(len(FS))
    FA={FS[j] for n,j in enumerate(perm) if n%2==0}; A=[i for i in H if fn(i) in FA]; B=[i for i in H if fn(i) not in FA]
    def pick(L): return max(GRID,key=lambda g:(round(pooled(g,L),12),-GRID.index(g)))
    gA=pick(B); gB=pick(A)
    cand={i:(ROWS[gA][i] if i in A else ROWS[gB][i]) for i in H}
    pq_c=get_pq_score([cand[i][0] for i in H]); pq_b=get_pq_score([INC[i][0] for i in H])
    gT=pick(H)
    res=dict(round='r12',prereg='comps/filament/PREREG_r12_A1.md',prereg_base_sha='acad78bab5e7ecddb8161c081243308f3e172bee483c5f6a41fd25aa1a3c37d8',
             prereg_a1_sha='e2347bf49ca3d6d3254aa47ece121220042752b7fdddd8b7122ef4c784e87964',
             model='Grok Bot executor; frozen r5 replay U-Net (r8 v3) + r10 YOLO11s-seg; fusion + small-add post-processing',
             n_A=len(A),n_B=len(B),cfg_A=gA,cfg_B=gB,cfg_test=gT,pq_cand=pq_c[0],pq_base=pq_b[0],cand_counts=pq_c[1],base_counts=pq_b[1],
             best_in_sample=pooled(gT,H),
             grid_full={str(g):pooled(g,H) for g in GRID},grid_A={str(g):pooled(g,A) for g in GRID},grid_B={str(g):pooled(g,B) for g in GRID},
             rows_cand=[cand[i][1] for i in H],rows_base=[INC[i][1] for i in H],secs=time.time()-T0,
             true_rsi=False,true_agi=False,counts_as_aha=False,counts_as_rsi_evidence=False)
    json.dump(res,open('/workspace/fil/r12/eval_r12.json','w'),indent=1,default=str)
    print('R12RESULT',json.dumps({k:v for k,v in res.items() if not k.startswith(('rows','grid'))},default=str))
