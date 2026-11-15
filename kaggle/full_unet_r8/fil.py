# CAGE filament r8 (PREREG comps/filament/PREREG_r8.md). ONE change vs r5: the MASK net is trained on native-resolution 1024x1024 tiles
# cropped from the 2048 image and predicts the mask at full 2048 resolution (r5: whole image downsampled to 1024, mask upsampled back).
# Fresh FILE-GROUPED holdout seed8_holdout100 (no image file in both train and holdout; no id of the retired seed0 holdout).
# In-kernel r5 replay on the same split: r5 mask net (1024, 22 ep, seed 5) + r5 hull net (1024, 16 ep, seed 6); the hull net is shared
# by both arms; grouping = r5 config B (th .5, minA 250, close 51, ht .3) for both. CUDA AMP (fp16) for all three nets alike;
# 2+ GPUs -> nets train in parallel forked workers. TPU (torch_xla) / CPU paths serial fp32. Classical compute only. true_rsi=false true_agi=false
import subprocess,sys,os,glob
if os.path.exists('/kaggle'):
    subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','segmentation-models-pytorch','timm','pycocotools'])
    subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub','safetensors','scikit-image'])
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK','1')   # A1 (infra): NVML-based availability/count so the parent never initialises CUDA before the fork
import json,time,hashlib,collections,numpy as np,cv2,torch,pandas as pd
cv2.setNumThreads(0)   # no OpenCV thread pool in the parent before the fork
import multiprocessing as mp
import segmentation_models_pytorch as smp
from pycocotools import mask as mu
SMOKE=int(os.environ.get('SMOKE','0'))   # 0 full, 1 kaggle smoke, 2 box smoke
T0=time.time()
try:
    import torch_xla.core.xla_model as xm; dev=xm.xla_device(); XLA=True
except Exception as e:
    print('no xla',e); dev='cuda' if torch.cuda.is_available() else 'cpu'; XLA=False
NG=(len([l for l in subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True).stdout.splitlines() if l.startswith('GPU ')]) if dev=='cuda' else 0)   # A1: count GPUs without torch.cuda
assert not torch.cuda.is_initialized(),'A1: CUDA initialised in parent before fork'
AMP=(dev=='cuda')
print('DEVICE',dev,'NGPU',NG,'AMP',AMP,'SMOKE',SMOKE,flush=True)   # no CUDA init in the parent before the fork
if os.path.exists('/kaggle/input'):
    R=glob.glob('/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026/',recursive=True)[0]
else:
    R='/workspace/fil/MAGFiLO_1.0_Kaggle_2026/'
d=json.load(open(R+'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
files=set(os.listdir(R+'train/train_images'))
imgs={i['id']:i for i in d['images'] if i['file_name'] in files}
anns={}
for a in d['annotations']:
    if a['image_id'] in imgs: anns.setdefault(a['image_id'],[]).append(a)
def decode(seg):
    if isinstance(seg,list): rle=mu.merge(mu.frPyObjects(seg,2048,2048))
    elif isinstance(seg['counts'],list): rle=mu.frPyObjects(seg,2048,2048)
    else: rle=seg
    return mu.decode(rle).astype(np.uint8)
ids=sorted(imgs)
# ---- fresh file-grouped holdout (frozen: comps/_guard/holdout_rows/filament_seed8_holdout100.json) ----
S0={ids[k] for k in np.random.RandomState(0).permutation(len(ids))[:100]}   # retired seed0_holdout100 ids
grp=collections.defaultdict(list)
for i in ids: grp[imgs[i]['file_name']].append(i)
F=sorted(grp); H=[]
for j in np.random.RandomState(8).permutation(len(F)):
    if set(grp[F[j]])&S0: continue
    H+=grp[F[j]]
    if len(H)>=100: break
H=sorted(H); HSHA=hashlib.sha256(json.dumps(H).encode()).hexdigest()
assert HSHA=='fdb6aeee750984370ef9389b15e6f6af628d53ac52e52397cbded05cc8bd300b',HSHA
HFILES={imgs[i]['file_name'] for i in H}
vi=[ids.index(i) for i in H]; ti=[k for k in range(len(ids)) if imgs[ids[k]]['file_name'] not in HFILES]
assert not ({imgs[ids[k]]['file_name'] for k in ti}&HFILES)
print('holdout ids',len(vi),'files',len(HFILES),'train ids',len(ti),'sha',HSHA[:16],flush=True)
if SMOKE==1: vi,ti=vi[:6],ti[:24]
if SMOKE==2: vi,ti=vi[:3],ti[:4]
fn=lambda k: imgs[ids[k]]['file_name']
gray={f:cv2.imread(R+'train/train_images/'+f,0) for f in {fn(k) for k in list(ti)+list(vi)}}
S=1024; TILE=1024; NEV=2048   # r8 mask inference resolution (native)
if SMOKE==2: S,TILE,NEV=256,256,512   # box CPU smoke only (shared box RAM): same code paths at reduced sizes
ell=lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
def targets(k):   # r5 targets at 1024 (unchanged) + r8 native 2048 mask (packed bits)
    full=[decode(a_['segmentation']) for a_ in anns.get(ids[k],[])]
    ms=[cv2.resize(m,(S,S),interpolation=cv2.INTER_NEAREST) for m in full]
    mask=np.zeros((S,S),np.uint8); cnt=np.zeros((S,S),np.uint8); hs=[]
    for m in ms:
        if m.sum()==0: continue
        mask|=m; h=cv2.morphologyEx(m,cv2.MORPH_CLOSE,ell(25))|m; hs.append(h); cnt+=cv2.dilate(h,ell(5))
    hull=np.zeros((S,S),np.uint8)
    for h in hs: hull|=h
    hull[cnt>1]=0
    nat=np.zeros((2048,2048),np.uint8)
    for m in full: nat|=m
    return np.stack([mask,hull]),np.packbits(nat)
TG,NAT={},{}
for k in ti: TG[k],NAT[k]=targets(k)
G1024={f:cv2.resize(gray[f],(S,S),interpolation=cv2.INTER_AREA) for f in {fn(k) for k in ti}}   # r5 precomputed X
print('loaded',round(time.time()-T0),flush=True)
def norm(x): return (torch.tensor(x,dtype=torch.float32)/255.-0.5)/0.25
# job: (name, kind 'whole'|'tile', channel, epochs, seed); epochs/seeds = r5's (mask 22/seed 5, hull 16/seed 6)
EPM,EPH={0:22,1:1,2:1}[SMOKE],{0:16,1:1,2:1}[SMOKE]; BS=1 if SMOKE==2 else 4
JOBS={'base_mask':('whole',0,EPM,5),'hull':('whole',1,EPH,6),'r8_mask':('tile',0,EPM,5)}
def batch(kind,ch,b,rr,rc):
    if kind=='whole':
        X=np.stack([G1024[fn(k)] for k in b]); Y=np.stack([TG[k][ch] for k in b])
    else:   # r8: native-res tile; half the tiles centred on a random filament pixel of this id, half uniform
        X=[];Y=[]
        for k in b:
            m=np.unpackbits(NAT[k]).reshape(2048,2048)
            if rc.rand()<.5 and m.any():
                ys,xs=np.nonzero(m); j=rc.randint(len(ys)); y0=int(np.clip(ys[j]-TILE//2,0,2048-TILE)); x0=int(np.clip(xs[j]-TILE//2,0,2048-TILE))
            else: y0,x0=rc.randint(0,2048-TILE+1),rc.randint(0,2048-TILE+1)
            X.append(gray[fn(k)][y0:y0+TILE,x0:x0+TILE]); Y.append(m[y0:y0+TILE,x0:x0+TILE])
        X=np.stack(X); Y=np.stack(Y)
    return X,Y
def train(name,device):
    kind,ch,EP,seed=JOBS[name]; torch.manual_seed(seed); rr=np.random.RandomState(seed); rc=np.random.RandomState(seed+1000)
    net=smp.Unet('resnet34',encoder_weights='imagenet',in_channels=1,classes=1).to(device)
    dl=smp.losses.DiceLoss('multilabel'); bce=torch.nn.BCEWithLogitsLoss(); n=len(ti); nb=(n+BS-1)//BS
    opt=torch.optim.AdamW(net.parameters(),3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=EP*nb)
    amp=AMP and str(device).startswith('cuda'); scaler=torch.amp.GradScaler('cuda',enabled=amp); tr=np.array(ti); t0=time.time()
    for ep in range(EP):
        net.train(); o=rr.permutation(n); tl=torch.zeros((),device=device)
        for i in range(0,n,BS):
            X,Y=batch(kind,ch,tr[o[i:i+BS]],rr,rc); x=norm(X)[:,None]; y=torch.tensor(Y[:,None]).float()
            if rr.rand()<.5: x,y=x.flip(-1),y.flip(-1)
            if rr.rand()<.5: x,y=x.flip(-2),y.flip(-2)
            x=x*(0.8+0.4*rr.rand())+0.2*(rr.rand()-.5); x,y=x.to(device),y.to(device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                lo=net(x)
            lo=lo.float(); loss=dl(lo,y)+bce(lo,y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl=tl+loss.detach()
            if XLA: xm.mark_step()
        print(name,'ep',ep,round(float(tl.cpu()),3),'epoch_s',round((time.time()-t0)/(ep+1),1),'t',round(time.time()-T0),flush=True)
    torch.save({k:v.cpu() for k,v in net.state_dict().items()},f'{name}.pt'); print('TRAINED',name,'secs',round(time.time()-t0),flush=True)
def worker(names,gi):
    time.sleep(45*gi)   # stagger the imagenet-weights fetch
    torch.cuda.set_device(gi); print('GPU',gi,torch.cuda.get_device_name(gi),names,flush=True)
    for nm in names: train(nm,f'cuda:{gi}')
if NG>=2:   # parallel nets on separate GPUs (fork before any CUDA init in this process)
    plan=[['base_mask','hull'],['r8_mask']] if NG==2 else [['base_mask'],['hull'],['r8_mask']]
    ps=[mp.get_context('fork').Process(target=worker,args=(p,g)) for g,p in enumerate(plan)]
    [p.start() for p in ps]; [p.join() for p in ps]
    assert all(p.exitcode==0 for p in ps),[p.exitcode for p in ps]
else:
    for nm in JOBS: train(nm,dev)
del TG,NAT,G1024
if NG: print('GPUs',[torch.cuda.get_device_name(i) for i in range(NG)],flush=True)
def load(name):
    net=smp.Unet('resnet34',encoder_weights=None,in_channels=1,classes=1); net.load_state_dict(torch.load(f'{name}.pt',map_location='cpu')); return net.to(dev).eval()
BM,HU,R8=load('base_mask'),load('hull'),load('r8_mask')
@torch.no_grad()
def prob(net,g,size):
    a=cv2.resize(g,(size,size),interpolation=cv2.INTER_AREA) if size!=g.shape[0] else g
    x=norm(a[None])[:,None].to(dev)
    q=(torch.sigmoid(net(x))+torch.sigmoid(net(x.flip(-1))).flip(-1))/2
    return (q[0,0].float().cpu().numpy()*255).astype(np.uint8)
def finish(out,m,minA):
    out=out.copy(); out[m==0]=0
    u,c=np.unique(out,return_counts=True); keep=np.zeros(out.max()+1,np.int32); k=0
    for i_,n in zip(u,c):
        if i_>0 and n>=minA: k+=1; keep[i_]=k
    return keep[out],k
def labels_B(Q,th=.5,minA=250,close=51,ht=.3):   # r5 config B (unchanged); Q = (2048,2048,2) uint8 [mask, hull]
    m=(Q[...,0]>th*255).astype(np.uint8)
    lm=cv2.morphologyEx(m,cv2.MORPH_CLOSE,ell(close)); _,base=cv2.connectedComponents(lm,8)
    h=(Q[...,1]>ht*255).astype(np.uint8); nh,hl=cv2.connectedComponents(h,8)
    out=np.where(hl>0,hl,0).astype(np.int32); miss=(m>0)&(out==0); out[miss]=base[miss]+nh
    return finish(out,m,minA),m
def Q_arms(g):   # base: r5 path (1024 probs resized to 2048 together); r8: native 2048 mask + same resized hull
    qb=np.stack([prob(BM,g,S),prob(HU,g,S),np.zeros((S,S),np.uint8)],-1); Qb=cv2.resize(qb,(2048,2048))
    qr=prob(R8,g,NEV); qr=qr if NEV==2048 else cv2.resize(qr,(2048,2048))
    Qr=np.stack([qr,Qb[...,1]],-1)
    return Qb[...,:2],Qr
def pq_img(gl,ng,pl,npr):
    J=np.bincount(gl.ravel().astype(np.int64)*(npr+1)+pl.ravel(),minlength=(ng+1)*(npr+1)).reshape(ng+1,npr+1)
    ga=J.sum(1); pa=J.sum(0); I=J[1:,1:]; U=ga[1:,None]+pa[None,1:]-I; iou=np.where(U>0,I/np.maximum(U,1),0)
    mt=iou>0.5; t=int(mt.sum()); s=float(iou[mt].sum()); den=t+.5*(npr-t)+.5*(ng-t)
    return dict(tp=t,s=s,fn=ng-t,fp=npr-t,pq=(s/den if den>0 else 1.0))
def pq(rows): tp=sum(r['tp'] for r in rows); s=sum(r['s'] for r in rows); fp=sum(r['fp'] for r in rows); fn_=sum(r['fn'] for r in rows); return s/max(tp+.5*fp+.5*fn_,1e-9)
def dice(a,b): return float(2*(a&b).sum()/max(a.sum()+b.sum(),1))
rows={'base':[],'r8':[]}; diag={'base':[],'r8':[]}; QC={}
for n,k in enumerate(vi):
    f=fn(k)
    if f not in QC: QC[f]=Q_arms(gray[f])
    Qb,Qr=QC[f]; al=anns.get(ids[k],[]); gl=np.zeros((2048,2048),np.int32)
    for j,a in enumerate(al,1): gl[decode(a['segmentation'])>0]=j
    for arm,Q in (('base',Qb),('r8',Qr)):
        (lb,kb),m=labels_B(Q); r=pq_img(gl,len(al),lb,kb); r['id']=str(ids[k]); r['n_inst']=kb; rows[arm].append(r); diag[arm].append(dice(m>0,gl>0))
    if n%10==0: print(n,'base',round(pq(rows['base']),4),'r8',round(pq(rows['r8']),4),round(time.time()-T0),flush=True)
res=dict(round='r8',smoke=SMOKE,device=str(dev),n_gpu=NG,amp=AMP,n_images=len(vi),holdout_id='seed8_holdout100',holdout_sha256=HSHA,
         pq_base=pq(rows['base']),pq_r8=pq(rows['r8']),mask_dice_mean=dict(base=float(np.mean(diag['base'])),r8=float(np.mean(diag['r8']))),
         per_image=rows,secs=time.time()-T0,true_rsi=False,true_agi=False,counts_as_aha=False,counts_as_rsi_evidence=False)
res['bar']=res['pq_base']+0.02
res['sanity_ok']=bool(0.25<=res['pq_base']<=0.55) if SMOKE==0 else None
res['decision']=('PASS_BAR' if res['sanity_ok'] and res['pq_r8']>=res['bar'] else 'NO_PASS') if SMOKE==0 else 'SMOKE'
json.dump(res,open('eval_r8.json','w'),indent=1)
print('R8RESULT',json.dumps({k:v for k,v in res.items() if k!='per_image'}),flush=True)
tf=sorted(os.listdir(R+'test/test_images'))[:{0:None,1:5,2:2}[SMOKE]]
out=[]
for f in tf:
    g=cv2.imread(R+'test/test_images/'+f,0); (lab,k),_=labels_B(Q_arms(g)[1])
    for j in range(1,k+1):
        bm=lab==j
        if bm.any(): r=mu.encode(np.asfortranarray(bm.astype(np.uint8))); out.append((f[:-5]+f'_{j}',r['counts'].decode()))
pd.DataFrame(out,columns=['filament_id','segmentation_rle']).to_csv('submission.csv',index=False)
print('wrote',len(out),'rows for',len(tf),'images',round(time.time()-T0),flush=True)
