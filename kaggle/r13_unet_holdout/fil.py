# CAGE filament r13 U-Net retrain (PREREG comps/filament/PREREG_r13.md). r5 recipe verbatim (base_mask 22 ep seed 5, hull 16 ep seed 6, 1024 whole image),
# trained on every id whose file is NOT in the fresh holdout fresh13_holdout200 (sha asserted). Writes raw holdout probs only (no metric in-kernel).
# Classical compute only. true_rsi=false true_agi=false. Model: Grok Bot executor.
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
# ---- r13 fresh holdout (frozen: /workspace/fil/r13/fresh13_holdout200.json) ----
S0={ids[k] for k in np.random.RandomState(0).permutation(len(ids))[:100]}
grp=collections.defaultdict(list)
for i in ids: grp[imgs[i]['file_name']].append(i)
F=sorted(grp); H=[]
for j in np.random.RandomState(8).permutation(len(F)):
    if set(grp[F[j]])&S0: continue
    H+=grp[F[j]]
    if len(H)>=100: break
HFILES={imgs[i]['file_name'] for i in H}
TRF=sorted({imgs[i]['file_name'] for i in ids}-HFILES); IV=[]
for j in np.random.RandomState(10).permutation(len(TRF)):
    IV+=grp[TRF[j]]
    if len(IV)>=60: break
USED={imgs[i]['file_name'] for i in S0}|HFILES|{imgs[i]['file_name'] for i in IV}
REM=sorted(f for f in F if f not in USED); N=[]
for j in np.random.RandomState(13).permutation(len(REM)):
    N+=grp[REM[j]]
    if len(N)>=200: break
N=sorted(N); NSHA=hashlib.sha256(json.dumps(N).encode()).hexdigest()
assert NSHA=='7637297a7477e09e72b80257c397a7a1df9effe1666ce8a038f3cf7f0b3036fc',NSHA
NFILES=sorted({imgs[i]['file_name'] for i in N})
ti=[k for k in range(len(ids)) if imgs[ids[k]]['file_name'] not in set(NFILES)]
assert not ({imgs[ids[k]]['file_name'] for k in ti}&set(NFILES))
print('holdout ids',len(N),'files',len(NFILES),'train ids',len(ti),'sha',NSHA[:16],flush=True)
vi=[]
if SMOKE==1: NFILES,ti=NFILES[:3],ti[:24]
if SMOKE==2: NFILES,ti=NFILES[:2],ti[:4]
fn=lambda k: imgs[ids[k]]['file_name']
gray={f:cv2.imread(R+'train/train_images/'+f,0) for f in {fn(k) for k in ti}}
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
JOBS={'base_mask':('whole',0,EPM,5),'hull':('whole',1,EPH,6)}
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
    plan=[['base_mask'],['hull']]
    ps=[mp.get_context('fork').Process(target=worker,args=(p,g)) for g,p in enumerate(plan)]
    [p.start() for p in ps]; [p.join() for p in ps]
    assert all(p.exitcode==0 for p in ps),[p.exitcode for p in ps]
else:
    for nm in JOBS: train(nm,dev)
del TG,NAT,G1024
if NG: print('GPUs',[torch.cuda.get_device_name(i) for i in range(NG)],flush=True)
def load(name):
    net=smp.Unet('resnet34',encoder_weights=None,in_channels=1,classes=1); net.load_state_dict(torch.load(f'{name}.pt',map_location='cpu')); return net.to(dev).eval()
BM,HU=load('base_mask'),load('hull')
@torch.no_grad()
def prob(net,g,size):   # identical to r8/r11 cache prob(): 1024 INTER_AREA, hflip TTA, uint8
    a=cv2.resize(g,(size,size),interpolation=cv2.INTER_AREA) if size!=g.shape[0] else g
    x=norm(a[None])[:,None].to(dev)
    q=(torch.sigmoid(net(x))+torch.sigmoid(net(x.flip(-1))).flip(-1))/2
    return (q[0,0].float().cpu().numpy()*255).astype(np.uint8)
Q={}
for f in NFILES:
    g=cv2.imread(R+'train/train_images/'+f,0); Q[f]=np.stack([prob(BM,g,S),prob(HU,g,S)],-1)
np.savez_compressed('holdout_probs_r13.npz',**{f:Q[f] for f in Q})
json.dump(dict(round='r13',part='unet',holdout_id='fresh13_holdout200',holdout_sha256=NSHA,n_files=len(Q),n_train_ids=len(ti),smoke=SMOKE,secs=time.time()-T0,
               true_rsi=False,true_agi=False,counts_as_aha=False,counts_as_rsi_evidence=False),open('done_r13_unet.json','w'))
print('R13UNET wrote probs for',len(Q),'files',round(time.time()-T0),flush=True)
