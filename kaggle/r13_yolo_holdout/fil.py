# CAGE filament r13 YOLO retrain (PREREG comps/filament/PREREG_r13.md). r10 recipe verbatim (YOLO11s-seg, imgsz 2048, 40 ep, batch 8 DDP, seed 0),
# trained on every id whose file is NOT in the fresh holdout fresh13_holdout200 (sha asserted); inner val = RandomState(10) perm of the r13 train files
# until >= 60 ids (best.pt selection only). Writes raw holdout instances only (conf + RLE; no metric in-kernel). true_rsi=false true_agi=false. Model: Grok Bot executor.
import subprocess,sys,os,glob,json,time,hashlib,collections
if os.path.exists('/kaggle'):
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.212','pycocotools'])
import numpy as np,cv2,pandas as pd
from pycocotools import mask as mu
SMOKE=int(os.environ.get('SMOKE','0'))   # 0 full, 1 kaggle smoke, 2 box CPU smoke
T0=time.time(); W=os.getcwd()
NG=len([l for l in subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True).stdout.splitlines() if l.startswith('GPU ')]) if subprocess.run('command -v nvidia-smi',shell=True,capture_output=True).returncode==0 else 0
print('NGPU',NG,'SMOKE',SMOKE,flush=True)
R=glob.glob('/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026/',recursive=True)[0] if os.path.exists('/kaggle/input') else '/workspace/fil/MAGFiLO_1.0_Kaggle_2026/'
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
# ---- r13 fresh holdout (identical code to kern13u) ----
S0={ids[k] for k in np.random.RandomState(0).permutation(len(ids))[:100]}
grp=collections.defaultdict(list)
for i in ids: grp[imgs[i]['file_name']].append(i)
F=sorted(grp); H8=[]
for j in np.random.RandomState(8).permutation(len(F)):
    if set(grp[F[j]])&S0: continue
    H8+=grp[F[j]]
    if len(H8)>=100: break
H8F={imgs[i]['file_name'] for i in H8}
TRF8=sorted({imgs[i]['file_name'] for i in ids}-H8F); IV8=[]
for j in np.random.RandomState(10).permutation(len(TRF8)):
    IV8+=grp[TRF8[j]]
    if len(IV8)>=60: break
USED={imgs[i]['file_name'] for i in S0}|H8F|{imgs[i]['file_name'] for i in IV8}
REM=sorted(f for f in F if f not in USED); H=[]
for j in np.random.RandomState(13).permutation(len(REM)):
    H+=grp[REM[j]]
    if len(H)>=200: break
H=sorted(H); HSHA=hashlib.sha256(json.dumps(H).encode()).hexdigest()
assert HSHA=='7637297a7477e09e72b80257c397a7a1df9effe1666ce8a038f3cf7f0b3036fc',HSHA
HFILES={imgs[i]['file_name'] for i in H}
TRF=sorted({imgs[i]['file_name'] for i in ids}-HFILES)
# ---- inner validation (best.pt selection only): r13 train files, RandomState(10) perm, files until >= 60 ids (r10 rule) ----
IV=[]
for j in np.random.RandomState(10).permutation(len(TRF)):
    IV+=grp[TRF[j]]
    if len(IV)>=60: break
IV=sorted(IV); IVF={imgs[i]['file_name'] for i in IV}
TR=sorted(i for i in ids if imgs[i]['file_name'] not in HFILES|IVF)
assert not ({imgs[i]['file_name'] for i in TR}&(HFILES|IVF))
print('train ids',len(TR),'inner-val ids',len(IV),'files',len(IVF),'holdout ids',len(H),'sha',HSHA[:16],flush=True)
if SMOKE==1: TR,IV=TR[:24],IV[:6]
if SMOKE==2: TR,IV=TR[:4],IV[:2]
IMGSZ={0:2048,1:2048,2:256}[SMOKE]; EP={0:40,1:1,2:1}[SMOKE]
# ---- YOLO dataset: one image entry per annotator id (symlink), one class, single-piece polygon (largest part) ----
DS=W+'/yds'
for sp,L in (('train',TR),('val',IV)):
    os.makedirs(f'{DS}/images/{sp}',exist_ok=True); os.makedirs(f'{DS}/labels/{sp}',exist_ok=True)
    for i in L:
        p=f'{DS}/images/{sp}/{i}.jpg'
        if not os.path.exists(p): os.symlink(R+'train/train_images/'+imgs[i]['file_name'],p)
        lines=[]
        for a in anns.get(i,[]):
            seg=a['segmentation']
            if not isinstance(seg,list) or not seg: continue
            poly=max(seg,key=len)
            if len(poly)<6: continue
            xy=np.clip(np.array(poly,float).reshape(-1,2)/2048.,0,1)
            lines.append('0 '+' '.join(f'{v:.6f}' for v in xy.ravel()))
        open(f'{DS}/labels/{sp}/{i}.txt','w').write('\n'.join(lines))
open(DS+'/data.yaml','w').write(f'path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n')
dev='0,1' if NG>=2 else ('0' if NG==1 else 'cpu')
BATCH={0:8 if NG>=2 else 4,1:4 if NG>=2 else 2,2:2}[SMOKE]
TRAIN=f'''from ultralytics import YOLO
m=YOLO('yolo11s-seg.pt')
m.train(data='{DS}/data.yaml',imgsz={IMGSZ},epochs={EP},batch={BATCH},device='{dev}',workers=4,seed=0,deterministic=True,
        project='{W}/runs',name='r13',exist_ok=True,plots=False,patience=0,flipud=0.5,fliplr=0.5,hsv_h=0.0,hsv_s=0.0,hsv_v=0.4,
        degrees=0.0,mosaic=1.0,close_mosaic=10,max_det=100,amp={NG>0},verbose=True)
'''
open(W+'/train_r13.py','w').write(TRAIN)
r=subprocess.run([sys.executable,W+'/train_r13.py']); print('train rc',r.returncode,'t',round(time.time()-T0),flush=True)
if r.returncode!=0 and NG>=2:   # infra fallback only: DDP failure -> same recipe on one GPU
    open(W+'/train_r13.py','w').write(TRAIN.replace(f"device='{dev}'","device='0'").replace(f'batch={BATCH}','batch=4'))
    r=subprocess.run([sys.executable,W+'/train_r13.py']); print('train rc (1-GPU fallback)',r.returncode,flush=True)
assert r.returncode==0
WT=W+'/runs/r13/weights/best.pt' if os.path.exists(W+'/runs/r13/weights/best.pt') else W+'/runs/r13/weights/last.pt'
from ultralytics import YOLO
model=YOLO(WT)
def infer(path):   # -> list of (conf, RLE) at 2048, sorted by conf desc
    res=model.predict(path,imgsz=IMGSZ,conf=0.05,iou=0.7,max_det=100,retina_masks=True,device=('0' if NG else 'cpu'),verbose=False)[0]
    out=[]
    if res.masks is not None:
        M=res.masks.data; C=res.boxes.conf.cpu().numpy()
        for j in np.argsort(-C):
            m=(M[j]>0.5).cpu().numpy().astype(np.uint8)
            if m.shape!=(2048,2048): m=cv2.resize(m,(2048,2048),interpolation=cv2.INTER_NEAREST)
            out.append((float(C[j]),mu.encode(np.asfortranarray(m))))
    return out
import pickle
HF=sorted(HFILES)[:{0:None,1:3,2:2}[SMOKE]]
INST={f:infer(R+'train/train_images/'+f) for f in HF}
pickle.dump(INST,open(W+'/holdout_inst_r13.pkl','wb'))
json.dump(dict(round='r13',part='yolo',holdout_id='fresh13_holdout200',holdout_sha256=HSHA,n_files=len(INST),n_train_ids=len(TR),n_innerval_ids=len(IV),
               weights=os.path.basename(WT),epochs=EP,batch=BATCH,imgsz=IMGSZ,smoke=SMOKE,secs=time.time()-T0,
               true_rsi=False,true_agi=False,counts_as_aha=False,counts_as_rsi_evidence=False),open(W+'/done_r13_yolo.json','w'))
print('R13YOLO wrote instances for',len(INST),'files',round(time.time()-T0),flush=True)
