# CAGE filament r10 (PREREG comps/filament/PREREG_r10.md). ONE change vs r5: model family. The r5 U-Net mask + hull grouping is replaced by
# an instance-segmentation model (Ultralytics YOLO11s-seg, COCO-pretrained, 1 class "filament"), trained at native 2048 px on the same
# file-grouped split as r8/r9 (holdout seed8_holdout100 asserted by sha256). Detection confidence threshold and min area are picked on an
# INNER validation split carved from the training files (never on the gate holdout); the gate holdout is scored once with that fixed config.
# Predicted instances are made non-overlapping (higher confidence keeps contested pixels). Incumbent rows = in-kernel r5 replay rows from r8 v3
# (same 100 ids; reproduced by r9 to 1e-5). Classical compute only. true_rsi=false true_agi=false
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
# ---- frozen file-grouped gate holdout seed8_holdout100 (identical code to r8) ----
S0={ids[k] for k in np.random.RandomState(0).permutation(len(ids))[:100]}
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
TRF=sorted({imgs[i]['file_name'] for i in ids}-HFILES)
# ---- inner validation (tuning only): train files, RandomState(10) perm, files until >= 60 ids ----
IV=[]
for j in np.random.RandomState(10).permutation(len(TRF)):
    IV+=grp[TRF[j]]
    if len(IV)>=60: break
IV=sorted(IV); IVF={imgs[i]['file_name'] for i in IV}
TR=sorted(i for i in ids if imgs[i]['file_name'] not in HFILES|IVF)
assert not ({imgs[i]['file_name'] for i in TR}&(HFILES|IVF))
print('train ids',len(TR),'inner-val ids',len(IV),'files',len(IVF),'holdout ids',len(H),'sha',HSHA[:16],flush=True)
if SMOKE==1: TR,IV,H=TR[:24],IV[:6],H[:6]
if SMOKE==2: TR,IV,H=TR[:4],IV[:2],H[:2]
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
        project='{W}/runs',name='r10',exist_ok=True,plots=False,patience=0,flipud=0.5,fliplr=0.5,hsv_h=0.0,hsv_s=0.0,hsv_v=0.4,
        degrees=0.0,mosaic=1.0,close_mosaic=10,max_det=100,amp={NG>0},verbose=True)
'''
open(W+'/train_r10.py','w').write(TRAIN)
r=subprocess.run([sys.executable,W+'/train_r10.py']); print('train rc',r.returncode,'t',round(time.time()-T0),flush=True)
if r.returncode!=0 and NG>=2:   # infra fallback only: DDP failure -> same recipe on one GPU
    open(W+'/train_r10.py','w').write(TRAIN.replace(f"device='{dev}'","device='0'").replace(f'batch={BATCH}','batch=4'))
    r=subprocess.run([sys.executable,W+'/train_r10.py']); print('train rc (1-GPU fallback)',r.returncode,flush=True)
assert r.returncode==0
WT=W+'/runs/r10/weights/best.pt' if os.path.exists(W+'/runs/r10/weights/best.pt') else W+'/runs/r10/weights/last.pt'
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
def dec_inst(inst):   # decode once -> (conf, y0,y1,x0,x1, cropped bool mask)
    out=[]
    for c,rle in inst:
        m=mu.decode(rle)>0; ys=np.flatnonzero(m.any(1)); xs=np.flatnonzero(m.any(0))
        if len(ys): out.append((c,ys[0],ys[-1]+1,xs[0],xs[-1]+1,m[ys[0]:ys[-1]+1,xs[0]:xs[-1]+1]))
    return out
def labels(dinst,th,minA):   # non-overlapping label map; higher conf keeps contested pixels
    lab=np.zeros((2048,2048),np.int32); k=0
    for c,y0,y1,x0,x1,m in dinst:
        if c<th: break
        sub=lab[y0:y1,x0:x1]; mm=m&(sub==0)
        if mm.sum()>=max(minA,1): k+=1; sub[mm]=k
    return lab,k
def pq_img(gl,ng,pl,npr):   # identical to r8 (per-id rows)
    J=np.bincount(gl.ravel().astype(np.int64)*(npr+1)+pl.ravel(),minlength=(ng+1)*(npr+1)).reshape(ng+1,npr+1)
    ga=J.sum(1); pa=J.sum(0); I=J[1:,1:]; U=ga[1:,None]+pa[None,1:]-I; iou=np.where(U>0,I/np.maximum(U,1),0)
    mt=iou>0.5; t=int(mt.sum()); s=float(iou[mt].sum()); den=t+.5*(npr-t)+.5*(ng-t)
    return dict(tp=t,s=s,fn=ng-t,fp=npr-t,pq=(s/den if den>0 else 1.0))
def pq(rows): tp=sum(r['tp'] for r in rows); s=sum(r['s'] for r in rows); fp=sum(r['fp'] for r in rows); fn_=sum(r['fn'] for r in rows); return s/max(tp+.5*fp+.5*fn_,1e-9)
def gt_layers(i): return [decode(a['segmentation']) for a in anns.get(i,[])]
def official_rows(i,lab,k):   # host self-evaluation logic: GT layers vs predicted layers, IoU>0.5 hits
    G=gt_layers(i); ng=len(G)
    if ng==0: return dict(tp=0,s=0.,fp=k,fn=0)
    if k==0: return dict(tp=0,s=0.,fp=0,fn=ng)
    flat=lab.ravel(); iou=np.zeros((ng,k))
    pa=np.bincount(flat,minlength=k+1)[1:]
    for gi,g in enumerate(G):
        gm=g.ravel()>0; inter=np.bincount(flat[gm],minlength=k+1)[1:]; iou[gi]=inter/np.maximum(gm.sum()+pa-inter,1)
    hit=iou>0.5
    return dict(tp=int(hit.sum()),s=float(iou[hit].sum()),fp=int((hit.sum(0)==0).sum()),fn=int((hit.sum(1)==0).sum()))
def gtmap(i):
    gl=np.zeros((2048,2048),np.int32); al=anns.get(i,[])
    for j,a in enumerate(al,1): gl[decode(a['segmentation'])>0]=j
    return gl,len(al)
# ---- tune (th, minA) on inner val only ----
CACHE={}
def inst_of(f):
    if f not in CACHE: CACHE[f]=infer(R+'train/train_images/'+f)
    return CACHE[f]
GRID=[(th,mA) for th in [round(x,2) for x in np.arange(0.10,0.751,0.05)] for mA in (0,100,250)]
GIV={i:gtmap(i) for i in IV}
TROWS={g:[] for g in GRID}
for i in IV:
    di=dec_inst(inst_of(imgs[i]['file_name'])); gl,ng=GIV[i]
    for g in GRID: lab,k=labels(di,*g); TROWS[g].append(pq_img(gl,ng,lab,k))
tune={f'{th}_{mA}':pq(TROWS[(th,mA)]) for th,mA in GRID}
best=max(tune,key=tune.get); TH,MINA=float(best.split('_')[0]),int(best.split('_')[1])
print('inner-val tune best',best,round(tune[best],4),'t',round(time.time()-T0),flush=True)
del GIV; CACHE={}
# ---- gate holdout, scored once with the fixed config ----
rows=[]; orows=[]; dice=[]
for n,i in enumerate(H):
    lab,k=labels(dec_inst(inst_of(imgs[i]['file_name'])),TH,MINA); gl,ng=gtmap(i)
    r_=pq_img(gl,ng,lab,k); r_['id']=str(i); r_['n_inst']=k; rows.append(r_); orows.append(official_rows(i,lab,k))
    dice.append(float(2*((lab>0)&(gl>0)).sum()/max((lab>0).sum()+(gl>0).sum(),1)))
    if n%20==0: print(n,'r10',round(pq(rows),4),round(time.time()-T0),flush=True)
res=dict(round='r10',smoke=SMOKE,n_gpu=NG,model='ultralytics YOLO11s-seg (yolo11s-seg.pt COCO init), 1 class',imgsz=IMGSZ,epochs=EP,batch=BATCH,
         weights=os.path.basename(WT),n_images=len(H),holdout_id='seed8_holdout100',holdout_sha256=HSHA,inner_val_ids=IV,tune=tune,th=TH,minA=MINA,
         pq_r10=pq(rows),pq_r10_official=pq(orows),mask_dice_mean=float(np.mean(dice)),per_image=rows,secs=time.time()-T0,
         true_rsi=False,true_agi=False,counts_as_aha=False,counts_as_rsi_evidence=False)
json.dump(res,open(W+'/eval_r10.json','w'),indent=1)
print('R10RESULT',json.dumps({k:v for k,v in res.items() if k not in('per_image','inner_val_ids','tune')}),flush=True)
CACHE={}
tf=sorted(os.listdir(R+'test/test_images'))[:{0:None,1:5,2:2}[SMOKE]]
out=[]
for f in tf:
    lab,k=labels(dec_inst(infer(R+'test/test_images/'+f)),TH,MINA)
    for j in range(1,k+1):
        r_=mu.encode(np.asfortranarray((lab==j).astype(np.uint8))); out.append((f[:-5]+f'_{j}',r_['counts'].decode()))
pd.DataFrame(out,columns=['filament_id','segmentation_rle']).to_csv(W+'/submission.csv',index=False)
print('wrote',len(out),'rows for',len(tf),'images',round(time.time()-T0),flush=True)
