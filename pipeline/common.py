# r11 shared code: data/split identical to r8/r10 (seed8_holdout100 sha asserted; inner val = r10 RandomState(10) split)
import os,json,hashlib,collections,numpy as np,cv2
from pycocotools import mask as mu
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
IV=[]
for j in np.random.RandomState(10).permutation(len(TRF)):
    IV+=grp[TRF[j]]
    if len(IV)>=60: break
IV=sorted(IV); IVF={imgs[i]['file_name'] for i in IV}
fn=lambda i: imgs[i]['file_name']
def gt_layers(i): return [decode(a['segmentation']) for a in anns.get(i,[])]
