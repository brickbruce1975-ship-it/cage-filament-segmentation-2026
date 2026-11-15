# r13 test submission (only used if r13 gates PASS): frozen r12 recipe fuse(P,0.2,None,0.15,5000) on the EXISTING full models' cached test outputs
# (/dev/shm/fil_r11/test: r8 v3 base_mask+hull config-B instances + r10 YOLO instances). Model: Grok Bot executor.
import sys,os,pickle,numpy as np,pandas as pd
sys.path.insert(0,'/workspace/fil/r11'); sys.path.insert(0,'/workspace/fil/r12')
import r12_fuse as RF
from pycocotools import mask as mu
CT='/dev/shm/fil_r11/test/'; TD='/workspace/fil/MAGFiLO_1.0_Kaggle_2026/test/test_images/'
TF=sorted(os.listdir(TD)); assert len(TF)==180 and all(os.path.exists(CT+f+'.pkl') for f in TF),'test cache incomplete'
out=[]; nb=na=0
for f in TF:
    D=pickle.load(open(CT+f+'.pkl','rb')); lab=np.zeros((2048,2048),np.int32)
    for j,r in enumerate(D['base'],1): lab[mu.decode(r)>0]=j
    P=RF.prep_file(lab,len(D['base']),D['inst']); L,k=RF.fuse(P,0.2,None,0.15,5000); L=L.reshape(2048,2048); nb+=len(D['base']); na+=k
    for j in range(1,k+1):
        bm=L==j
        if bm.any(): r=mu.encode(np.asfortranarray(bm.astype(np.uint8))); out.append((f[:-5]+f'_{j}',r['counts'].decode()))
pd.DataFrame(out,columns=['filament_id','segmentation_rle']).to_csv('/workspace/fil/r13/submission.csv',index=False)
print('wrote',len(out),'rows; base instances',nb,'fused',na)
