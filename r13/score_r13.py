# Filament r13 scoring (PREREG comps/filament/PREREG_r13.md sha 8c87eab1...). Run ONCE. Frozen r12 recipe fuse(P,0.2,None,0.15,5000) vs r5 config B,
# both from the r13 retrained models, on fresh13_holdout200. Official self-eval port. Model: Grok Bot executor.
import sys,os,json,pickle,time,hashlib,numpy as np
sys.path.insert(0,'/workspace/fil/r11'); sys.path.insert(0,'/workspace/fil/r12')
import r12_fuse as RF
from analyze import labels_B
from common import imgs,grp,fn
from selfeval import get_pq_score
SPL=json.load(open('/workspace/fil/r13/fresh13_holdout200.json')); N=SPL['ids']; NF=SPL['files']
assert hashlib.sha256(json.dumps(N).encode()).hexdigest()=='7637297a7477e09e72b80257c397a7a1df9effe1666ce8a038f3cf7f0b3036fc'
OU,OY=sys.argv[1],sys.argv[2]   # dirs with holdout_probs_r13.npz / holdout_inst_r13.pkl
Q=np.load(OU+'/holdout_probs_r13.npz'); INST=pickle.load(open(OY+'/holdout_inst_r13.pkl','rb'))
assert sorted(Q.files)==NF and sorted(INST)==NF, 'missing holdout files'
yolo_cov=np.mean([len(INST[f])>0 for f in NF])
REC='FROZEN r12 cfg_test (t_agree 0.2, c_add off, c_small 0.15, A_small 5000)'; CFG=(0.2,None,0.15,5000)
rows_c={}; rows_b={}
for f in NF:
    bl,kb=labels_B(Q[f]); P=RF.prep_file(bl,kb,INST[f]); lab,k=RF.fuse(P,*CFG); base=bl.ravel().copy()
    for i in grp[f]:
        rows_c[i]=RF.score_row(i,lab,k); rows_b[i]=RF.score_row(i,base,kb)
pc,cc=get_pq_score([rows_c[i][0] for i in N]); pb,cb=get_pq_score([rows_b[i][0] for i in N])
gain=pc-pb; sanity=(0.30<=pb<=0.50) and yolo_cov>=0.9
res=dict(round='r13',prereg='comps/filament/PREREG_r13.md',prereg_sha='8c87eab178d2684c0952154f1ee95642950a50decbfed76ec328f5d289332fe8',recipe=REC,
  holdout_id='fresh13_holdout200',n_ids=len(N),n_files=len(NF),pq_cand=pc,pq_base=pb,gain=gain,cand_counts=cc,base_counts=cb,yolo_file_coverage=yolo_cov,sanity_ok=sanity,
  gate1_bar=0.015,gate1_ok=bool(sanity and gain>=0.015),rows_cand=[rows_c[i][1] for i in N],rows_base=[rows_b[i][1] for i in N],
  model='Grok Bot executor; r13 retrained r5 U-Nets + r10-recipe YOLO11s-seg; frozen r12 fusion',time=time.strftime('%F %T %Z'),
  true_rsi=False,true_agi=False,counts_as_aha=False,counts_as_rsi_evidence=False)
json.dump(res,open('/workspace/fil/r13/eval_r13.json','w'),indent=1,default=str)
print('R13RESULT',json.dumps({k:v for k,v in res.items() if not k.startswith('rows')},default=str))
