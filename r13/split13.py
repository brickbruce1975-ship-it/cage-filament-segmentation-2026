# r13 fresh holdout: files never used in any earlier selection (not a file of any seed0_holdout100 id, not seed8_holdout100, not r10 inner val).
import sys,json,hashlib,numpy as np; sys.path.insert(0,'/workspace/fil/r11'); from common import *
S0F={fn(i) for i in S0}; USED=S0F|HFILES|IVF
REM=sorted(f for f in F if f not in USED)
N=[]
for j in np.random.RandomState(13).permutation(len(REM)):
    N+=grp[REM[j]]
    if len(N)>=200: break
N=sorted(N); NF=sorted({fn(i) for i in N}); NSHA=hashlib.sha256(json.dumps(N).encode()).hexdigest()
assert not (set(NF)&USED) and not (set(N)&S0)
json.dump(dict(holdout_id='fresh13_holdout200',ids=N,files=NF,sha256=NSHA,used_excluded=dict(S0files=len(S0F),H=len(HFILES),IVF=len(IVF)),remaining_files=len(REM)),open('fresh13_holdout200.json','w'),indent=0)
print(len(N),len(NF),NSHA,'train ids',len(ids)-len(N))
