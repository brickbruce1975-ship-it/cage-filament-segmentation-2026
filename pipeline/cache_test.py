# test-image cache: r5-replay config-B instances (RLE) + r10 YOLO raw instances (conf, RLE). Inference only, no decisions.
import sys,os,pickle,time,numpy as np,cv2,torch
torch.set_num_threads(4); cv2.setNumThreads(2)
sys.path.insert(0,'/workspace/fil/r11'); from analyze import *
import segmentation_models_pytorch as smp
from ultralytics import YOLO
CT='/dev/shm/fil_r11/test/'; os.makedirs(CT,exist_ok=True)
def load(p):
    net=smp.Unet('resnet34',encoder_weights=None,in_channels=1,classes=1); net.load_state_dict(torch.load(p,map_location='cpu')); return net.eval()
BM=load('/workspace/fil/r8/out_f/base_mask.pt'); HU=load('/workspace/fil/r8/out_f/hull.pt')
Y=YOLO('/workspace/fil/r10/out_f/runs/r10/weights/best.pt')
def norm(x): return (torch.tensor(x,dtype=torch.float32)/255.-0.5)/0.25
@torch.no_grad()
def prob(net,g,size=1024):
    a=cv2.resize(g,(size,size),interpolation=cv2.INTER_AREA); x=norm(a[None])[:,None]
    q=(torch.sigmoid(net(x))+torch.sigmoid(net(x.flip(-1))).flip(-1))/2
    return (q[0,0].float().numpy()*255).astype(np.uint8)
TD=R+'test/test_images/'; TF=sorted(os.listdir(TD)); T0=time.time()
for n,f in enumerate(TF):
    out=CT+f+'.pkl'
    if os.path.exists(out): continue
    g=cv2.imread(TD+f,0); assert g.shape==(2048,2048),g.shape
    lab,k=labels_B(np.stack([prob(BM,g),prob(HU,g)],-1))
    base=[mu.encode(np.asfortranarray((lab==j).astype(np.uint8))) for j in range(1,k+1)]
    res=Y.predict(TD+f,imgsz=2048,conf=0.05,iou=0.7,max_det=100,retina_masks=True,device='cpu',verbose=False)[0]
    inst=[]
    if res.masks is not None:
        M=res.masks.data; Cf=res.boxes.conf.numpy()
        for j in np.argsort(-Cf):
            m=(M[j]>0.5).numpy().astype(np.uint8)
            if m.shape!=(2048,2048): m=cv2.resize(m,(2048,2048),interpolation=cv2.INTER_NEAREST)
            inst.append((float(Cf[j]),mu.encode(np.asfortranarray(m))))
    pickle.dump(dict(base=base,inst=inst),open(out+'.tmp','wb')); os.rename(out+'.tmp',out)
    print(n,f,k,len(inst),round(time.time()-T0),flush=True)
print('DONE',len(TF),flush=True)
