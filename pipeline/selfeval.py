# Port of the organizers' self-evaluation notebook (azimahmadzadeh/self-evaluation-notebook), PQ IoU>0.5.
# get_overlap_matrices / fp_count_hit / fn_count_hit / get_pq_score copied verbatim (torch); plus a memory-light exact
# overlap for non-overlapping predicted label maps (verified equal to the verbatim matmul on sample images).
import numpy as np,torch
def get_overlap_matrices(gt_layers,pred_layers):
    n_gt,height,width=gt_layers.shape; n_pred=pred_layers.shape[0]
    gt_flat=gt_layers.reshape(n_gt,height*width); pred_flat=pred_layers.reshape(n_pred,height*width)
    intersection=torch.matmul(gt_flat,pred_flat.t())
    gt_areas=gt_flat.sum(dim=1).view(-1,1); pred_areas=pred_flat.sum(dim=1).view(1,-1)
    union=gt_areas+pred_areas-intersection
    iou_matrix=torch.where(union==0,torch.tensor(0.,device=gt_layers.device),intersection/union)
    dice_matrix=torch.where(union==0,torch.tensor(0.,device=gt_layers.device),2*intersection/(gt_areas+pred_areas))
    return iou_matrix,dice_matrix
def fp_count_hit(hit_matrix): return (hit_matrix.sum(dim=0)==0).sum().item()
def fn_count_hit(hit_matrix): return (hit_matrix.sum(dim=1)==0).sum().item()
def get_pq_score(rows):   # rows: iterable of (iou_matrix, n_gt, n_pred)
    iou_threshold=0.5; tp_iou_scores=[]; fp_count=0; fn_count=0
    for iou_matrix,n_gt,n_pred in rows:
        if n_gt==0: fp_count+=n_pred; continue
        if n_pred==0: fn_count+=n_gt; continue
        hit_matrix=iou_matrix>iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count+=fp_count_hit(hit_matrix); fn_count+=fn_count_hit(hit_matrix)
    tp_count=len(tp_iou_scores); den=tp_count+0.5*fp_count+0.5*fn_count
    return (sum(tp_iou_scores)/den if den>0 else 0.0), dict(tp=tp_count,fp=fp_count,fn=fn_count,sum_iou=sum(tp_iou_scores))
def overlap_fast(G,lab,k):   # G: list of (H,W) uint8 GT layers; lab: int label map 1..k (non-overlapping preds)
    ng=len(G); flat=lab.ravel(); pa=np.bincount(flat,minlength=k+1)[1:].astype(np.float64)
    I=np.zeros((ng,k)); ga=np.zeros(ng)
    for gi,g in enumerate(G):
        gm=g.ravel()>0; ga[gi]=gm.sum(); I[gi]=np.bincount(flat[gm],minlength=k+1)[1:]
    U=ga[:,None]+pa[None,:]-I
    iou=np.where(U==0,0.,I/np.maximum(U,1e-9)); dice=np.where(U==0,0.,2*I/np.maximum(ga[:,None]+pa[None,:],1e-9))
    return torch.tensor(iou,dtype=torch.float32),torch.tensor(dice,dtype=torch.float32),ga,pa
