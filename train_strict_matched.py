"""Strict matched >=M experiment: binary CNN-LSTM or 29-feature XGBoost.

All outputs are new and live below results/strict_matched/. Historical runs are untouched.
Use --smoke-limit before full training; use --model cnn or --model tabular.
"""
from __future__ import annotations
import argparse, json, os, platform, random, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/matched_experiment")
IMG_ROOT = Path("data/images/Lat60_Lon60_Nans0_png_224")
OUT_ROOT = Path("results/strict_matched")
SEEDS = [11, 12, 34, 42, 56, 69, 90, 93, 104, 702, 1000]
FEATURES = [f"feature_{i:02d}" for i in range(29)]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_frames():
    frames = {s: pd.read_csv(ROOT / f"{s.lower()}_all_label_sequences.csv") for s in ["Train", "Validation", "Test"]}
    full = pd.concat(frames.values(), ignore_index=True)
    assert full.sample_id.is_unique, "duplicate sample_id"
    assert set(full.split) == {"Train", "Validation", "Test"}
    assert full.target_ge_m.isin([0, 1]).all()
    groups = {s: set(frames[s].active_region.astype(str)) for s in frames}
    assert not groups["Train"] & groups["Validation"] and not groups["Train"] & groups["Test"] and not groups["Validation"] & groups["Test"], "active-region overlap"
    for _, r in full.sample(frac=1, random_state=1).head(min(1000, len(full))).iterrows():
        for i in range(10): assert (IMG_ROOT / r[f"frame_{i}"]).is_file(), r[f"frame_{i}"]
    return frames, full

def load_merged():
    frames, full = load_frames()
    feat = pd.read_csv(ROOT / "matched_all_label_endpoint_features.csv")
    assert feat.sample_id.is_unique and len(feat) == len(full)
    keys = ["sample_id", "split", "active_region", "endpoint_filename", "endpoint_timestamp", "target_ge_m"]
    left = full[keys].sort_values("sample_id").reset_index(drop=True)
    right = feat[keys].sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    return frames, full, feat

def select_threshold(y, p):
    best = (0.1, -1.0, -1.0)
    for t in np.arange(0.1, 1.0, 0.01):
        pred = (p >= t).astype(int); pos = y == 1; neg = ~pos
        pod = float(pred[pos].mean()) if pos.any() else 0
        fpr = float(pred[neg].mean()) if neg.any() else 0
        tss = pod - fpr; rec = pod
        if (tss, rec, -t) > (best[1], best[2], -best[0]): best = (float(t), tss, rec)
    return best[0]

def binary_metrics(y, p, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, brier_score_loss
    pred = (p >= threshold).astype(int); cm = confusion_matrix(y, pred, labels=[0,1]); tn, fp, fn, tp = cm.ravel()
    tpr = tp/(tp+fn) if tp+fn else 0; fpr = fp/(fp+tn) if fp+tn else 0
    precision = tp/(tp+fp) if tp+fp else 0
    return {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(np.mean(y)), "threshold": threshold, "roc_auc": float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None, "pr_auc": float(average_precision_score(y,p)) if len(np.unique(y))>1 else None, "pr_auc_baseline": float(np.mean(y)), "tss": float(tpr-fpr), "hss": float(2*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn))) if ((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)) else 0, "csi": float(tp/(tp+fn+fp)) if tp+fn+fp else 0, "mcc": float(matthews_corrcoef(y,pred)), "f1": float(f1_score(y,pred,zero_division=0)), "precision": float(precision), "recall": float(tpr), "specificity": float(1-fpr), "far": float(fp/(tp+fp)) if tp+fp else 0, "brier": float(brier_score_loss(y,p)), "confusion_matrix": cm.tolist()}

def save_predictions(frame, p, threshold, model, seed, split, out):
    d = frame[["sample_id","endpoint_filename","endpoint_timestamp","active_region","split","target_ge_m"]].copy()
    d["probability"] = p; d["predicted_label"] = (p >= threshold).astype(int); d["model"] = model; d["seed"] = seed; d["checkpoint"] = str(out / "best.pt")
    d.to_csv(out / f"predictions_{split.lower()}.csv", index=False)

def tabular(args):
    from sklearn.impute import SimpleImputer
    from xgboost import XGBClassifier
    _, full, feat = load_merged(); seed_all(args.seed)
    if args.smoke_limit:
        full = pd.concat([g.head(args.smoke_limit) for _,g in full.groupby(["split","target_ge_m"], sort=False)]).drop_duplicates("sample_id")
    out = OUT_ROOT / "tabular" / f"seed_{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    df = full.merge(feat[["sample_id"]+FEATURES], on="sample_id", validate="one_to_one")
    tr, va, te = [df[df.split.eq(s)].copy() for s in ["Train","Validation","Test"]]
    imp=SimpleImputer(strategy="median"); Xtr=imp.fit_transform(tr[FEATURES]); Xva=imp.transform(va[FEATURES]); Xte=imp.transform(te[FEATURES])
    model=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective="binary:logistic",eval_metric="logloss",random_state=args.seed,n_jobs=1)
    model.fit(Xtr,tr.target_ge_m,eval_set=[(Xva,va.target_ge_m)],verbose=False)
    pv=model.predict_proba(Xva)[:,1]; threshold=select_threshold(va.target_ge_m.to_numpy(),pv)
    for part,X in [("Validation",Xva),("Test",Xte)]:
        frame=va if part=="Validation" else te; p=model.predict_proba(X)[:,1]; save_predictions(frame,p,threshold,"strict_tabular_xgboost",args.seed,part,out)
        (out/f"metrics_{part.lower()}.json").write_text(json.dumps(binary_metrics(frame.target_ge_m.to_numpy(),p,threshold),indent=2))
    model.save_model(out/"model.json"); import joblib; joblib.dump(imp,out/"imputer.joblib")
    (out/"config.json").write_text(json.dumps({"model":"strict_tabular_xgboost","seed":args.seed,"features":FEATURES,"threshold_source":"validation_max_TSS","smoke_limit":args.smoke_limit},indent=2))
    print(f"TABULAR COMPLETE: {out}")

def cnn(args):
    import torch
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    from PIL import Image
    from torchvision.transforms import ToTensor, Normalize, Compose
    frames, full, _ = load_merged(); seed_all(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke_limit: raise RuntimeError("GPU unavailable; refusing full CNN training")
    class DS(Dataset):
        def __init__(self,df,cache_path=None):
            self.df=df.reset_index(drop=True); self.tf=Compose([ToTensor(),Normalize((.5,),(.5,))]); self.cache=None
            if cache_path:
                self.cache=np.memmap(cache_path,dtype=np.uint8,mode="r",shape=(len(self.df),10,224,224))
        def __len__(self): return len(self.df)
        def __getitem__(self,i):
            r=self.df.iloc[i]
            if self.cache is not None: x=torch.from_numpy(np.asarray(self.cache[i],dtype=np.float32)/255.0).unsqueeze(1); x=(x-.5)/.5
            else: x=torch.stack([self.tf(Image.open(IMG_ROOT/r[f"frame_{j}" ]).convert("L")) for j in range(10)])
            return x,torch.tensor(int(r.target_ge_m))
    if args.smoke_limit: full=pd.concat([g.head(args.smoke_limit) for _,g in full.groupby(["split","target_ge_m"],sort=False)]).drop_duplicates("sample_id")
    tr,va,te=[full[full.split.eq(s)].copy() for s in ["Train","Validation","Test"]]
    cache_dir=Path(args.cache_dir) if args.cache_dir else None
    if cache_dir and args.smoke_limit: print("Cache disabled in smoke mode because row filtering changes cache indices",flush=True); cache_dir=None
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.cnn=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.lstm=nn.LSTM(64,128,batch_first=True); self.fc=nn.Linear(128,2)
        def forward(self,x):
            b,t=x.shape[:2]; z=self.cnn(x.reshape(b*t,1,224,224)).flatten(1).reshape(b,t,64); z,_=self.lstm(z); return self.fc(z[:,-1])
    out=OUT_ROOT/"cnn"/f"seed_{args.seed}"; out.mkdir(parents=True,exist_ok=True); model=Net().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss()
    def cache_for(name,df): return str(cache_dir/f"{name.lower()}_images.uint8.memmap") if cache_dir else None
    loader=DataLoader(DS(tr,cache_for("Train",tr)),batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True)
    t0=time.time(); steps=0; best_tss=-1e9; best_epoch=0; history=[]
    model.eval();
    def predict(df):
        split_name="Validation" if df is va else "Test"; dl=DataLoader(DS(df,cache_for(split_name,df)),batch_size=args.batch_size,shuffle=False,num_workers=0); ps=[]
        with torch.no_grad():
            for x,_ in dl: ps.extend(torch.softmax(model(x.to(device)),1)[:,1].cpu().numpy())
        return np.asarray(ps)
    total_batches = len(loader)
    for epoch in range(args.epochs):
        model.train(); epoch_loss=0.0; epoch_steps=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=lossfn(model(x),y); loss.backward(); opt.step(); epoch_loss += float(loss.detach().cpu()); epoch_steps += 1; steps += 1
            if epoch_steps == 1 or epoch_steps % args.progress_every == 0 or epoch_steps == total_batches:
                done = min(epoch_steps * args.batch_size, len(tr))
                pct = 100.0 * epoch_steps / max(total_batches, 1)
                elapsed_now = time.time() - t0
                eta = elapsed_now / max(epoch_steps, 1) * max(total_batches - epoch_steps, 0)
                print(f"Epoch {epoch + 1}/{args.epochs} | batch {epoch_steps}/{total_batches} ({pct:.1f}%) | sequences {done}/{len(tr)} | loss {float(loss.detach().cpu()):.4f} | elapsed {elapsed_now/60:.1f} min | ETA {eta/60:.1f} min", flush=True)
            if args.smoke_limit and epoch_steps>=args.smoke_steps: break
        model.eval(); pv=predict(va); threshold=select_threshold(va.target_ge_m.to_numpy(),pv); vm=binary_metrics(va.target_ge_m.to_numpy(),pv,threshold)
        history.append({"epoch":epoch+1,"train_loss":epoch_loss/max(epoch_steps,1),"validation_tss":vm["tss"],"validation_threshold":threshold})
        if vm["tss"] > best_tss:
            best_tss=vm["tss"]; best_epoch=epoch+1; torch.save(model.state_dict(),out/"best.pt")
        if args.smoke_limit: break
    model.load_state_dict(torch.load(out/"best.pt",map_location=device,weights_only=True)); model.eval(); elapsed=time.time()-t0
    pv=predict(va); threshold=select_threshold(va.target_ge_m.to_numpy(),pv)
    for name,df in [("Validation",va),("Test",te)]:
        p=predict(df); save_predictions(df,p,threshold,"strict_binary_cnn_lstm",args.seed,name,out); (out/f"metrics_{name.lower()}.json").write_text(json.dumps(binary_metrics(df.target_ge_m.to_numpy(),p,threshold),indent=2))
    (out/"training_history.json").write_text(json.dumps(history,indent=2)); (out/"config.json").write_text(json.dumps({"model":"strict_binary_cnn_lstm","seed":args.seed,"device":str(device),"epochs_requested":args.epochs,"best_epoch":best_epoch,"steps":steps,"seconds":elapsed,"threshold_source":"validation_max_TSS","smoke_limit":args.smoke_limit},indent=2))
    print(f"CNN COMPLETE: {out}; device={device}; seconds={elapsed:.2f}; steps={steps}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=["cnn","tabular"],required=True); p.add_argument("--seed",type=int,default=11); p.add_argument("--smoke-limit",type=int); p.add_argument("--smoke-steps",type=int,default=2); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--epochs",type=int,default=20); p.add_argument("--progress-every",type=int,default=100); p.add_argument("--cache-dir"); a=p.parse_args(); tabular(a) if a.model=="tabular" else cnn(a)
if __name__=="__main__": main()
