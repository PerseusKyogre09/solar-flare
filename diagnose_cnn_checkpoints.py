"""Evaluate every CNN-LSTM checkpoint with one fixed, CPU-safe evaluator."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/images/Lat60_Lon60_Nans0_png_224"
OUT = ROOT / "results/cnn_lstm/current_checkpoint_results"
OUT.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["C", "M", "X"]

class SolarSequenceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.class_map = {"C": 0, "M": 1, "X": 2}
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        images = [self.transform(Image.open(DATA_DIR / row[f"frame_{i}"].strip()).convert("L")) for i in range(10)]
        return torch.stack(images), self.class_map[row["target"]]

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
    def forward(self, x): return self.features(x).view(x.size(0), -1)

class LegacyCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
    def forward(self, x): return self.features(x).view(x.size(0), -1)

class CNNLSTM(nn.Module):
    def __init__(self, legacy=False):
        super().__init__()
        self.cnn = LegacyCNNEncoder() if legacy else CNNEncoder()
        self.lstm = nn.LSTM(64,128,num_layers=1,batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,3))
    def forward(self, x):
        b,t,c,h,w=x.shape
        z=self.cnn(x.view(b*t,c,h,w)).view(b,t,-1)
        z,_=self.lstm(z)
        return self.classifier(z[:,-1,:])

def evaluate(checkpoint, loader):
    state=torch.load(ROOT/checkpoint,map_location="cpu",weights_only=True)
    model=CNNLSTM(legacy=("cnn.features.1.running_mean" not in state)).eval()
    model.load_state_dict(state)
    ys=[]; ps=[]
    start=time.time()
    with torch.inference_mode():
        for x,y in loader:
            ps.extend(model(x).argmax(1).numpy().tolist()); ys.extend(y.numpy().tolist())
    cm=confusion_matrix(ys,ps,labels=[0,1,2])
    prec,rec,f1,_=precision_recall_fscore_support(ys,ps,labels=[0,1,2],zero_division=0)
    row={"checkpoint":checkpoint,"accuracy":accuracy_score(ys,ps),
         "macro_precision":prec.mean(),"macro_recall":rec.mean(),"macro_f1":f1.mean(),
         "weighted_f1":precision_recall_fscore_support(ys,ps,labels=[0,1,2],average="weighted",zero_division=0)[2],
         **{f"{c}_precision":float(prec[i]) for i,c in enumerate(CLASS_NAMES)},
         **{f"{c}_recall":float(rec[i]) for i,c in enumerate(CLASS_NAMES)},
         **{f"{c}_f1":float(f1[i]) for i,c in enumerate(CLASS_NAMES)}}
    payload={"checkpoint":checkpoint,"confusion_matrix":cm.tolist(),"metrics":row,"elapsed_seconds":time.time()-start}
    (OUT/(Path(checkpoint).stem+".json")).write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
    return row

if __name__ == "__main__":
    torch.set_num_threads(4)
    ds=SolarSequenceDataset(ROOT/"data/Test_sequences.csv")
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,pin_memory=False)
    rows=[evaluate(name,loader) for name in ["cnn_lstm.pth","cnn_lstm_best.pth","cnn_lstm_final.pth"]]
    pd.DataFrame(rows).to_csv(ROOT/"results/cnn_lstm/checkpoint_comparison.csv",index=False)
