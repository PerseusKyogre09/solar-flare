"""Build a compact memory-mapped cache for the strict matched sequences.

The cache preserves CSV row order and stores uint8 grayscale pixels, avoiding
repeated PNG decoding during every training epoch.
"""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path("results/matched_experiment")
IMG_ROOT=Path("data/images/Lat60_Lon60_Nans0_png_224")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="results/image_cache"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); summary={}
    for split in ["Train","Validation","Test"]:
        df=pd.read_csv(ROOT/f"{split.lower()}_all_label_sequences.csv")
        path=out/f"{split.lower()}_images.uint8.memmap"
        mm=np.memmap(path,dtype=np.uint8,mode="w+",shape=(len(df),10,224,224))
        for n,(_,row) in enumerate(df.iterrows()):
            for j in range(10):
                p=IMG_ROOT/row[f"frame_{j}"]
                with Image.open(p) as im: mm[n,j]=np.asarray(im.convert("L"),dtype=np.uint8)
            if (n+1)%1000==0: print(f"{split}: {n+1}/{len(df)}",flush=True)
        mm.flush(); del mm
        (out/f"{split.lower()}_shape.json").write_text(json.dumps({"shape":[len(df),10,224,224],"dtype":"uint8","csv":str(ROOT/f'{split.lower()}_all_label_sequences.csv')},indent=2))
        summary[split]=len(df)
    (out/"cache_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
