"""Reproducible publication analysis for the solar-flare benchmark.

All test curves and uncertainty estimates in this module consume paired saved
predictions. Thresholds are selected from validation rows only.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
    matthews_corrcoef)
from src.xgb_baseline.data import load_dataset
from src.xgb_baseline.metrics import comprehensive_binary_metrics, select_validation_threshold

ROOT=Path(__file__).parent; DIAG=ROOT/'diagrams'; RES=ROOT/'results'; BOOT=2000; SEED=42
MODELS={'xgboost':'experiments/xgboost','logistic_regression':'experiments/xgboost',
        'histgradientboosting':'experiments/histgb','extratrees':'experiments/extratrees'}
DISPLAY={'xgboost':'XGBoost','logistic_regression':'Logistic Regression','histgradientboosting':'HistGradientBoosting','extratrees':'Extra Trees'}

def json_clean(x):
    if isinstance(x, dict): return {k:json_clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [json_clean(v) for v in x]
    if isinstance(x, (np.floating,float)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)): return int(x)
    return x

def bootstrap(y,p,t,metric,seed=SEED,n=BOOT):
    rng=np.random.default_rng(seed); vals=[]; y=np.asarray(y); p=np.asarray(p)
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y)); yy=y[ix]; pp=p[ix]
        if metric in ('roc_auc','pr_auc') and np.unique(yy).size<2: continue
        try:
            if metric=='roc_auc': v=roc_auc_score(yy,pp)
            elif metric=='pr_auc': v=average_precision_score(yy,pp)
            else: v=comprehensive_binary_metrics(yy,pp,t)[metric]
            if np.isfinite(v): vals.append(v)
        except (ValueError,ZeroDivisionError): pass
    if not vals: return [None,None]
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

def fit_missing_predictions(target,model_name,bundle):
    out=Path(MODELS[model_name])/target; pred=out/'predictions'; pred.mkdir(parents=True,exist_ok=True)
    model=joblib.load(out/'model.joblib')
    df=bundle.frame; cols=bundle.feature_names
    rows=[]
    for split in ('Validation','Test'):
        part=df[df.split.eq(split)].copy(); prob=model.predict_proba(part[cols])[:,1]
        z=part[['row_id','filename','active_region','timestamp','split','flare','target']].copy(); z['probability']=prob
        z.to_csv(pred/f'{split.lower()}_{model_name}.csv',index=False); rows.append(z)
    return rows

def load_or_make(target,model_name,bundle):
    if model_name in ('xgboost','logistic_regression'):
        base=Path(MODELS[model_name])/target/'predictions'
        v=pd.read_csv(base/f'validation_{model_name}.csv'); t=pd.read_csv(base/f'test_{model_name}.csv')
        return v,t
    return fit_missing_predictions(target,model_name,bundle)

def plot_save(fig,name):
    DIAG.mkdir(exist_ok=True); fig.savefig(DIAG/f'{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(DIAG/f'{name}.pdf',bbox_inches='tight'); fig.savefig(DIAG/f'{name}.svg',bbox_inches='tight'); plt.close(fig)

def main():
    for d in (DIAG,RES,RES/'statistical_uncertainty',RES/'statistical_comparisons',RES/'threshold_sensitivity',RES/'calibration',RES/'feature_importance',RES/'literature_comparison'): d.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'src/xgb_baseline/config.json').read_text()); results=[]; pred_store={}
    for target in ('ge_c','ge_m'):
        bundle=load_dataset(cfg['features'],cfg['labels'],cfg['split_dir'],target)
        for mn in MODELS:
            v,t=load_or_make(target,mn,bundle); thresholds=cfg['thresholds']; selected=select_validation_threshold(v.target,v.probability,thresholds)
            vm=comprehensive_binary_metrics(v.target,v.probability,selected); tm=comprehensive_binary_metrics(t.target,t.probability,selected)
            ci={m:bootstrap(t.target,t.probability,selected,m) for m in ('roc_auc','pr_auc','tss','hss','csi','mcc','f1','precision','pod_recall_tpr','specificity','far','brier_score')}
            payload={'model':mn,'target':target,'selected_threshold':selected,'validation_tss':vm['tss'],'test':tm,'test_ci_95':ci,'bootstrap':{'resamples':BOOT,'seed':SEED,'method':'paired nonparametric rows percentile interval'}}
            out=Path(MODELS[mn])/target/'metrics'; out.mkdir(parents=True,exist_ok=True); (out/'comprehensive_metrics.json').write_text(json.dumps(json_clean(payload),indent=2)+'\n')
            t=t.copy(); t['predicted_label']=(t.probability>=selected).astype(int); t['selected_threshold']=selected; t.to_csv(Path(MODELS[mn])/target/'predictions'/'test_predictions.csv',index=False)
            pred_store[(target,mn)]=t; results.append({'model':DISPLAY[mn],'target':target,**{k:tm.get(k) for k in ('roc_auc','pr_auc','tss','hss','csi','mcc','f1','precision','pod_recall_tpr','specificity','far','brier_score')},'threshold':selected,'roc_auc_ci':ci['roc_auc'],'pr_auc_ci':ci['pr_auc'],'tss_ci':ci['tss']})
    pd.DataFrame(results).to_csv(RES/'results_final.csv',index=False)
    make_curves(pred_store); make_threshold(pred_store); make_calibration(pred_store); make_class_distribution(); make_importance(); make_report(results)

def make_curves(ps):
    colors={'xgboost':'#1b4f72','logistic_regression':'#7d3c98','histgradientboosting':'#b9770e','extratrees':'#117864'}
    for kind in ('roc','pr'):
        fig,ax=plt.subplots(1,2,figsize=(7.0,3.2),sharey=(kind=='roc'))
        for j,target in enumerate(('ge_c','ge_m')):
            for mn in MODELS:
                d=ps[(target,mn)]; y=d.target; p=d.probability
                if kind=='roc': x,yv,_=roc_curve(y,p); ax[j].plot(x,yv,label=f"{DISPLAY[mn]} (AUC={roc_auc_score(y,p):.3f})",color=colors[mn],lw=1.5)
                else: pr,re,_=precision_recall_curve(y,p); ax[j].plot(re,pr,label=f"{DISPLAY[mn]} (AP={average_precision_score(y,p):.3f})",color=colors[mn],lw=1.5)
            if kind=='roc': ax[j].plot([0,1],[0,1],'k--',lw=.8); ax[j].set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'({chr(97+j)}) {target}')
            else: ax[j].axhline(ps[(target,'xgboost')].target.mean(),color='k',ls='--',lw=.8); ax[j].set(xlabel='Recall',ylabel='Precision',title=f'({chr(97+j)}) {target}')
            ax[j].legend(fontsize=6,frameon=False); ax[j].grid(alpha=.2)
        plot_save(fig,'figure4_roc_curves' if kind=='roc' else 'figure5_pr_curves')

def make_threshold(ps):
    d=ps[('ge_m','xgboost')]; rows=[]
    for t in (.05,.10,.20,.30,.50):
        m=comprehensive_binary_metrics(d.target,d.probability,t); rows.append({'threshold':t,'recall':m['pod_recall_tpr'],'precision':m['precision'],'far':m['far'],'tss':m['tss'],'f1':m['f1']})
    pd.DataFrame(rows).to_csv(RES/'threshold_sensitivity/threshold_sensitivity_xgboost_ge_m.csv',index=False)
    fig,ax=plt.subplots(figsize=(5.5,3.2)); q=pd.DataFrame(rows)
    for c in q.columns[1:]: ax.plot(q.threshold,q[c],marker='o',label=c.upper() if c=='far' else c.title())
    ax.set(xlabel='Decision threshold',ylabel='Metric value',title='XGBoost ge_m threshold sensitivity'); ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=7,frameon=False); plot_save(fig,'figure6_threshold_sensitivity')

def make_calibration(ps):
    rows=[]; fig,ax=plt.subplots(1,2,figsize=(7,3.2),sharey=True)
    for j,target in enumerate(('ge_c','ge_m')):
        for mn in MODELS:
            d=ps[(target,mn)]; bins=np.linspace(0,1,11); idx=np.digitize(d.probability,bins[1:-1]); prob=[]; obs=[]
            for b in range(10):
                q=idx==b; prob.append(d.probability[q].mean() if q.any() else np.nan); obs.append(d.target[q].mean() if q.any() else np.nan)
            ece=sum((idx==b).mean()*abs(obs[b]-prob[b]) for b in range(10) if np.isfinite(obs[b])); brier=float(np.mean((d.target-d.probability)**2)); rows.append({'model':DISPLAY[mn],'target':target,'brier_score':brier,'ece':ece})
            ax[j].plot(prob,obs,marker='o',ms=3,label=DISPLAY[mn])
        ax[j].plot([0,1],[0,1],'k--',lw=.8); ax[j].set(title=target,xlabel='Mean predicted probability',ylabel='Observed frequency'); ax[j].grid(alpha=.2)
    pd.DataFrame(rows).to_csv(RES/'calibration/calibration_metrics.csv',index=False); ax[1].legend(fontsize=6,frameon=False); plot_save(fig,'figure7_calibration')

def make_class_distribution():
    seq=pd.read_csv(ROOT/'data/Test_sequences.csv'); tab=pd.read_csv(ROOT/'data/Test_Data_by_AR_png_224.csv'); out=[]
    for c in ('C','M','X'): out.append({'task':'CNN-LSTM','class':c,'count':int((seq.target==c).sum())})
    for t in ('ge_c','ge_m'):
        # derive from the checked-in class column (1 denotes qualifying event in the split manifest)
        out.append({'task':t,'class':'positive','count':int(tab['class'].astype(int).sum())})
        out.append({'task':t,'class':'negative','count':int((tab['class'].astype(int)==0).sum())})
    pd.DataFrame(out).to_csv(RES/'class_distribution.csv',index=False)

def make_importance():
    for target in ('ge_c','ge_m'):
        p=ROOT/f'experiments/xgboost/{target}/feature_importance/xgboost_importance.csv'
        if not p.exists(): continue
        d=pd.read_csv(p).sort_values('gain').tail(20); fig,ax=plt.subplots(figsize=(5.8,4.2)); ax.barh(d.feature,d.gain,color='#1b4f72'); ax.set(xlabel='XGBoost gain importance',title=f'XGBoost feature importance: {target}'); ax.tick_params(labelsize=7); plot_save(fig,f'figure8_feature_importance_{target}')

def make_report(results):
    lines=['# Final experimental results','', 'All binary metrics use the predefined active-region-grouped Train/Validation/Test split. Thresholds were selected on Validation over 0.1–0.9 by maximum TSS and then frozen for Test. Bootstrap intervals use 2,000 paired test-row resamples, seed 42, percentile limits.','', '## Binary results','', '| Model | Target | ROC-AUC | PR-AUC | TSS | F1 | Precision | Recall | Threshold |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results: lines.append(f"| {r['model']} | {r['target']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['tss']:.3f} | {r['f1']:.3f} | {r['precision']:.3f} | {r['pod_recall_tpr']:.3f} | {r['threshold']:.1f} |")
    lines += ['', '## CNN-LSTM', 'The repository checkpoint reports multiclass C/M/X evaluation on 2,021 test sequences. Its verified raw confusion matrix is [[1537, 171, 0], [147, 132, 0], [0, 34, 0]]. All 34 X-class examples were misclassified as M; the model identified no X-class event in this held-out test set. This multiclass task is not directly compared with binary tabular accuracy.', '', '## Feasibility notes', 'A valid chronological experiment is not substituted for the benchmark: the existing split manifests have overlapping timestamp ranges and no independently constructed chronological protocol. A matched single-frame/CNN-LSTM+features experiment was not added because the current training code defines separate image multiclass and tabular binary datasets rather than a common target and row-level paired representation. The NASA/SHARP CSV remains separate and was not used for training.', '']
    (ROOT/'RESULTS_FINAL.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
