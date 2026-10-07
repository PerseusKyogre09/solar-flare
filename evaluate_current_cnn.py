from pathlib import Path
import json, platform, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from diagnose_cnn_checkpoints import SolarSequenceDataset, CNNLSTM

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/cnn_lstm'
OUT.mkdir(parents=True, exist_ok=True)
ds = SolarSequenceDataset(ROOT / 'data/Test_sequences.csv')
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
model = CNNLSTM().eval()
model.load_state_dict(torch.load(ROOT / 'cnn_lstm_best.pth', map_location='cpu', weights_only=True))
y, pp = [], []
start = time.time()
with torch.inference_mode():
    for x, t in loader:
        pp.append(torch.softmax(model(x), dim=1).numpy())
        y.extend(t.numpy().tolist())
probs = np.concatenate(pp)
y = np.asarray(y)
pred = probs.argmax(1)
names = ['C', 'M', 'X']
cm = confusion_matrix(y, pred, labels=[0, 1, 2])
pr, re, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
metrics = {'model':'CNN-LSTM', 'checkpoint':'cnn_lstm_best.pth', 'n':len(y), 'class_order':names, 'confusion_matrix':cm.tolist(), 'accuracy':accuracy_score(y,pred), 'macro_precision':pr.mean(), 'macro_recall':re.mean(), 'macro_f1':f1.mean(), 'weighted_f1':precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)[2], 'multiclass_mcc':matthews_corrcoef(y,pred)}
metrics['per_class'] = {c:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'support':int(sup[i])} for i,c in enumerate(names)}
aucs, aps, briers = {}, {}, {}
for i, c in enumerate(names):
    yy = (y == i).astype(int)
    aucs[c] = roc_auc_score(yy, probs[:,i])
    aps[c] = average_precision_score(yy, probs[:,i])
    briers[c] = brier_score_loss(yy, probs[:,i])
metrics['probability_metrics'] = {'roc_auc_ovr':aucs, 'macro_roc_auc':roc_auc_score(y,probs,multi_class='ovr',average='macro'), 'average_precision':aps, 'macro_average_precision':float(np.mean(list(aps.values()))), 'brier':briers, 'multiclass_brier':float(np.mean(np.sum((probs-np.eye(3)[y])**2,axis=1)))}
(OUT / 'cnn_lstm_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
raw = pd.read_csv(ROOT / 'data/Test_sequences.csv')
ts = raw['frame_9'].str.rsplit('/', n=1).str[-1].str.extract(r'_(\d{8}_\d{6})_TAI')[0]
out = pd.DataFrame()
out['sequence_id'] = np.arange(len(raw))
out['active_region'] = raw['ar'].values
out['timestamp/end_time'] = ts.values
out['true_class'] = [names[i] for i in y]
out['predicted_class'] = [names[i] for i in pred]
out['prob_C'], out['prob_M'], out['prob_X'] = probs[:,0], probs[:,1], probs[:,2]
out.to_csv(OUT / 'test_predictions.csv', index=False)
(OUT / 'confusion_matrix.json').write_text(json.dumps({'raw':cm.tolist(), 'row_normalized':(cm/cm.sum(axis=1,keepdims=True)).tolist(), 'flows':{'C_to_M':int(cm[0,1]),'M_to_C':int(cm[1,0]),'X_to_C':int(cm[2,0]),'X_to_M':int(cm[2,1])}}, indent=2) + '\n')
env = f'''OS: {platform.platform()}\nKernel: {platform.release()}\nGPU: AMD Radeon RX 9060 XT (PCIe visible; device nodes unavailable)\nROCm utilities: rocminfo/rocm-smi/hipconfig unavailable\nPyTorch: {torch.__version__}\nHIP: {torch.version.hip}\nGPU visible: {torch.cuda.is_available()}\nDevice count: {torch.cuda.device_count()}\nInference device: CPU\nInference seconds: {time.time()-start:.3f}\nLimitation: /dev/kfd and /dev/dri are absent from this environment; GPU acceleration is unavailable because AMD kernel/device interfaces are not exposed.\n'''
(OUT / 'cnn_lstm_environment.txt').write_text(env)
print(json.dumps(metrics, indent=2))
