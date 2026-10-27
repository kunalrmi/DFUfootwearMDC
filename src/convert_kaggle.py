"""Merge Kaggle CSV results into the analysis format.
Usage: PAKDD_DATA=<csv> python3 -I convert_kaggle.py <main_out_dir> <kaggle_csv_dir> <new_out_dir>
- LightGBM (predictions_lgbm.csv etc.) REPLACES the scikit-learn GBT rows, oracle gains, learning curve and S5 ceiling.
- PyTorch MLP (predictions_mlp.csv) REPLACES the NumPy MLP rows.
- TabPFN (predictions_tfm.csv, learning_curve_tfm.csv) is ADDED as learner TFM."""
import sys, os, glob, shutil, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import K, m, n, DOMIDX
MAIN, KAG, NEW = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(NEW, exist_ok=True)
shutil.copy(f'{MAIN}/nontrivial.json', NEW)

def stores_from_csv(path, rename=None):
    d = pd.read_csv(path); out = {}
    for (rep, L, S), g in d.groupby(['rep', 'learner', 'strategy']):
        L2 = rename.get(L, L) if rename else L
        s = {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)}
        for j, gj in g.groupby('decision'):
            rows = gj.row.values
            s['P'][j][rows] = gj[[f'p{c}' for c in range(K[j])]].values
            s['Y'][rows, j] = gj.y_dec.values; s['Yraw'][rows, j] = gj.y_raw.values
        cover = g.row.nunique()
        if cover != n: print(f'WARNING: {path} rep {rep} {L} {S} covers {cover}/{n} records')
        out.setdefault(int(rep), {})[(L2, S)] = s
    return out

has = lambda f: os.path.exists(f'{KAG}/{f}')
lgbm = stores_from_csv(f'{KAG}/predictions_lgbm.csv', {'LGBM': 'GBT'}) if has('predictions_lgbm.csv') else None
mlp = stores_from_csv(f'{KAG}/predictions_mlp.csv') if has('predictions_mlp.csv') else None
tfm = stores_from_csv(f'{KAG}/predictions_tfm.csv') if has('predictions_tfm.csv') else None
orc = pd.read_csv(f'{KAG}/oracle_lgbm.csv') if has('oracle_lgbm.csv') else None
lcl = pd.read_csv(f'{KAG}/learning_curve_lgbm.csv') if has('learning_curve_lgbm.csv') else None
lct = pd.read_csv(f'{KAG}/learning_curve_tfm.csv') if has('learning_curve_tfm.csv') else None
unr = pd.read_csv(f'{KAG}/s5_unreachable_lgbm.csv') if has('s5_unreachable_lgbm.csv') else None

for p in sorted(glob.glob(f'{MAIN}/rep[0-9]*.pkl')):
    r = pickle.load(open(p, 'rb')); rep = r['rep']
    if lgbm is not None and rep not in lgbm: print(f'WARNING: no LightGBM results for rep {rep}; keeping scikit-learn GBT')
    if lgbm is not None and rep in lgbm:
        for k in [k for k in r['store'] if k[0] == 'GBT']: del r['store'][k]
        r['store'].update(lgbm[rep])
        r['chain_aux']['GBT'] = {'greedyP': lgbm[rep][('GBT', 'S3-greedy')]['P'], 'teacher': None}
        if orc is not None:
            q = orc[orc.rep == rep]
            r['oracle']['GBT'] = {'g_pre': q.pivot(index='fold', columns='decision', values='g_pre').values,
                                  'g_all': q.pivot(index='fold', columns='decision', values='g_all').values}
        if lcl is not None:
            r['lc'] = [x for x in r['lc'] if x['learner'] != 'GBT'] + lcl[lcl.rep == rep].assign(learner='GBT').to_dict('records')
        if unr is not None:
            r['s5_unreach']['GBT'] = unr[unr.rep == rep].groupby('domain').n.sum().to_dict()
    pickle.dump(r, open(f'{NEW}/rep{rep}.pkl', 'wb'))
    if mlp is not None and rep in mlp:
        pickle.dump(dict(rep=rep, store=mlp[rep]), open(f'{NEW}/extra_mlp_rep{rep}.pkl', 'wb'))
    elif os.path.exists(f'{MAIN}/extra_mlp_rep{rep}.pkl'):
        shutil.copy(f'{MAIN}/extra_mlp_rep{rep}.pkl', NEW)
    if tfm is not None and rep in tfm:
        lc = lct[lct.rep == rep].to_dict('records') if lct is not None else []
        pickle.dump(dict(rep=rep, store=tfm[rep], lc=lc), open(f'{NEW}/extra_tfm_rep{rep}.pkl', 'wb'))
print('merged into', NEW, '| lgbm' if lgbm else '', '| mlp(torch)' if mlp else '', '| tfm' if tfm else '')
