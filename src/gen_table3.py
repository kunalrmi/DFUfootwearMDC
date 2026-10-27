"""Build the learner x strategy matrix (Table 3) from stats.py output.
Usage: python3 gen_table3.py <outdir>  -> prints LaTeX tabular body"""
import sys, pandas as pd
O = sys.argv[1]
mn = pd.read_csv(f'{O}/table_main.csv'); cp = pd.read_csv(f'{O}/table_comparisons.csv')
Ls = ['LR', 'GBT', 'TFM', 'RF', 'MLP']; Ss = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
def get(L, S, col):
    r = mn[(mn.learner == L) & (mn.strategy == S)]
    return None if r.empty else 100 * r[col].values[0]
def sig(L, S, name):
    r = cp[(cp.learner == L) & (cp.strategy == S)]
    return (not r.empty) and r[f'{name}_t_holm'].values[0] < 0.05
def cell(L, S, col, name=None):
    v = get(L, S, col)
    if v is None: return '--' if S == 'S6' else '\\emph{pend.}'
    if S == 'S1' or name is None: return f'{v:.1f}'
    d = round(v, 1) - round(get(L, 'S1', col), 1); s = f'{v:.1f} ({d:+.1f})'.replace('-', '$-$')
    return s + ('$^{\\dagger}$' if sig(L, S, name) else '')
NAMES = {'S1': 'S1 BR', 'S2': 'S2 stacking', 'S3': 'S3 chain', 'S4': 'S4 ens.\\ chains', 'S5': 'S5 powerset', 'S6': 'S6 multi-output'}
def best(L, col):
    vals = {S: get(L, S, col) for S in Ss}; vals = {k: round(v, 1) for k, v in vals.items() if v is not None}; mx = max(vals.values())
    return {k for k, v in vals.items() if v == mx}
out = []
for title, col, name in [('Exact match', 'em', 'EM'), ('Sub-exact match', 'subem', 'SubEM')]:
    out.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\')
    B = {L: best(L, col) for L in Ls}
    for S in Ss:
        cells = []
        for L in Ls:
            c = cell(L, S, col, name)
            if S in B[L]: v, rest = c.split(' ', 1) if ' ' in c else (c, ''); c = f'\\textbf{{{v}}}' + (' ' + rest if rest else '')
            cells.append(c)
        out.append(f'{NAMES[S]} & ' + ' & '.join(cells) + ' \\\\')
    out.append('\\midrule')
print('\n'.join(out[:-1]))
