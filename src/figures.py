"""Result figures (drawn at printed size: full LNCS text width = 4.8 in, 7 pt).
  fig_clinical_strategies.pdf : clinical data, exact-match difference from S1 (95% corrected t interval) per strategy, one panel per learner
  fig_dependence.pdf          : clinical data, the chain's teacher-forced gain split into decoder / implied given parent / residual
  fig_bench_heatmap.pdf       : benchmarks, exact-match difference from S1 per dataset x strategy, one panel per learner
  fig_bench_dependence.pdf    : benchmarks, relative dependence gain next to the chain's exact-match gain, datasets sorted by dependence
Usage: python3 figures.py <out_lev dir> <bench agg prefix> <out dir>"""
import sys, json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
LEV, AGG, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42, 'axes.linewidth': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2})
col = {'LR': '#2b6cb0', 'GBT': '#c05621', 'TFM': '#2f855a', 'RF': '#805ad5', 'MLP': '#4a5568'}
NAME = {'S2': 'S2 stacking', 'S3': 'S3 chain', 'S4': 'S4 ens. chains', 'S5': 'S5 powerset', 'S6': 'S6 multi-output'}
def clean(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', metadata={'CreationDate': None, 'Creator': None, 'Producer': None}); fig.savefig(f'{OUT}/{name}.png', dpi=250); plt.close(fig)

# ---------- 1. clinical results by strategy ----------
ci = pd.read_csv(f'{LEV}/table_ci_corrected.csv'); ci = ci[(ci.measure == 'EM') & ~ci.strategy.str.contains('greedy')]
cp = pd.read_csv(f'{LEV}/table_comparisons.csv'); mn = pd.read_csv(f'{LEV}/table_main.csv')
Ls = ['LR', 'GBT', 'TFM', 'RF', 'MLP']; Ss = ['S2', 'S3', 'S4', 'S5', 'S6']
fig, axs = plt.subplots(1, 5, figsize=(4.8, 1.75), sharey=True, sharex=True)
for ax, L in zip(axs, Ls):
    ax.axvline(0, color='#999', lw=0.6, zorder=0)
    for i, S in enumerate(Ss):
        r = ci[(ci.learner == L) & (ci.strategy == S)]
        if r.empty:
            ax.text(0, i, 'n/a', ha='center', va='center', fontsize=5.5, color='#aaa'); continue
        r = r.iloc[0]; sig = cp[(cp.learner == L) & (cp.strategy == S)].EM_t_holm.values[0] < 0.05
        ax.plot([r.lo, r.hi], [i, i], color=col[L], lw=1.0, solid_capstyle='butt')
        ax.plot(r.diff_pts, i, marker='o', ms=3.6, color=col[L] if sig else 'white', mec=col[L], mew=0.9)
    s1 = 100 * mn[(mn.learner == L) & (mn.strategy == 'S1')].em.values[0]
    ax.set_title(f'{L}\n(S1: {s1:.1f}%)', fontsize=6.5, pad=3); clean(ax); ax.set_xlim(-19, 9); ax.set_xticks([-15, -10, -5, 0, 5]); ax.set_xticklabels(['$-$15', '', '$-$5', '0', '5'])
axs[0].set_yticks(range(len(Ss))); axs[0].set_yticklabels([NAME[S] for S in Ss]); axs[0].invert_yaxis()
fig.supxlabel('Exact match: difference from S1 of the same learner (points, 95% interval)', fontsize=6.5, y=0.02)
fig.tight_layout(pad=0.25, w_pad=0.4, rect=(0, 0.04, 1, 1)); save(fig, 'fig_clinical_strategies')

# ---------- 2. clinical: where the chain's gain lies ----------
R = json.load(open(f'{LEV}/dependence_equivalence.json'))['dependence_test_folds']
fig, ax = plt.subplots(figsize=(3.4, 1.15))
seg = [('recovered by the coherent decoder', '#4a5568'), ('implied by constraints given the true parent', '#a0aec0'), ('residual dependence', '#ed8936')]
for i, L in enumerate(['LR', 'GBT', 'TFM']):
    G = R[L]['G']; d = R[L]['share_deployed']; o = R[L]['share_oracle']; left = 0
    for (lab, c), v in zip(seg, [d * G, (o - d) * G, (1 - o) * G]):
        ax.barh(i, v, left=left, height=0.6, color=c, lw=0)
        ax.text(left + v / 2, i, f'{100 * v / G:.0f}%', ha='center', va='center', fontsize=5.8, color='#222' if c == '#a0aec0' else 'white')
        left += v
    ax.text(left + 0.02, i, f'{G:.2f}', va='center', fontsize=5.8, color='#444')
ax.set_yticks(range(3)); ax.set_yticklabels(['LR', 'GBT', 'TFM']); ax.invert_yaxis(); ax.set_xlim(0, 1.45)
ax.set_xlabel("Chain's teacher-forced gain over BR (nats per record)"); clean(ax)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in seg], [l for l, _ in seg], frameon=False, fontsize=5.8, loc='lower left',
          bbox_to_anchor=(0, 1.0), ncol=2, handlelength=0.9, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
fig.tight_layout(pad=0.25); save(fig, 'fig_dependence')

# ---------- 3. benchmarks by strategy (heat map) ----------
t = pd.read_csv(f'{AGG}_table.csv')
order = t.groupby('dataset').n.first().sort_values().index.tolist()
Lb = [L for L in ['LR', 'GBT', 'RF', 'TFM'] if L in set(t.learner)]; Sb = ['S2', 'S3', 'S4', 'S5']
norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5); cmap = plt.get_cmap('RdBu')
fig, axs = plt.subplots(1, len(Lb), figsize=(4.8, 2.1), sharey=True)
axs = np.atleast_1d(axs)
for ax, L in zip(axs, Lb):
    M = np.array([[t[(t.dataset == d) & (t.learner == L) & (t.strategy == S)].d_em.values[0] for S in Sb] for d in order])
    P = np.array([[t[(t.dataset == d) & (t.learner == L) & (t.strategy == S)].p.values[0] for S in Sb] for d in order])
    ax.imshow(np.clip(M, -5, 5), cmap=cmap, norm=norm, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; txt = f'{v:+.1f}' if abs(v) >= 0.05 else '0.0'
            ax.text(j, i, txt + ('*' if P[i, j] < 0.05 else ''), ha='center', va='center', fontsize=5.5, color='white' if abs(v) > 3.5 else '#222')
    ax.set_xticks(range(len(Sb))); ax.set_xticklabels(Sb); ax.tick_params(length=0); ax.set_title(L, fontsize=6.5, pad=3)
    for s in ax.spines.values(): s.set_visible(False)
nd = t.groupby('dataset')[['n', 'dims']].first()
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([f'{d} ({nd.loc[d, "n"]}, {nd.loc[d, "dims"]})' for d in order], fontsize=6)
fig.supxlabel('Exact match: difference from S1 of the same learner (points; * p < 0.05 on that dataset, uncorrected)', fontsize=6.3, y=0.02)
fig.tight_layout(pad=0.25, w_pad=0.3, rect=(0, 0.05, 1, 1)); save(fig, 'fig_bench_heatmap')

# ---------- 4. benchmarks: chain gain with datasets ordered by dependence gain (lollipops) ----------
q = t[t.strategy == 'S3'].copy(); q['dep'] = 100 * q.G_rel
SH = {'Edm': 'Edm', 'Flare1': 'Flare1', 'Oes97': 'Oes97', 'Jura': 'Jura', 'Enb': 'Enb', 'Song': 'Song', 'WQanimals': 'WQanim.', 'WQplants': 'WQplants', 'BeLaE': 'BeLaE'}
nr = 2 if len(Lb) > 3 else 1; nc = int(np.ceil(len(Lb) / nr))
fig, axs = plt.subplots(nr, nc, figsize=(4.8, 1.85 * nr), sharey=True)
axs = np.atleast_1d(axs).ravel()
for ax, L in zip(axs, Lb):
    g = q[q.learner == L].sort_values('dep').reset_index(drop=True); x = np.arange(len(g))
    ax.axhline(0, color='#999', lw=0.5, zorder=0)
    ax.vlines(x, 0, g.d_em, color=col[L], lw=1.2); ax.scatter(x, g.d_em, s=12, color=col[L], zorder=3)
    for xi, v in zip(x, g.dep):
        ax.text(xi, -2.55, (f'{v:.0f}' if abs(v) >= 10 else f'{v:.1f}'), ha='center', va='center', fontsize=4.9, color='#666')
    ax.set_xticks(x); ax.set_xticklabels([SH[d] for d in g.dataset], rotation=55, ha='right', fontsize=5.6)
    ax.set_ylim(-2.9, 5.4); ax.set_xlim(-0.6, len(g) - 0.4); clean(ax); ax.tick_params(axis='x', length=0)
    rho = spearmanr(g.dep, g.d_em).statistic; ax.set_title(f'{L} ($\\rho$ = {rho:.2f})', fontsize=6.5, pad=3)
for k in range(0, len(axs), nc): axs[k].set_ylabel('Chain $-$ S1, EM (pts)')
axs[0].text(-0.5, -2.2, 'dependence gain (%):', fontsize=4.9, color='#666', ha='left', va='bottom')
rho_all = spearmanr(q.dep, q.d_em)
fig.supxlabel(f"Datasets ordered by dependence gain; all {len(q)} pairs: Spearman $\\rho$ = {rho_all.statistic:.2f}, $p$ {'< 0.001' if rho_all.pvalue < 0.001 else '= ' + format(rho_all.pvalue, '.3f')}", fontsize=6.3, y=0.01)
fig.tight_layout(pad=0.25, w_pad=0.8, h_pad=0.6, rect=(0, 0.03, 1, 1)); save(fig, 'fig_bench_dependence')
print('figures written to', OUT)
