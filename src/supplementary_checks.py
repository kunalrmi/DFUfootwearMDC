"""Supplementary analyses reported in the paper: non-trivial thresholds, corrected t-tests on the learning curve,
and the share of the oracle gain explained by the coherence constraints.
Usage: PAKDD_DATA=<csv> python3 -I supplementary_checks.py <dir with common.py> <outdir>"""
import sys,os,json,pickle,glob; sys.path.insert(0, sys.argv[1]); OUT=sys.argv[2]
import numpy as np, pandas as pd
from scipy.stats import t as tdist, binomtest, chi2_contingency
from common import TGT, K, m, n, Y as Y0, NONE, ROCK, FFD, FFM, DOMIDX
reps=[pickle.load(open(p,'rb')) for p in sorted(glob.glob(f'{OUT}/rep[0-9]*.pkl'))]
Y=reps[0]['Y']; assert (Y==Y0).all()
nt=json.load(open(f'{OUT}/nontrivial.json'))
print({k:len(v) for k,v in nt.items() if k!='ba'}, 'same 0.95/0.98:', nt['0.95']==nt['0.98'])
for thr in ['0.9','0.95','0.98']:
    NT=[TGT.index(c) for c in nt[thr]]
    for L in ['LR','GBT']:
        e=[np.mean([((r['store'][(L,S)]['Y'][:,NT]!=Y[:,NT]).sum(1)==0).mean() for r in reps]) for S in ['S1','S3']]
        print('thr',thr,L,'S1 vs S3 em_nt', np.round(100*np.array(e),2))
# learning curve per-fold test
lc=pd.DataFrame(sum([r['lc'] for r in reps],[]))
for L in ['LR','GBT']:
    for fr in [0.25,0.5,0.75]:
        a=lc[(lc.learner==L)&(lc.frac==fr)]
        p=a.pivot_table(index=['rep','fold'],columns='strategy',values=['n_em','n'])
        d=(p[('n_em','S3')]-p[('n_em','S1')])/p[('n','S1')]; k=len(d)
        ntr=390*fr; ratio=97/ntr
        tt=d.mean()/np.sqrt((1/k+ratio)*d.var(ddof=1)); pt=2*tdist.sf(abs(tt),k-1)
        tt0=d.mean()/np.sqrt((1/k)*d.var(ddof=1)); pt0=2*tdist.sf(abs(tt0),k-1)
        print(L,fr,'mean diff pts',round(100*d.mean(),2),'folds S3>S1',(d>0).sum(),'<',(d<0).sum(),'corr t p',round(pt,3),'naive t p',round(pt0,4))
# corrected t at full size for S3 vs S1 printed in comparisons; also SubEM at 25%
# constraint-adjusted oracle on outer test: BR renormalised to coherent classes given true predecessors
def ll(P,y): return -np.log(np.clip(P[np.arange(len(y)),y],1e-15,1))
for L in ['LR','GBT']:
    tot={'br':0,'brc':0,'teach':0}; per={}
    for r in reps:
        Pb=r['store'][(L,'S1')]['P']; T=r['chain_aux'][L]['teacher']
        for j in range(m):
            pb=Pb[j].copy()
            if j in ROCK[1:] or j==FFM:
                par = ROCK[0] if j in ROCK[1:] else FFD
                pn = Y[:,par]==NONE[par]
                mask=np.ones_like(pb,bool); mask[pn]=False; mask[pn,NONE[j]]=True; mask[~pn,NONE[j]]=False
                pb=np.where(mask,pb,0); pb=pb/np.clip(pb.sum(1,keepdims=True),1e-15,None)
            a,b,c=ll(Pb[j],Y[:,j]).mean(),ll(pb,Y[:,j]).mean(),ll(T[j],Y[:,j]).mean()
            per.setdefault(TGT[j],np.zeros(3)); per[TGT[j]]+=np.array([a,b,c])/len(reps)
    per=pd.DataFrame(per,index=['BR','BR|C','teacher']).T
    G=(per.BR-per.teacher).sum(); Gc=(per['BR|C']-per.teacher).sum()
    print(L,'outer-test teacher G',round(G,3),'G beyond constraints',round(Gc,3),'share explained by constraints',round(1-Gc/G,3))
    print((per.BR-per.teacher).round(3)[lambda s:s.abs()>0.03].to_dict(), (per['BR|C']-per.teacher).round(3)[lambda s:s.abs()>0.03].to_dict())
# dataset descriptives
vec=pd.Series(['|'.join(map(str,r)) for r in Y]); vc=vec.value_counts(); print('distinct',len(vc),'once',(vc==1).sum())
def V(a,b):
    ct=pd.crosstab(a,b); 
    if min(ct.shape)<2: return 0
    c2=chi2_contingency(ct,correction=False)[0]; return np.sqrt(c2/(len(a)*(min(ct.shape)-1)))
vs=[V(Y[:,i],Y[:,j]) for i in range(m) for j in range(i+1,m)]; vs=np.array(vs)
print('pairs',len(vs),'median',round(np.median(vs),3),'>0.3',(vs>0.3).sum(),'>0.5',(vs>0.5).sum())
print('rocker none', (Y[:,ROCK[0]]==NONE[ROCK[0]]).sum())
for dn,idx in DOMIDX.items(): print(dn, len(np.unique(Y[:,idx],axis=0)))
