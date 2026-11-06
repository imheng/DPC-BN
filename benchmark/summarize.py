"""Summarize the complete rerun and regenerate aggregate comparison figures."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader(a.results.open()))
 rows=[r for r in rows if not r['duplicate_of'] and r['dataset'].lower()!='jain.mat']
 names=sorted({r['dataset'] for r in rows});lookup={(r['dataset'],int(r['pct']),r['method']):float(r['ari']) for r in rows}
 def values(method,pct):return np.array([lookup[n,pct,method] for n in names])
 stats={}
 for pct in [1,2,3]:
  x=values('DPC',pct);y=values('DPC_BN',pct);d=y-x;wins=int((d>1e-12).sum());losses=int((d< -1e-12).sum())
  stats[pct]=dict(n=len(names),mean_dpc=float(x.mean()),mean_bn=float(y.mean()),mean_delta=float(d.mean()),median_delta=float(np.median(d)),wins=wins,ties=len(names)-wins-losses,losses=losses,sign_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.)
 means={m:float(values(m,2).mean()) for m in ['DPC','centers_only','assignment_only','DPC_BN']}
 (a.output/'summary.json').write_text(json.dumps(dict(sensitivity=stats,ablation=means),indent=2))
 x=values('DPC',2);y=values('DPC_BN',2);d=np.sort(y-x)
 fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].scatter(x,y,s=16);ax[0].plot([min(x.min(),y.min()),1],[min(x.min(),y.min()),1],'--',color='gray');ax[0].set(xlabel='DPC ARI',ylabel='DPC-BN ARI');ax[1].bar(np.arange(len(d)),d,color=np.where(d>=0,'#0072B2','#D55E00'));ax[1].set(xlabel='Dataset rank by improvement',ylabel='Paired ARI change');fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(a.output/f'overview.{ext}',dpi=300)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Original','Centers only','Assignment only','Both'],list(means.values()),color=['gray','#D55E00','#009E73','#0072B2']);ax.set_ylabel(f'Mean ARI ({len(names)} datasets)');fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(a.output/f'ablation.{ext}',dpi=300)
 plt.close(fig);print(json.dumps(stats[2]));print(means)
if __name__=='__main__':main()
