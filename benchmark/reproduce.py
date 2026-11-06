"""Standalone complete fixed-parameter rerun; Python scientific stack required."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist,pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import bn_core as bn

def baseline(D,rho,C):
 order=np.argsort(-rho,kind='stable');parent=np.full(len(D),-1);delta=np.zeros(len(D))
 for pos,i in enumerate(order):
  if not pos:delta[i]=D[i].max()
  else:
   ids=order[:pos];v=D[i,ids];parent[i]=ids[v==v.min()].min();delta[i]=v.min()
 others=np.array(sorted(order[1:]));c=np.r_[order[0],others[np.argsort(-(rho*delta)[others],kind='stable')[:C-1]]]
 return propagate(order,parent,c),c,order,parent

def propagate(order,parent,c):
 pred=np.full(len(order),-1);pred[c]=np.arange(len(c))
 for i in order:
  if pred[i]<0:pred[i]=pred[parent[i]]
 return pred

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);bn.tests();rows=[];seen={}
 for f in sorted(args.data.glob('*.mat')):
  raw=loadmat(f);X=np.asarray(raw['data'],float);y=raw['label'].ravel()
  if len(X)!=len(y):X=X.T
  key=hashlib.sha256(np.ascontiguousarray(X).tobytes()+np.asarray(y,float).tobytes()).hexdigest();duplicate=seen.get(key,'');seen.setdefault(key,f.name)
  X=StandardScaler().fit_transform(X);D=cdist(X,X);T=bn.mst(D);near=D.copy();np.fill_diagonal(near,np.inf);nn=np.argsort(near,axis=1,kind='stable')[:,:min(5,len(X)-1)].copy();del near;C=len(np.unique(y))
  for p in [1,2,3]:
   dc=float(np.percentile(pdist(X),p));dc=dc if dc>0 else float(np.min(D[D>0])) if np.any(D>0) else 1.
   rho=np.empty(len(X))
   for start in range(0,len(X),256):rho[start:start+256]=np.exp(-(D[start:start+256]/dc)**2).sum(axis=1)-1
   rho=np.maximum(rho,0);base,c,order,parent=baseline(D,rho,C);full,new,peaks,fallback=bn.fit(D,T,rho,C,nn);B=np.column_stack([bn.column(T,int(i)) for i in c]);alloc=np.where(B==B.min(axis=1)[:,None],D[:,c],np.inf).argmin(axis=1);alloc[c]=np.arange(C)
   methods={'DPC':base,'centers_only':propagate(order,parent,new),'assignment_only':alloc,'DPC_BN':full}
   for method,pred in methods.items():rows.append(dict(dataset=f.name,pct=p,method=method,ari=adjusted_rand_score(y,pred),duplicate_of=duplicate,C=C,n=len(y)))
   np.savez_compressed(args.output/f'{f.stem}_p{p}.npz',centers=new,original_centers=c,**methods)
  with (args.output/'results.csv').open('w') as out:
   w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f.name,flush=True)
if __name__=='__main__':main()
