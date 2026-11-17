"""Run the official starter forecast method on the two date-held-out folds."""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from pipeline import ROOT, load, locate_data, pseudo

warnings.filterwarnings('ignore',category=RuntimeWarning)
train,test,_,_=load(locate_data(None))
rows=[]
for rec,feed,targets in pseudo(train,test,None):
    day=train.loc[train.recording.eq(rec),'day'].iloc[0]
    fit=train[train.day.ne(day)]
    def tables(df):
        out={}
        for name,g in df[df.people.notna()].groupby('recording'):
            t=g.pivot_table(index='minute_of_day',columns='unit',values='people')
            out[name]=t.reindex(np.arange(t.index.min(),t.index.max()+1))
        return out
    def features(t,now,H):
        a=t.loc[now-15:now].to_numpy(float)
        last=np.full(a.shape[1],np.nan)
        for row in a[-6:]:
            last=np.where(np.isfinite(row),row,last)
        with np.errstate(all='ignore'):
            mean15=np.nanmean(a,0)
            allmean=np.nanmean(np.log1p(a[-6:]))
        return pd.DataFrame({'log_last':np.log1p(last),'log_mean15':np.log1p(mean15),
                             'log_table':allmean,'horizon':float(H)},index=t.columns)
    X,y=[],[]
    for t in tables(fit).values():
        for now in t.index[20:-5:3]:
            for H in (5,15,30):
                if now+H not in t.index:
                    continue
                f=features(t,now,H)
                target=np.log1p(t.loc[now+H].to_numpy(float))-f.log_last.to_numpy()
                ok=np.isfinite(target)
                X.append(f[ok]);y.append(target[ok])
    X=pd.concat(X);y=np.concatenate(y)
    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,
            learning_rate=.05,random_state=0)
    model.fit(X,y,sample_weight=np.maximum(np.expm1(X.log_last),1.0))
    t=tables(feed)[rec]
    target=targets[targets.kind.eq('forecast')]
    for (now,H),g in target.groupby(['now_minute_of_day','horizon']):
        f=features(t,int(now),int(H)).reindex(g.unit)
        pred=np.expm1(f.log_last.to_numpy()+model.predict(f))
        for truth,reference,p in zip(g.people,np.expm1(f.log_last),pred):
            if np.isfinite(p):
                rows.append((day,rec,truth,max(0.,p),reference))
detail=pd.DataFrame(rows,columns=['fold','recording','truth','starter','reference'])
summary=detail.groupby(['fold','recording']).apply(lambda x: pd.Series({
    'n':len(x),'starter_mae':np.abs(x.truth-x.starter).mean(),
    'reference_mae':np.abs(x.truth-x.reference).mean()}),include_groups=False).reset_index()
summary['normalized_mae']=summary.starter_mae/summary.reference_mae
summary.to_csv(ROOT/'outputs/starter_validation.csv',index=False)
print(summary.to_string(index=False))
