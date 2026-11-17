"""Assert every hidden-future prediction path ignores post-now observations."""
import numpy as np
from pipeline import load,locate_data,table
from v2_experiments import fit_direct,predict_hf
import v2_hidden

train,test,_,_=load(locate_data(None))
tables=table(train)
hidden=v2_hidden.fit(tables)
direct=fit_direct(tables)
feed=table(test[test.id.isna() & test.people.notna()])
n=0
for r in test[test.kind.eq('hidden-future')].itertuples():
    t=feed[r.recording]
    now=int(r.now_minute_of_day)
    h=int(r.minute_of_day)-now
    changed=t.copy()
    changed.loc[changed.index>now]=1e6
    for method in ('fixed','donor_forecast','direct'):
        a=predict_hf(t,r.unit,now,h,hidden,direct,method)
        b=predict_hf(changed,r.unit,now,h,hidden,direct,method)
        assert np.isclose(a,b,equal_nan=True),(r.id,method,a,b)
        n+=1
print('All post-now invariance checks passed:',n)
