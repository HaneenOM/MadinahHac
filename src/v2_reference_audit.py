"""Check how closely reconstructed references match released sample values."""
import numpy as np
import pandas as pd
from pipeline import ROOT,load,locate_data,table
import v2_hidden
import v2_missing

train,test,sample,_=load(locate_data(None))
models=v2_hidden.fit(table(train))
feed=table(test[test.id.isna()])
ref=sample.set_index('id').people
rows=[]
for r in test[test.id.notna() & test.kind.isin(['hidden','hidden-future','missing'])].itertuples():
    t=feed[r.recording]
    if r.kind=='hidden':
        pred=v2_hidden.predict(t,r.unit,int(r.minute_of_day),models,'single')[0]
    elif r.kind=='hidden-future':
        pred=v2_hidden.predict(t.loc[:int(r.now_minute_of_day)],r.unit,int(r.now_minute_of_day),models,'single')[0]
    else:
        pred=v2_missing.predict(t,r.unit,int(r.minute_of_day),{},'linear')[0]
    rows.append((r.kind,r.unit,pred,ref[r.id]))
x=pd.DataFrame(rows,columns=['kind','unit','reconstruction','sample'])
x['difference']=(x.reconstruction-x['sample']).abs()
summary=x.groupby(['kind','unit']).agg(n=('difference','size'),mean_abs_difference=('difference','mean'),
                                       max_abs_difference=('difference','max')).reset_index()
summary.to_csv(ROOT/'outputs/v2_reference_audit.csv',index=False)
print(summary.to_string(index=False))
