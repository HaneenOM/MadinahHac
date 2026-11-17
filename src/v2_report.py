"""Create focused v2 breakdowns from saved validation predictions."""
from pathlib import Path
import numpy as np
import pandas as pd

out=Path(__file__).resolve().parents[1]/'outputs'
raw=pd.read_csv(out/'v2_validation_predictions.csv')
raw['error']=(raw.truth-raw.prediction).abs()
raw['reference_error']=(raw.truth-raw.reference).abs()


def aggregate(frame,groups,name):
    x=frame.groupby(groups,dropna=False,observed=True).agg(
        n=('truth','size'),mae=('error','mean'),
        reference_mae=('reference_error','mean')).reset_index()
    x['normalized_mae']=x.mae/x.reference_mae.replace(0,np.nan)
    x.to_csv(out/name,index=False)


aggregate(raw,['fold','recording','kind','unit','method'],'v2_scores_by_date_camera.csv')
selected=pd.read_csv(out/'v2_selected_validation_predictions.csv')
selected['error']=(selected.truth-selected.candidate).abs()
selected['reference_error']=(selected.truth-selected.reference).abs()
aggregate(selected,['fold','recording','kind','unit'],'v2_selected_scores_by_date_camera.csv')
missing=selected[selected.kind.eq('missing')].copy()
missing['gap_band']=pd.cut(missing.gap_length,[0,5,10,20,30],labels=['1-5','6-10','11-20','21-30'])
missing['donor_band']=pd.cut(missing.donor_available,[-1,0,2,99],labels=['0','1-2','3+'])
aggregate(missing,['fold','gap_band','donor_band'],'v2_selected_scores_missing_gap.csv')
hf=selected[selected.kind.eq('hidden-future')].copy()
hf['horizon_band']=pd.cut(hf.horizon,[0,5,10,15],labels=['1-5','6-10','11-15'])
aggregate(hf,['fold','horizon_band'],'v2_selected_scores_horizon.csv')

for frame,prefix,method_column in [(raw,'v2_hidden_conditions.csv','method'),
                                    (selected,'v2_selected_hidden_conditions.csv',None)]:
    hidden=frame[frame.kind.eq('hidden')].copy()
    hidden['weekday']=pd.to_datetime(hidden.fold).dt.day_name()
    hidden['time_band']=pd.cut(hidden.minute,[0,720,960,1200,1440],
                               labels=['before-noon','12-16','16-20','after-20'])
    hidden['donor_band']=pd.cut(hidden.donor_available,[-1,0,2,99],labels=['0','1-2','3+'])
    groups=['fold','weekday','unit','time_band','donor_band']
    if method_column:
        groups.append(method_column)
    aggregate(hidden,groups,prefix)
