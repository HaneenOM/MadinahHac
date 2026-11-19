"""Fast, strictly cross-recording I-M trial. Keeps own-camera observations causal."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,time,sys,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'noran/experiments'))
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import prompt56 as session
SELECT='--select' in sys.argv
PREFIX='forecast_im_select' if SELECT else 'forecast_im'
M=session.mod('noran/tracks/p56_forecast_im_select/model.py' if SELECT else 'noran/tracks/p56_forecast_im/model.py','p56_im')
OUT=session.OUT
def output(suffix):return OUT/f'{PREFIX}_{suffix}'

def main():
    session.initialize();session.check_time(900)
    paths=['noran/configs/prompt56_forecast_im_select.json','noran/tracks/p56_forecast_im_select/model.py'] if SELECT else ['noran/configs/prompt56_forecast_im.json','noran/tracks/p56_forecast_im/model.py']
    lock=output('lock.json');hashes={p:session.sha(ROOT/p) for p in paths}
    if lock.exists():assert json.loads(lock.read_text())['hashes']==hashes
    else:lock.write_text(json.dumps(dict(hashes=hashes,locked_time=time.time(),selection='none; fixed recipe before results'),indent=2))
    if output('result.json').exists():print('Reusing completed I-M trial');return
    started=time.perf_counter();train,test,_,_=session.setup()
    released=test[test.id.isna()&test.people.notna()&test.unit.isin(M.UNITS)].copy()
    # Build causal features; truth is stored separately from the model input matrices.
    x,y,meta=M.examples(released)
    allrows=[];fits=[];check_count=failures=0
    with threadpool_limits(limits=2):
        for day,group in meta.groupby('day'):
            fitting=released[released.day.ne(day)].copy();assert not fitting.day.eq(day).any()
            tick=time.perf_counter();model=M.fit(fitting,day,{'test_released':fitting.iloc[:0]})
            fit_seconds=time.perf_counter()-tick
            idx=group.index.to_numpy();tick=time.perf_counter()
            correction=model['model'].predict(x[idx])
            if 'eligible_units' in model:correction=np.where(group.unit.isin(model['eligible_units']),correction,0.)
            values=np.maximum(0,group.v7.to_numpy()+group.scale.to_numpy()*correction)
            part=group.drop(columns='truth').copy();part['trial']=values
            inference=time.perf_counter()-tick
            for r in group.iloc[::25].itertuples():
                frame=released[released.recording.eq(r.recording)].pivot(index='minute_of_day',columns='unit',values='people').sort_index()
                case=SimpleNamespace(unit=r.unit,now=r.now,minute=r.minute)
                original=M.predict(case,frame,model,{})
                frame.loc[frame.index>r.now,:]=1e8
                altered=M.predict(case,frame,model,{})
                np.testing.assert_allclose(original,values[np.flatnonzero(idx==r.Index)[0]],rtol=0,atol=1e-8)
                check_count+=1;failures+=int(not np.isclose(original,altered,rtol=0,atol=1e-8))
            allrows.append(part)
            fits.append(dict(day=day,training_recordings=model['training_recordings'],fit_seconds=fit_seconds,inference_seconds=inference,
                selected_units=model.get('eligible_units'),selection=model.get('selection_audit')))
    pred=pd.concat(allrows).sort_index();pred.to_csv(output('predictions.csv'),index=False)
    pred['truth']=meta.loc[pred.index,'truth']
    for col in ('v7','trial','reference'):pred['e_'+col]=(pred[col]-pred.truth).abs()
    # Released stretches are clusters, not the repeated minute/horizon pairs.
    blocks={}
    for (rec,unit),g in released.groupby(['recording','unit']):
        minutes=np.sort(g.minute_of_day.unique());b=np.cumsum(np.r_[True,np.diff(minutes)>1])
        blocks.update({(rec,unit,int(m)):int(k) for m,k in zip(minutes,b)})
    pred['block']=[blocks[(r.recording,r.unit,r.minute)] for r in pred.itertuples()]
    pred['window']=pred.recording+'|'+pred.unit+'|'+pred.block.astype(str)
    pred['gain']=pred.e_v7-pred.e_trial
    pred['band']=pd.cut(pred.horizon,[0,5,10,15],labels=['1-5','6-10','11-15'])
    for label,keys in dict(recording=['recording','unit'],horizon=['recording','unit','band'],window=['window']).items():
        g=pred.groupby(keys,observed=True).agg(n=('truth','size'),distinct_targets=('minute','nunique'),windows=('window','nunique'),
            v7_mae=('e_v7','mean'),trial_mae=('e_trial','mean'),reference_mae=('e_reference','mean'),gain=('gain','mean'))
        g['v7_ratio']=g.v7_mae/g.reference_mae;g['trial_ratio']=g.trial_mae/g.reference_mae
        g.to_csv(output(f'by_{label}.csv'))
        if label=='recording':print(g.to_string())
    daily=pred.groupby('day')[['e_v7','e_trial','e_reference']].mean();windows=pred.groupby('window').gain.sum().sort_values(ascending=False)
    sensitivity=[]
    for n in (0,1,2):
        rem=pred[~pred.window.isin(windows.head(n).index)]
        sensitivity.append(dict(n=n,mae_gain=float(rem.gain.mean())))
    result=dict(mission='forecast',protocol='released I-M, leave one recording/date out; not fixed/mirror/testday',
        rows=len(pred),distinct_target_readings=len(pred[['recording','unit','minute']].drop_duplicates()),
        windows=len(windows),wins=int((windows>1e-8).sum()),losses=int((windows< -1e-8).sum()),
        v7_mae=float(pred.e_v7.mean()),trial_mae=float(pred.e_trial.mean()),
        v7_ratio=float((daily.e_v7/daily.e_reference).mean()),trial_ratio=float((daily.e_trial/daily.e_reference).mean()),
        sensitivity=sensitivity,checks=check_count,failures=failures,folds=fits,runtime_seconds=time.perf_counter()-started)
    pred.to_csv(output('scored.csv'),index=False)
    output('result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert failures==0;session.initialize()
if __name__=='__main__':main()
