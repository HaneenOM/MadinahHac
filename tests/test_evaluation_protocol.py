"""Safety gates and a hand-calculated PROJECT summary, not a Kaggle metric test."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import pandas as pd
import evaluation_artifacts as a
from reliable_evaluation import prefix_record,summarize
from t05_forecast.core import load_recordings
from pipeline import table


class EvaluationSafety(unittest.TestCase):
    def test_cache_gate_rejects_tampering_context_and_recipe_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            root=Path(temp);(root/'historical').mkdir()
            blob=root/'historical/model.npz';blob.write_bytes(b'covariance fixture')
            recipe=dict(heldout='r',member_key='m',filename='model.npz',field_sha256='original')
            entry=dict(**recipe,setting='historical',sha256=a.sha(blob),provenance='legacy_not_fully_verified')
            ctx=dict(inputs={'train':'original'},config={'steps':140},gp_code={'gp.py':'original'})
            (root/'manifest.json').write_text(json.dumps(dict(context_at_import=ctx,entries=[entry])))
            with patch.object(a,'ASSETS',root),patch.object(a,'context',return_value=ctx),patch.object(a,'recipes',return_value=[recipe]):
                with self.assertRaisesRegex(ValueError,'provenance'):a.validated_entries('historical')
                self.assertEqual(len(a.validated_entries('historical',True)),1)
                for field in ('inputs','config','gp_code'):
                    changed=copy.deepcopy(ctx);changed[field]['changed']='tampered'
                    with patch.object(a,'context',return_value=changed):
                        with self.assertRaisesRegex(ValueError,'context'):a.validated_entries('historical',True)
                with patch.object(a,'recipes',return_value=[dict(recipe,field_sha256='changed')]):
                    with self.assertRaisesRegex(ValueError,'recipe'):a.validated_entries('historical',True)
                blob.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'bytes'):a.validated_entries('historical',True)

    def test_future_and_masked_target_cannot_enter_forecast_prefix(self):
        rec=load_recordings(ROOT/'data/raw')['20260918_1100']
        test=pd.read_csv(ROOT/'data/raw/test.csv')
        feed=table(test[test.id.isna()])['20260918_1100']
        cases=pd.DataFrame([dict(minute=rec.first+16,unit='cam_A')]);now=rec.first+15
        actual=prefix_record(rec,feed,cases,now)
        altered=feed.copy();altered.loc[altered.index>now]=1e8
        rec.y[:]=1e8
        np.testing.assert_array_equal(actual.y,prefix_record(rec,altered,cases,now).y)
        self.assertFalse(np.isfinite(actual.y[actual.minutes>now]).any())

    def test_project_summary_hand_calculation(self):
        rows=[];labels=[]
        for d,errors in [('day1',[2,4,6,8]),('day2',[1,2,3,4])]:
            for kind,error in zip(['forecast','hidden','hidden-future','missing'],errors):
                key=f'{d}:{kind}'
                rows.append(dict(case_id=key,day=d,kind=kind,camera_group='A-H',unit='cam_A',
                    recording=d,now=1,minute=2,target_id=key,window_id=key,horizon_band='1-5',
                    prediction=10+error,reference=14,model=True,fallback=False))
                labels.append(dict(case_id=key,truth=10))
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            _,aggregate=summarize(pd.DataFrame(rows),pd.DataFrame(labels),Path(temp),'historical')
        # day1=(.5+1+1.5+2)/4=1.25; day2=(.25+.5+.75+1)/4=.625
        self.assertEqual(aggregate,(1.25+.625)/2)

    def test_heldout_counts_cannot_change_training_fields_or_template(self):
        recs=load_recordings(ROOT/'data/raw');heldout='20260918_1100'
        original={e['member_key']:e['field_sha256'] for e in a.recipes('representative',recs) if e['heldout']==heldout}
        tmpl=a.independent_template(recs,heldout,.8,complete_new_cameras=True)
        changed=dict(recs);changed[heldout]=recs[heldout].copy();changed[heldout].y[:]=1e8
        poisoned={e['member_key']:e['field_sha256'] for e in a.recipes('representative',changed) if e['heldout']==heldout}
        self.assertEqual(original,poisoned)
        np.testing.assert_array_equal(tmpl,a.independent_template(changed,heldout,.8,complete_new_cameras=True))


if __name__=='__main__':unittest.main()
