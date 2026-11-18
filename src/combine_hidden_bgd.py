"""P44: combine existing HIDDEN candidates by full ID, preserving CSV value text."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
OUTPUT=ROOT/'outputs/submission_hidden_bgd_v1.csv'
VERIFY=ROOT/'outputs/submission_hidden_bgd_v1_verification.json'
SOURCES={
    'B':('outputs/submission_catboost_b_only_v1.csv','cam_B','20260918_1100','2026-09-18','Fri',284),
    'G':('outputs/prompt40_friday_hidden/submission_friday_hidden_v1.csv','cam_G','20260918_1100','2026-09-18','Fri',280),
    'D':('outputs/submission_t05_hidden_d_only_v1.csv','cam_D','20260919_1100','2026-09-19','Sat',284),
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def indexed(path):
    rows=read(path);result={r['id']:r['people'] for r in rows}
    assert len(rows)==len(result)==4457 and all(result),f'Invalid IDs: {path}'
    assert all(math.isfinite(float(v)) and float(v)>=0 for v in result.values()),path
    return result

def main():
    assert not OUTPUT.exists() and not VERIFY.exists(),'Never overwrite an existing candidate or verification'
    protected=list((ROOT/'outputs').rglob('submission*.csv'))
    protected += [ROOT/'notebooks/current_baseline.ipynb']+list((ROOT/'src').rglob('*.py'))
    before={str(p):sha(p) for p in protected}
    assert sha(BASE)=='c7e0a0785fee013449b30a1041abdf939a58461bb5f359c5294e69b2c8ec6cd5'
    baseline=indexed(BASE)
    sample_ids=[r['id'] for r in read(ROOT/'data/raw/sample_submission.csv')]
    assert len(sample_ids)==len(set(sample_ids))==4457 and set(sample_ids)==set(baseline)
    target_rows=[r for r in read(ROOT/'data/raw/test.csv') if r['id']]
    targets={r['id']:r for r in target_rows}
    assert len(target_rows)==len(targets)==4457 and set(targets)==set(sample_ids)
    combined=baseline.copy();subsets={};counts={};sources={}
    for camera,(filename,unit,recording,day,weekday,expected_changes) in SOURCES.items():
        path=ROOT/filename;source=indexed(path);assert set(source)==set(sample_ids)
        scope={key for key,r in targets.items() if r['kind']=='hidden' and r['unit']==unit and r['recording']==recording}
        assert scope and all(targets[key]['day']==day and targets[key]['weekday']==weekday for key in scope)
        assert not set().union(*subsets.values()).intersection(scope),'Scopes overlap'
        assert all(source[key]==baseline[key] for key in baseline if key not in scope),f'Source changes outside intended scope: {camera}'
        changes={key for key in scope if source[key]!=baseline[key]}
        assert len(changes)==expected_changes,f'{camera}: expected {expected_changes} changes, found {len(changes)}'
        for key in scope:combined[key]=source[key]
        assert all(combined[key]==source[key] for key in scope)
        subsets[camera]=scope;counts[camera]=len(changes)
        sources[camera]=dict(path=filename,sha256=sha(path),intended_rows=len(scope),changed_rows=len(changes),
            unchanged_in_scope=len(scope)-len(changes),task='hidden',unit=unit,recording=recording,
            selected_subset_matches_source_text=True)
    changed={key for key in combined if combined[key]!=baseline[key]}
    assert len(changed)==848 and sum(counts.values())==848
    assert sum(combined[key]==baseline[key] for key in combined)==3609
    union=set().union(*subsets.values())
    assert changed<=union and all(combined[key]==baseline[key] for key in combined if key not in union)
    assert all(math.isfinite(float(v)) and float(v)>=0 for v in combined.values())
    with OUTPUT.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['id','people']);writer.writeheader()
        writer.writerows(dict(id=key,people=combined[key]) for key in sample_ids)
    written=read(OUTPUT)
    assert [r['id'] for r in written]==sample_ids
    assert indexed(OUTPUT)==combined
    assert all(sha(Path(path))==digest for path,digest in before.items())
    verification=dict(output=str(OUTPUT.relative_to(ROOT)),sha256=sha(OUTPUT),baseline_sha256=sha(BASE),
        ids=4457,sample_order_exact=True,full_id_join=True,test_metadata_checked=True,
        finite_nonnegative=True,changed_by_camera=counts,total_changed=848,baseline_text_identical=3609,
        sources=sources,all_selected_subsets_match_source_text=True,protected_files_unchanged=len(before),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/raw/test.csv',ROOT/'data/raw/sample_submission.csv']},
        source_sha256=sha(Path(__file__)),changed_ids=sorted(changed))
    VERIFY.write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k!='changed_ids'},indent=2))

if __name__=='__main__':main()
