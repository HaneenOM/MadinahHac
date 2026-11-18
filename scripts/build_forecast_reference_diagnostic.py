"""P47: DIAGNOSTIC ONLY. Replace exact-kind forecast by sample prediction text.

Full-ID joins only. No training, dependency on third-party packages, or upload.
"""
from pathlib import Path
import csv
from decimal import Decimal
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/submission_hidden_bgd_v1.csv'
OUT=ROOT/'outputs/diagnostics/submission_0800_forecast_reference.csv'
VERIFY=OUT.with_suffix('.verification.json')
NOTE=OUT.with_suffix('.DIAGNOSTIC_ONLY.md')
EXPECTED='342af77dd40e4120b589485ecf02bf25bf3434650e81473692328903bcf67a4a'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def mapping(p):
    source=rows(p);result={r['id']:r['people'] for r in source}
    assert len(source)==len(result)==4457 and all(result),f'Invalid full IDs: {p}'
    return result

def main():
    assert not any(p.exists() for p in (OUT,VERIFY,NOTE)),'Do not overwrite existing diagnostics'
    manifest=json.loads((ROOT/'models/current_baseline/manifest.json').read_text())
    assert sha(BASE)==EXPECTED==manifest['accepted_submission_sha256']
    sample_path=ROOT/'data/raw/sample_submission.csv';test_path=ROOT/'data/raw/test.csv'
    assert sha(sample_path)==manifest['inputs']['sample_submission.csv']
    assert sha(test_path)==manifest['inputs']['test.csv']
    protected=list((ROOT/'outputs').rglob('submission*.csv'))
    protected+=list((ROOT/'models').rglob('*'))+list((ROOT/'notebooks').glob('*.ipynb'))
    before={str(p):sha(p) for p in protected if p.is_file()}
    baseline=mapping(BASE);sample=mapping(sample_path)
    order=[r['id'] for r in rows(sample_path)]
    test_rows=[r for r in rows(test_path) if r['id']]
    test={r['id']:r for r in test_rows}
    assert len(test_rows)==len(test)==4457 and set(test)==set(sample)==set(baseline)
    forecast={key for key,r in test.items() if r['kind']=='forecast'}
    other=set(test)-forecast
    assert len(forecast)==2660 and len(other)==1797
    predictions={key:sample[key] if key in forecast else baseline[key] for key in order}
    assert all(predictions[key]==sample[key] for key in forecast)
    assert all(predictions[key]==baseline[key] for key in other)
    assert all(math.isfinite(float(v)) and float(v)>=0 for v in predictions.values())
    changed_text=[key for key in order if predictions[key]!=baseline[key]]
    changed_values=[key for key in order if Decimal(predictions[key])!=Decimal(baseline[key])]
    assert set(changed_text)<=forecast and set(changed_values)<=forecast
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['id','people']);writer.writeheader()
        writer.writerows(dict(id=key,people=predictions[key]) for key in order)
    written=rows(OUT)
    assert [r['id'] for r in written]==order and mapping(OUT)==predictions
    assert all(sha(Path(path))==digest for path,digest in before.items())
    info=dict(label='DIAGNOSTIC ONLY — not a candidate improvement',output=str(OUT.relative_to(ROOT)),
        output_sha256=sha(OUT),baseline_sha256=EXPECTED,ids=4457,unique_ids=4457,sample_order_exact=True,
        full_id_join=True,forecast_rows=2660,forecast_text_matches_sample=True,non_forecast_rows=1797,
        non_forecast_text_matches_baseline=True,finite_nonnegative=True,
        changed_values=len(changed_values),changed_prediction_strings=len(changed_text),
        forecast_values_already_equal=2660-len(changed_values),protected_files_unchanged=len(before),
        input_sha256={'test.csv':sha(test_path),'sample_submission.csv':sha(sample_path)},
        interpretation='Conditional on equal-weight mean of four task ratios and public forecast reference ratio=1: baseline forecast ratio = 1 - 4*(S-S0). Public subset only; use full score precision; display rounding creates uncertainty. This operation does not authenticate the official metric.',
        source_sha256=sha(Path(__file__)))
    VERIFY.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    NOTE.write_text('# DIAGNOSTIC ONLY\n\nNot a candidate improvement. Exact forecast rows use sample_submission.csv text; all other rows retain the protected 0.800 baseline text.\n\nConditional on the mean-of-four-ratios metric and forecast reference ratio 1:\n\n`baseline forecast ratio = 1 - 4 * (S - S0)`\n\nS and S0 must refer to this file and its exact baseline on the same public evaluation subset. Use full score precision. Displayed rounding introduces uncertainty; this does not measure private performance or authenticate an official aggregation formula.\n',encoding='utf-8')
    print(json.dumps(info,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
