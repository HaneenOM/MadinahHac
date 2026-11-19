"""Generate the Arabic session report from saved evidence; no training or prediction."""
from pathlib import Path
import json,time,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'noran/experiments'))
import prompt56 as S
P=S.OUT
def read(name):return json.loads((P/name).read_text())
def number(x):return f'{x:.6f}'
def main():
    scope=[read(f'hidden_{x}_gd_result.json') for x in ('fixed','mirror','testday')]
    strict_path=P/'hidden_testday_strict_gd_result.json'
    strict=read(strict_path.name) if strict_path.exists() else None
    missing=[read(f'missing_{x}_result.json') for x in ('fixed','mirror')]
    f1=read('forecast_im_result.json');f2=read('forecast_im_select_result.json')
    candidate=read('candidate/verification.json');verification=read('verification.json')
    table='\n'.join(f"| {r['protocol']} | {number(r['v7_mae'])} ← {number(r['gd_mae'])} | {number(r['v7_ratio'])} ← {number(r['gd_ratio'])} | {r['improved']} / {r['worsened']} | {number(r['sensitivity'][2]['mae_gain'])} |" for r in scope+([strict] if strict else []))
    ledger=read('ledger.json')
    for row in ledger:
        if row['id']=='H1':
            row.update(status='complete',decision='Separate exploratory G/D candidate; small fixed/mirror gains, weak testday support',
                       results=scope,strict_testday=strict,candidate=candidate['path'],
                       runtime_seconds_by_protocol={s:read(f'hidden_{s}_runtime.json')['seconds'] for s in ('fixed','mirror','testday','testday_strict')},
                       candidate_generation_seconds=candidate['runtime_seconds'])
        if row['id']=='M1':row.update(results=missing,runtime_seconds=sum(read(f'missing_{s}_runtime.json')['seconds'] for s in ('fixed','mirror')))
        if row['id']=='F1':row.update(results=f1,runtime_seconds=f1['runtime_seconds'])
        if row['id']=='F2':row['decision']='reject/no gain: no units selected by training-only robustness checks'
    (P/'ledger.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    runtimes=[]
    for mission,sets in [('hidden',('fixed','mirror','testday','testday_strict')),('missing',('fixed','mirror'))]:
        for which in sets:
            file=P/f'{mission}_{which}_runtime.json'
            if not file.exists():continue
            r=read(file.name)
            runtimes.append(f"| {mission} / {which} | {sum(f['fit_seconds'] for f in r['folds']):.1f} | {sum(f['inference_checks_seconds'] for f in r['folds']):.1f} | {r['seconds']:.1f} |")
    timing='\n'.join(runtimes)
    elapsed=(time.time()-S.START)/60
    strict_note=(f"اكتمل تدقيق testday الأشد: النسبة {number(strict['v7_ratio'])} ← {number(strict['gd_ratio'])}، وMAE {number(strict['v7_mae'])} ← {number(strict['gd_mae'])}. يُعرض منفصلًا عن البروتوكول الأصلي؛ لم تُضبط الوصفة على نتيجته." if strict else
                 'تدقيق testday الأشد لم يكتمل عند توليد هذه النسخة من التقرير؛ لا تُعامل نتيجته كدليل مكتمل.')
    text=f'''# PROMPT 56 — جلسة تحسين محدودة بساعة

**النتيجة: مرشح استكشافي صغير لـHIDDEN فقط، دون ترقية تلقائية.** بقيت Version 7 ومرجع 0.724 العام كما هما. لم يحدث رفع أو commit أو push. استغرقت الجلسة حتى كتابة التقرير نحو **{elapsed:.1f} دقيقة** من حد 60 دقيقة، شاملًا القراءة والتنفيذ والتصحيحات والتحقق والتقرير. البداية المحفوظة محافظة: 14:17 بتوقيت الرياض، وحد الإيقاف 15:17 يوم 19 نوفمبر 2026.

## التجارب والقرارات

| التجربة | النطاق | النتيجة المقاسة مقابل V7 | القرار |
|---|---|---|---|
| تنعيم HIDDEN بثلاث دقائق، ثم نطاق G/D المختار على fixed | G وD فقط في المرشح؛ التجربة الكاملة تشمل B أيضًا | fixed: {number(scope[0]['v7_ratio'])} ← {number(scope[0]['gd_ratio'])}؛ mirror: {number(scope[1]['v7_ratio'])} ← {number(scope[1]['gd_ratio'])} | حفظ مرشح استكشافي؛ المكسب صغير ودعم testday ضعيف |
| أوزان MISSING المحدبة | A–H؛ نفس مرشحات V7 ومزجها | fixed: {number(missing[0]['v7_ratio'])} ← {number(missing[0]['trial_ratio'])}؛ mirror: {number(missing[1]['v7_ratio'])} ← {number(missing[1]['trial_ratio'])} | رفض؛ مكسب fixed غير متين وmirror أسوأ |
| تصحيح FORECAST سببي مجمّع | I–M؛ Ridge واحد على المستوى والميل والأفق | {number(f1['v7_ratio'])} ← {number(f1['trial_ratio'])}؛ MAE {number(f1['v7_mae'])} ← {number(f1['trial_mae'])} | رفض التطبيق العام |
| مراجعة واحدة لـFORECAST: اختيار نطاق التطبيق من التدريب فقط | تحقق داخلي بحجب مقاطع مرصودة، دون إجابات تاريخ التقييم | {number(f2['v7_ratio'])} ← {number(f2['trial_ratio'])}؛ لم تُقبل أي كاميرا في الطيتين | لا مكسب؛ لم أخفف شروط الاختيار بعد كشف النتائج |
| مراجعة HIDDEN-FUTURE | أدلة PROMPT 54 المحفوظة | تغيير توقيت D مرفوض سابقًا؛ fixed وmirror والتسجيل القصير لم تدعمه | لا تكرار لتجربة مكافئة، ولا تغيير جديد |

لا توجد أعمال PROMPT 55 في هذه النسخة. راجعت توثيق V7 وإعداداتها وبيانها، والتنفيذ الحالي للمهام الأربع والتقارير الأخيرة. أعطيت الأولوية لإمكان تقليل ضوضاء التوقع النهائي في HIDDEN، ثم استقرار معاملات MISSING، ثم تجربة FORECAST خفيفة لكاميرات ليس لها T05. لم تُحمّل مكتبات أو أوزان جديدة، ولم يُشغّل تدريب أو استدلال T05.

## لماذا هذه التغييرات؟

**HIDDEN:** بعض أعضاء التجميع، ومنها الأشجار وTabICL، تعتمد على قراءة المانح في الدقيقة نفسها، رغم تنعيم ميزات أعضاء أخرى. الاختبار الوحيد هو المتوسط المثلثي للوغاريتم: أوزان 1،2،1 عند الدقيقة السابقة والحالية والتالية. لا يتجاوز تسجيلًا، ولا يضغط الدقائق المفقودة، ولا يعبر فراغًا زمنيًا. الدقيقة المجاورة تُستخدم عندما يوجد مانحان مرصودان على الأقل؛ خلاف ذلك يبقى إسهام الدقيقة الحالية. HIDDEN تسمح بهذه المشاهدات المحررة على الجانبين؛ لم يُطبق هذا على توقعات المستقبل.

تحسنت الكاميرات الثلاث على fixed، لكن مكسب B البالغ 0.000422 في MAE يتحول إلى خسارة عند حذف أفضل نافذة. يبقى مكسب G وD موجبًا بعد حذف أفضل نافذتين لكل منهما، ولذلك ثُبت **G/D قبل الاطلاع على نتائج mirror/testday** في `hidden_development_scope.json`. لم يُنتقَ نطاق جديد بعد ذلك. هذا اختيار على fixed، وليس مصادقة مستقلة.

**MISSING:** في الطية التطويرية الأولى ظهر 11 معاملًا سالبًا و5 نماذج فرعية مجموع أوزانها أكبر من واحد، بين 42 نموذجًا فرعيًا؛ مدخلات الشركاء المزاحة متشابهة وقد تجعل الاستقراء غير مستقر. اختبرت معاملات غير سالبة مجموعها لا يتجاوز واحدًا، مع نفس أعمدة V7 والخسائر وبقية الإعدادات. لم تثبت الفرضية: التحسن الكلي في MAE على fixed نحو 0.000016 فقط، ويصبح سالبًا بعد حذف أفضل فجوة. F تحسنت على تاريخَي fixed بمقدار 0.088 و0.149، لكنها تراجعت على تاريخَي mirror بمقدار 0.067 و0.008؛ لم تُقبل استثناءً.

**FORECAST I–M:** V7 تحتفظ بمستوى مستمد من آخر قراءة ووسيط آخر ثلاث، ولا يستعمل هذا الفرع ميل الحركة عند الأفق. اختبرت Ridge بمعامل 10 ثابت لتوقع الباقي المطبّع من المستوى والميل السببيين خلال 5/10 دقائق وتفاعلهما مع الأفق. لا تستخدم الميزات أي قراءة بعد now. ترك كل تقييم تسجيل الاختبار الآخر وحده للتدريب. تحسنت I على التاريخين، لكن مكسب التاريخ الثاني ينقلب عند حذف أفضل نافذة؛ لم أخترها يدويًا من إجابات التقييم. المراجعة الوحيدة جعلت اختيار الكاميرا يجري داخل تسجيلات التدريب، بحجب مقطع كامل واشتراط بقاء المكسب موجبًا بعد حذف أفضل مقطع؛ لم تجد دعمًا لتطبيق النموذج.

## المقارنة التفصيلية لنطاق HIDDEN المختار

النسبة هي متوسط MAE/MAE المرجع على التواريخ، وليست قسمة المتوسطين المجمّعين. المكسب = خطأ V7 ناقص خطأ التجربة؛ الموجب أفضل. حساسية الحذف أدناه تحذف الحدث الزمني كاملًا بما يشمله من كاميرات، لا دقيقة منفردة.

| البروتوكول | MAE V7 ← G/D | النسبة V7 ← G/D | نوافذ تحسنت / تراجعت | مكسب MAE بعد حذف أفضل حدثين |
|---|---:|---:|---:|---:|
{table}

في fixed، G: **2.728900 ← 2.721957** وD: **4.721037 ← 4.698475**. في mirror، G شبه ثابتة **2.108738 ← 2.108575** وD تتحسن **4.972844 ← 4.954211**. testday الأصلية لا تؤكد مكسبًا: MAE الإجمالي يزيد بنحو 0.000026، رغم تحسن النسبة الموزونة قليلًا؛ وتتراجع النتيجة عند حذف أفضل الفترات. لا أعد هذا فرقًا عمليًا أو مكسبًا يتجاوز ضوضاء إعادة الإنتاج.

{strict_note}

تحتوي fixed على 843 هدف HIDDEN و36 نافذة كاميرا في 24 فترة، وmirror على 830/36/24، وtestday على 738/36/24. الكاميرتان B/G تشتركان في فترات؛ ليست الأهداف تكرارات مستقلة. ملفات `hidden_*_by_recording.csv` و`hidden_*_by_window.csv` و`hidden_*_gd_result.json` تفصل التسجيلات والكاميرات والنوافذ.

MISSING: fixed لديها 358 هدفًا في 38 فجوة، منها 32 كاملة الإجابات؛ mirror لديها 368 هدفًا في 39 فجوة، منها 33 كاملة. التغطية الكاملة لا تُفترض للست فجوات الجزئية في كل مجموعة. توجد نتائج بحسب الكاميرا وطول الفجوة في `missing_*_by_gap_span.csv` وتغطية كل فجوة في `missing_*_gap_coverage.csv`.

FORECAST I–M: التقييم المنفصل يحتوي 7,021 زوج هدف/أفق، لكنها تعيد استخدام **793 قراءة هدف** ضمن 70 مقطع كاميرا. ليس هذا fixed أو mirror أو testday الخاصة بالمهام الأخرى، ولا تُجمع درجاته معها. نافذة البيانات المحررة لا تماثل تمامًا الفترات المستقبلية الحقيقية؛ الكاميرات I–M غير موجودة في التدريب التاريخي. توجد تفاصيل الأفق 1–5 و6–10 و11–15 في `forecast_im*_by_horizon.csv`.

## حماية التقييم والحدود

كل طية fixed/mirror تستبعد تاريخ التقييم بالكامل من التدريب. لا تُقرأ إجابات الحالات في الموديولات، وتُحفظ التوقعات قبل وصل الإجابات لحساب الأخطاء. في testday الأصلية تُستبعد الكاميرات المحجوبة معًا وفق القناع من سياق الملاءمة الخاص بطية تسجيلها، مع السماح بالصفوف المحررة وفق البروتوكول. أضيف تدقيق أشد يحجب إجابات الكاميرات المستهدفة في **التاريخين معًا من كل ملاءمة**، ويُعرض منفصلًا.

فُحص إعادة بناء المدخلات بعد استبدال الإجابات المحجوبة بـ1e8: بقيت جداول الاستدلال مطابقة، مع الاحتفاظ بحدود التسجيل والمحور الزمني. اختبارات HIDDEN تغيّر B/G معًا أو D، وتتحقق من الموديول الكامل على عينات عبر الفترات. FORECAST اجتازت 282 مقارنة ثبات عند تغيير المستقبل لكل وصفة، دون إخفاق. التنعيم لا يُستخدم في FORECAST أو HIDDEN-FUTURE. لا توجد ميزات مخفية معاد بناؤها يُدرّب عليها نموذج لاحق.

جميع هذه التواريخ سبق فحصها. fixed اختيرت للتطوير؛ mirror/testday تدقيق للوصفة المثبتة على تواريخ معروفة سابقًا، لا بيانات مستقلة جديدة. مراجعة FORECAST الثانية جاءت بعد تحليل نتائج الأولى، ولذلك تبقى استكشافية، رغم أن الاختيار البرمجي نفسه لا يدخل إجابات طية التقييم. لا توجد توقعات لدرجة Kaggle.

## المرشح والتحقق

الملف: `{candidate['path']}`.

بصمته SHA-256:
`{candidate['sha256']}`

انطلق من المرجع المقبول بالضبط، وحُسب التغيير **المرجع المقبول + (V7 المحلية بعد التنعيم − V7 المحلية)**. بذلك لا يُنقل انحراف إعادة الإنتاج الكامل إلى المرشح. التقييمات المحلية تقارن التنعيم بالأساس الناتج من الملاءمة نفسها، وليس بملف ذي بيئة مختلفة.

تغيرت **{candidate['changed_rows']} قيمة HIDDEN لـG/D فقط**؛ وبقيت **{candidate['exact_unchanged_prediction_strings']} قيمة** أخرى مطابقة نصيًا، بما فيها كل FORECAST وHIDDEN-FUTURE وMISSING وHIDDEN لـB. تحققت 4,457 هوية فريدة وترتيب sample ومجموعة المعرّفات والقيم المنتهية غير السالبة، وأعيدت قراءة CSV للتحقق المستقل. متوسط مقدار التغيير {candidate['mean_abs_delta']:.6f} وأقصاه {candidate['max_abs_delta']:.6f}؛ لا تعني هذه المقادير مكسب دقة متوقعًا.

أعيد استخدام cache توقعات V7. كانت خمس دقائق مجاورة لـG غير موجودة في ملف الأهداف، فحُسبت بوصفة G الأصلية دون TabICL غير المستخدم لـG أصلًا، مع تحقق مطابقتها لنقاط cache موجودة واختبارات حجب. D كانت مغطاة بالكامل من cache. نجحت 565 مقارنة لتحويل التنعيم بعد حجب الكاميرات، وثمانية فحوص إضافية للموديول الكامل عند استكمال الجيران. لا تُستعمل أي حقيقة مخفية لهذه الدقائق.

حُفظ `applied_deltas.csv` و`extra_neighbor_predictions.json` و`candidate/verification.json`، والتحقق المستقل في `outputs/prompt56/verification.json`. لم تتغير بصمة المرجع المقبول `2eced7c2ea9765d93d798fc0130126a523e51e69b1fb4ac5bf6d5783d81cc88c`، ولا بصمة النسخة المحلية `f5ab51ecd76287544afbadd12b2d230752e01b929575e6cb37b6436506d209da`. بقي {verification['protected_files_unchanged']} ملفًا محميًا دون تغيير.

## الزمن وإعادة التوليد

الأزمنة التالية بالثواني، وتشمل إعادة حساب V7 اللازمة للمقارنة. الاستدلال مجمع مع فحوصه؛ بعض الوظائف شُغلت بالتوازي، فلا تُجمع لتمثل زمن الجلسة.

| المهمة / البروتوكول | الملاءمة | الاستدلال والفحوص | التشغيل الكامل |
|---|---:|---:|---:|
{timing}

FORECAST الأولى {f1['runtime_seconds']:.1f} ثانية، والمراجعة {f2['runtime_seconds']:.1f} ثانية. بناء المرشح {candidate['runtime_seconds']:.1f} ثانية، منها {candidate['g_only_v7_fit_seconds']:.1f} ثانية لإعادة ملاءمة G الأصلية لاستكمال الجيران. واجهت أداة HIDDEN خطأ فهرسة لدقيقة قليلة المانحين وأُصلح قبل حفظ النتيجة المكتملة؛ وعولجت نافذة ميزات فارغة في FORECAST قبل أول نتيجة مقاسة. زمن هاتين المحاولتين المحبطتين محسوب ضمن زمن الجلسة، ولا تُعدان تجارب نمذجة إضافية.

من جذر المشروع، لتوليد نسخة جديدة مع حفظ الملف الحالي:

```powershell
.\\.venv_lab\\Scripts\\python.exe -X utf8 noran\\experiments\\prompt56_build.py --output-dir outputs/prompt56/rebuild_01
```

يرفض المولّد الكتابة فوق ملف مرشح موجود؛ اختر اسم مجلد جديدًا للتكرار. الوصفة في `noran/configs/prompt56_candidate.json` والموديول `noran/tracks/p56_hidden_gd/model.py`، وسجل جميع القرارات في `outputs/prompt56/ledger.json`.

**الخطوة الموصى بها:** مراجعة هذا المرشح الصغير وتثبيت وصفته لتقييم جديد ممثل لـG/D؛ لا أغيّر الدفتر الافتراضي ولا أعدّه بديلًا مؤكدًا لنتيجة 0.724.
'''
    (ROOT/'reports/prompt56_one_hour_ar.md').write_text(text,encoding='utf-8')
    print('Report saved; elapsed minutes:',elapsed,'strict complete:',strict is not None)
if __name__=='__main__':main()
