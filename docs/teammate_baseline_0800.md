# خط الأساس0.800 — تشغيل وتعديل

0.800 درجة Kaggle عامة **مبلّغ عنها من المستخدم** للتقديم الموحّد B/G/D؛ لا درجة خاصة متوقعة.

## التشغيل

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
.\.venv\Scripts\python.exe scripts/run_current_baseline_notebook.py
```

ضع train/test/recordings/sample_submission الرسمية في `data/raw/`، أو افتح
`notebooks/current_baseline.ipynb` بنواة `.venv` ثم Restart Kernel وRun All.
الناتج `outputs/current_baseline_0800/submission_current_baseline.csv` مع الإعدادات
والتحقق والدفتر المنفذ. لا يُستبدل إخراج مختلف. أصول T05 الاثنا عشر وmanifest وجدول
الصلاة في `models/current_baseline/` مطلوبة ومتتبعة؛ لا caches أو تقديمات سابقة أو
نماذج HIDDEN محفوظة مطلوبة. نماذج HIDDEN يعاد تدريبها من المصدر. PyTorch للـoptimizer
المعطل فقط، وليس مطلوبًا في `requirements-baseline.txt`.

## الوصفات وأماكن التعديل

| المهمة | التنفيذ المشترك |
|---|---|
| FORECAST | `baseline_tasks.condition_forecast/forecast_value/im_forecast_value` و`t05_forecast.gp` |
| HIDDEN | `baseline_tasks.fit_hidden/predict_hidden/hidden_value` و`hidden_bgd.fit/predict` و`hidden_catboost.features/train_model` |
| HIDDEN-FUTURE | `baseline_tasks.fit_hidden_future/hidden_future_value` و`v2_experiments/prayer_experiments` |
| MISSING | `baseline_tasks.fit_missing/missing_value` و`v2_missing` |

B: CatBoost269،عمق4،L2=10،MAE،تعلم .05،seed42،وزن جمعة3،نسخة dropout20%؛
مزج .5 مع تقدير B السابق. G: dropout Ridge alpha3000 وثلاث نسخ20% ووزن جمعة3،
مزج .5 مع مرجع العينة. كلاهما يتدرب على train وtest المحرر؛ المانحون A,C,D,E,F,H.
D: HGB السابق من train فقط، عبر `hidden_reconstruction_trial.fit_new/make_features`.
نقص مانحين آنيين يعيد B/D للطريقة السابقة؛ G يستخدم استيفاء المانحين المحدود.

`catboost_iterations` وalpha يؤثران عند الملاءمة: أعد التدريب. `catboost_weight/g_weight`
يؤثران عند التنبؤ. بقية إعدادات الشجرة في `hidden_catboost.PARAMS/SPECS`؛ تغييرها يحتاج
إعادة الملاءمة. لا نموذج محمّل يتجاهل التعديل. `baseline_workflow` للتنسيق والتجميع؛
تعديل `v2_hidden` أو `v2_missing.interp_at` قد يؤثر في أكثر من مهمة.

## التقييم بعد التعديل

```powershell
.\.venv\Scripts\python.exe src/evaluate_baseline_task.py --task hidden --config outputs/current_baseline_0800/task_config.json --output-dir outputs/eval_hidden_strict
.\.venv\Scripts\python.exe src/evaluate_baseline_task.py --task hidden --hidden-protocol release_matched --config outputs/current_baseline_0800/task_config.json --output-dir outputs/eval_hidden_release
```

الصارم يستبعد تاريخ التقييم كله، ويضيف test المحرر خارج ذلك التاريخ لـB/G فقط.
المطابق يستبدل تسجيلَي11–12 بالـfeeds بعد **كل أقنعة المهام**؛ لا إجابة مقنّعة في التدريب
أو الميزات أو المعالجة. D يلتزم بجدول تدريب الطية دون التوسعة. لا تحميل لنماذج الإنتاج.
النتائج `by_date_camera.csv/by_window.csv`، ومصادر الملاءمة في `verification.json`.
للمهام الأخرى استخدم `--task missing` أو`hidden-future`؛ تقييم FORECAST موثق في
`docs/local_evaluation.md` مع أصول طيات مستقلة وإقرار provenance القديم، دون refit.
`reliable_evaluation.py` يستدعي HIDDEN الجديد أيضًا؛ أمر local_score القديم أرشيفي.
المقياس محلي خاص بالمشروع؛ التواريخ فُحصت مرارًا وليست تحققًا مستقلًا جديدًا.

## مقارنة0.806 ونسخة نظيفة

```powershell
.\.venv\Scripts\python.exe src/baseline_workflow.py --config configs/baseline_0806.json --output-dir outputs/previous_0806_reproduced
.\.venv\Scripts\python.exe scripts/check_clean_baseline.py --output-dir outputs/clean_baseline_check
```

الإعداد السابق و`manifest_0806.json` ودليله `teammate_baseline_0806.md` محفوظة؛
لا تجمع d_only التاريخي مع BGD. نسخة الاختبار النظيفة تنسخ المصدر والأصول والبيانات
فقط، وتعزل الحزم المثبتة المعلنة وتبعياتها. لا تنسخ outputs أو تقديمات أو نماذج HIDDEN.
تشغّل الدفتر بنواة جديدة وتقييم HIDDEN؛ ليست اختبار تنزيل pip جديد من الشبكة.

للمساهمة: `git switch -c task/hidden-update`، عدّل مهمة واحدة، واختر OUTPUT واسم مرشح
مستقلين و`VERIFY_ACCEPTED=False`، ثم Restart Kernel وRun All والتقييم **بنفس JSON**.
راجع التغطية والنوافذ و`git diff` قبل commit يدوي. لا تغير البصمة لإخفاء اختلاف.
ضمّن المصادر والإعدادات والدفتر والأصول المتتبعة؛ البيانات والتقديمات تبقى خارج Git.
