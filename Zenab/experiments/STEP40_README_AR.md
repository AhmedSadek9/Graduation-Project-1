# Step 40 — Verified Context Audit (Experimental)

**الهدف:** إعادة تقييم نتائج Step 39 بعد الرجوع إلى `benchmark/fact_manifest.json` للتحقق من ملكية الـFact وVerification، مع وضع أي Spoken Answer للمراجعة البشرية.

النسخة دي منفصلة ولا تعدل Step32 أو Step39 أو إعداد E5. وهي **لا تؤكد تطابق نص الإجابة مع الحقيقة** ولا صحة `fresh` من Session Manager.

## التثبيت

انسخي ملفَي Python إلى `Zenab/experiments/`، ولا تنقلي ملفات المشروع القديمة أو تستبدليها.

## التشغيل من `Zenab/`

```powershell
python experiments/test_step40_context_verified_audit.py
python experiments/step40_context_verified_audit.py
```

مع الـCase Manifest الفاضي حاليًا النتيجة ستكون `PREPARATION ONLY` ولا تُولد أي Metrics. بعد وصول الحالات الحقيقية وتشغيل Step39 بإذن الفريق، يقرأ السكريبت `experiments/results/step39_context_validation_predictions.jsonl`، ويفحص أن `case_id` و`run` و`question_id` و`image_id` و`expected_person_id` متطابقة مع الـManifest.

## المخرجات بعد وجود Predictions

- `experiments/results/step40_verified_context_validation_predictions.jsonl`
- `experiments/results/step40_verified_context_validation_report.json`

النتائج **Provisional** فقط، لأن مراجعة تطابق `spoken_text` مع الـFact وسلامة مصدر الـContext/Face/STT لازمة قبل التقرير النهائي. لا تشغلي Step39 بـ`--execute` إلا بعد تأكيد عقود الـFace IDs والـScore والـSession.
