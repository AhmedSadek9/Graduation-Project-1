# Step 38 — Verify Fact Ownership Before Reporting Outcomes

## الهدف
نقارن تصنيف Step 32 مع حقيقة مالك الـFact وحالة التحقق المسجلة في `benchmark/fact_manifest.json`، ونكتشف الاختلافات بدون تغيير Step 32.

## التركيب
انسخي `step38_verified_scoring.py` و`test_step38_verified_scoring.py` إلى `Zenab/experiments/`، جنب Step 32 وStep 37.

## التشغيل من داخل Zenab
```powershell
python experiments/test_step38_verified_scoring.py
python experiments/step38_verified_scoring.py
```

طالما `step32_paired_validation_predictions.jsonl` غير موجود (لأن الحالات الحقيقية لم تصل)، سيطبع `PREPARATION ONLY` ولن يصدر أي درجة. الاختبار 9/9 يستخدم fixtures مختلقة لا تعتبر نتائج نموذج حقيقي.

عند توفر تنبؤات Step 32 الموثقة، سيعيد فحصها ويكتب نسختين جديدتين في `experiments/results/` تحملان بادئة `step38_`. لن يعدل النتائج الأصلية. لا تشغيل للموديل ولا تنزيل.

## حدود مهمة
- يراجع الـFact ID والـPerson ID والـVerification Status والـAcceptable IDs.
- **لا يتحقق أن `spoken_text` نفسها مطابقة للـFact**. يجب إجراء تدقيق منفصل قبل الادعاء بصحة الكلام المسموع.
- لا يستبدل نجاح الفحوصات بوجود نتائج أحمد/تقى الفعلية أو الـIndependent Held-out.
- لا تغيري Frozen Threshold أو الملفات الأصلية.
