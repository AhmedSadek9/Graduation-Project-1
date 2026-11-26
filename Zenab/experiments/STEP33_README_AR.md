# WANAS Step 33 — Validation Case Pairing Plan

**الهدف:** قائمة مقترحة لربط أسئلة الـValidation بصور حقيقية لاحقًا، بدون اختلاق بيانات.

## التثبيت
انسخي `step33_prepare_validation_pairing.py` إلى `Zenab/experiments/`، ثم من داخل `Zenab/` شغّلي:

```powershell
python experiments/step33_prepare_validation_pairing.py
```

## المخرجات
- `handoff/step29/STEP33_VALIDATION_PAIRING_PLAN.csv`
- `handoff/step29/STEP33_VALIDATION_PAIRING_SUMMARY.json`

ملف الـPlan **ليس** `CASE_MANIFEST_WORKING.csv` التنفيذي. لا يضيف أي حالة، ولا يشغّل الموديل، ولا يعدّل الـFrozen Threshold.

## الخطوة البشرية قبل الربط
1. هناء وأحمد يؤكدان الـStable-ID Mapping والـGround Truth لكل صورة.
2. اختاري صورًا حقيقية للشخص المطابق للسؤال فقط، أو صور Unknown للحالات Unknown. متربطيش صور لأشخاص آخرين بالـPerson ID الموجود في السؤال.
3. سجلي `image_id` الفعلي و`pairing_rationale` و`conversation_context` ومراجعة الـCase، مع حالة `no_face/no_valid_face` **من مخرجات أحمد** لا بالتخمين.
4. تقى تسلّم paired actual STT مع نفس question_id / utterance_id / session_id.
5. بعد الاعتماد، انقلي الـCases المؤهلة بعناية إلى `CASE_MANIFEST_WORKING.csv` بالـSchema الكامل من Step29 وشغلي Step30/31 ثم Step32 preflight. لا تشغلي `--execute` قبل اعتماد الـFace Score Contract والـIDs.

## قيد مهم
الـ26 سؤالًا في Validation الحالية تخص فقط 4 Known Person IDs، ومعها حالتان Unknown. لذلك الـ50 صورة الحالية لا يمكن افتراض أن لها سؤالًا مناسبًا من الـ26. قد يلزم إعداد أسئلة/سيناريوهات Validation إضافية **قبل التقييم** مع حصرها وتجميدها، بدلاً من وضع Label غير صحيح. مجموعة Independent Test تظل مستقلة.
