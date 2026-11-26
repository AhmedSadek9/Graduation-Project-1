# WANAS Step 37 — Fact Ownership Audit

دي طبقة فحص **تجريبية منفصلة** لثغرة Step 32: `fact_id` الموجود في الـResponse مش كفاية للحكم على صحة الإجابة، لازم يتطابق مع `person_id` الفعلي في `benchmark/fact_manifest.json`.

## التثبيت

انسخي ملفي Python داخل `Zenab/experiments/`، وشغّلي من فولدر `Zenab`:

```powershell
python experiments/step37_fact_ownership_audit.py
python experiments/test_step37_fact_ownership_audit.py
```

## مهم

- **لا يستبدل** `step32_run_paired_evaluation.py` أو يغيّر الـBridge أو الـThreshold.
- كل الأمثلة Fictional، وده فحص منطق فقط، مش Actual Face/STT Results.
- التحقق من Ownership وVerification لا يثبت إن نص `spoken_text` مطابق للـFact؛ ده يحتاج فحص محتوى مستقل.
- تطبيق الفحص داخل Runner هيحتاج تعديل واختبارات Regression، وبعدها تشغيل A/B/C/D لما يتوفر Actual Data.
