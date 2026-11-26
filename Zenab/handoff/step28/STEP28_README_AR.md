# Step 28 — تعليمات بسيطة

دي **مرحلة تجهيز للربط**، مش ربط فعلي مع أحمد أو تقى، ومش Independent Evaluation.

## النقل
1. من الـZIP، انسخي `step28_readiness_check.py` إلى `experiments/`.
2. اعملي فولدر `handoff/step28/` وانسخي فيه ملفات `STEP28_INTEGRATION_PROTOCOL.md` و`FACE_OUTPUT_TEMPLATE.csv` و`STT_TRANSCRIPT_TEMPLATE.csv` و`STEP28_README_AR.md`.
3. **متكتبيش بيانات أشخاص حقيقيين في ملفات الأمثلة** قبل الموافقات المناسبة.

## التشغيل
من جذر `WANAS_Zinab_Final`:

```powershell
python experiments/step28_readiness_check.py
```

هيطبع PASS/MISSING للملفات المطلوبة وهيحفظ `handoff/step28/STEP28_LOCAL_READINESS_RESULT.json`.

النجاح هنا يعني **الملفات والإعدادات موجودة ومنظمة**، مش إن الـFace/STT فعليًا اتوصلوا أو إن 0.80 بقى آمن. مش هيحمّل موديلات، ولا يغير بيانات أو Threshold.

## تقسيم اللي بعد كده
- **هناء**: مراجعة 40 سؤال وMapping وتوثيق بيانات مراجعة حقيقية، وكمان تنسيق مجموعة اختبار جديدة غير مكشوفة لو محتاجين independent score.
- **أحمد**: Face results حقيقية وتعريف الـScore والـPerson IDs وعقد الربط.
- **تقى**: STT outputs paired مع reference transcripts وسلامة الـspoken_text gate.
- **زينب**: تقارن retrieval تحت نفس شروط الـSafety، وتجهز Final Integration Report بعد وصول بيانات الفريق.

**مفيش داعي لإعادة تشغيل أو تحميل E5 في Step 28.**
