# WANAS — Step 36: Outcome Classification & Paired Reliability Logic

## مكان الملف
انسخي **`test_step36_outcome_reliability.py` فقط** داخل `WANAS_Team_Repo/Zenab/experiments/` بجانب `step32_run_paired_evaluation.py`. لا تستبدلي ملف Step 32.

شغلي من فولدر `Zenab`:

```powershell
python experiments/test_step36_outcome_reliability.py
```

النتيجة المتوقعة: `STEP 36 OUTCOME / PAIRED RELIABILITY: 14/14 PASSED (synthetic logic-only)`.

## ماذا تختبر الخطوة؟
- Correct supported answer
- Wrong-person answer، بما في ذلك إجابة شخص معروف لسؤال متعلق بضيف Unknown
- Unsupported answer مع شخص صحيح
- Correct abstention عند Unknown أو حقيقة غير موجودة
- Unnecessary abstention لسؤال قابل للإجابة
- حساب **Run A − Run D** باستخدام **تقاطع case_id فقط**، وليس مقارنة عينتين مختلفتين
- إبقاء Run B وRun C منفصلين
- عدم إصدار Reliability Loss عند عدم وجود حالات A/D مشتركة

## ملاحظات مهمة ظهرت بالمراجعة
1. الدالة `categorize()` في Step 32 تتحقق من `response.person_id` وعضوية `fact_id` في `acceptable_fact_ids`. لكنها لا تتحقق بصورة مستقلة من مالك الـFact داخل Fact Manifest؛ هذا **تحقق إضافي لازم** قبل اعتماد نتائج فعلية.
2. إذا عاد `can_speak_personal_memory=false` مع `fact_id` غير فارغ، الدالة الحالية قد تعتبر الحالة Abstention؛ يلزم Audit للفصل بين Fact داخلي وعدم إفشاء فعلي في واجهة المستخدم.
3. Step 35 Context Adapter ليس متصلاً حاليًا بالـStep 32 Runner. السياق موجود في السجل لكنه غير مستهلك. لذلك النتائج الحالية لا تثبت حل ضمائر المحادثة End-to-End.
4. لا يتم تحميل E5 ولا تشغيل أي صور أو STT، ولا تغيير Threshold، ولا إنتاج نتائج حقيقية A/B/C/D.
5. مثال الـ50 percentage points في الاختبار *حساب صناعي من حالتين* وليس قياس أداء WANAS.

## بعد نجاح Step 36
احتفظي بالاختبارات، ثم ننتقل لتقوية Outcome Audit وربط Session/Context بالـRunner بشكل صحيح قبل استلام المخرجات الفعلية من أحمد وتقى. لا ترفعي Face Photos أو Audio خام إلى GitHub.
