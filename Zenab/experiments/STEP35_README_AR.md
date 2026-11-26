# WANAS Step 35 — Context Safety Adapter (Experimental)

الهدف: اختبار حل محدود للضمائر مثل `Tell me about him` بدون استخدام هوية شخص من الـConversation History وحدها.

## أين تضعين الملفات؟
ضعي `step35_context_adapter.py` و`test_step35_context_adapter.py` في مجلد `Zenab/experiments`.

من داخل `WANAS_Team_Repo/Zenab`:

```powershell
python experiments/test_step35_context_adapter.py
```

**المتوقع:** `STEP 35 UNIT TESTS: 16/16 PASSED`.

## شروط السماح بالـPronoun Resolution
- نتيجة الوجه الحالية `recognized` ومعها `person_id` صالح و`recognition_confidence` صالح وأعلى من الحد المحدد.
- `session_id` حاضر ويساوي Session السياق.
- السياق يعلن `fresh=True` بشكل صريح (الـCaller مسؤول عن تحقق صلاحية المدة؛ لا يكفي ضبط القيمة يدويًا).
- `referent_person_id` الموجود بالسياق يطابق الـCurrent Face ID؛ لا استعادة تلقائية لهوية قديمة.
- الـName Lookup يأتي من الـProfile المعتمد لا من نص النموذج.
- الرفض من الـUnderlying Bridge يُحترم كما هو.

## حدود الاختبار
هذا Adapter **تجريبي وليس جزءًا من Memory Bridge الإنتاجي**. يلتقط صيغتين بسيطتين من سؤال الضمير فقط، ويعيد صياغته إلى `Who is {verified-name}?` إذا استوفت الحالة الشروط. ليس فهمًا عامًا للضمائر أو السياق. اختبارات Step 35 تستعمل Stub Bridge متعمدًا؛ لا تختبر Face/STT Outputs حقيقية أو أداء E5 أو سلامة بيانات حقيقية. لا يتم تعديل 120 Fact أو الـFrozen Threshold.

بعد نجاح الاختبار نراجع مع أحمد مصدر/session validity وزمن صلاحية السياق ومع تقى نوع utterance والمتحدث. ثم نعمل اختبارات تكامل جديدة بالبيانات الفعلية.
