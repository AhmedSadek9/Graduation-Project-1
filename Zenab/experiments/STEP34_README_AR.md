# WANAS — Step 34: Stale Identity Safety Regression

## الهدف
اختبار أن حالة التعرف السابقة لا تؤدي إلى كشف حقائق شخصية عندما تتحول نتيجة الوجه التالية إلى `unknown` أو `no_face` أو `no_valid_face` أو `low_confidence`، حتى لو ورد سؤال غامض مثل “Tell me about him”.

## التثبيت والتشغيل
1. انقلي `test_step34_stale_identity_safety.py` إلى مجلد `Zenab/experiments/` داخل مستودع الفريق.
2. من داخل `Zenab` شغّلي:

```powershell
python experiments/test_step34_stale_identity_safety.py
```

المتوقع: `16/16 PASSED`. لا يحتاج تثبيت أي موديل أو تشغيل Hugging Face، لأن اختبار E5 يستخدم نموذجًا وهميًا ثابتًا (FixedModel) للتحقق من بوابات الأمان فقط، وليس الموديل الحقيقي.

## تفسير النتائج
- اختبارات **محلية فقط** تستخدم 120 حقيقة افتراضية والـBridges الموجودة.
- لا تثبت أن **Conversation Context** يُستهلك أو يُفهم؛ كود Step 32 يُسجّل السياق لكنه لا يدخله في استعلام E5.
- لا تستخدم مخرجات أحمد أو تقى؛ لا تضاف نتائجها إلى A/B/C/D الحقيقية.
- لا تستبدلي نتيجة `expected_person_id` بمخرجات `decision` الحقيقية عند بدء Run B/D.

## ماذا نعمل بعد ذلك؟
نُنشئ سياسة واضحة لتصفير/إبطال الهوية السابقة عند أي فشل Face Recognition، ثم نختبر التعامل مع الضمائر والسياق على تسلسل محادثة حقيقي في Demo مستقل قبل دمجها. ويظل الـFrozen E5 Threshold دون تعديل.
