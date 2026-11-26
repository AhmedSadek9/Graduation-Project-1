# Step 21 — E5 Demo Adapter (Experimental Only)

## مهم
- لم يتم تغيير `memory_integration.py` أو `Task1` أو `Task2` أو `Task3` أو الاختبارات الأصلية.
- يستخدم E5 في تجربة استرجاع تفاعلية منفصلة بالبيانات الصناعية فقط، وليس API منشورة ولا ربطًا حقيقيًا بكاميرا أو STT.
- الوضع الافتراضي `production` يرفض كل طلبات هذا الـAdapter عمدًا.
- يطبّق التحقق من هوية الشخص والـscore، ثم يفلتر الشخص والـfacts حسب حالة التوثيق، ويطبّق Fact-Type Guard، ثم يُرتّب بالنموذج ويستخدم threshold المجمد `0.8824205`.
- الأسئلة غير المعروفة للـGuard تُرفض عمدًا؛ تغطية اللغة محدودة. النموذج لا يثبت صلاحية الاستخدام الطبي أو الإنتاجي.
- موديلات Hugging Face موجودة في Cache الجهاز لا داخل ZIP المشروع. لا تحذفي `C:\Users\Zenab\.cache\huggingface\hub`.

## تشغيل من مجلد المشروع

```powershell
python experiments/test_e5_demo_bridge_step21.py
python experiments/e5_demo_bridge_step21.py
```

الأول 7 اختبارات منطقية فقط باستخدام Fake Embeddings ولا يشغّل E5؛ الثاني يشغّل الموديل الحقيقي من Cache الجهاز. قبل التشغيل الثاني ممكن استخدام:

```powershell
$env:HF_HUB_OFFLINE="1"
python experiments/e5_demo_bridge_step21.py
Remove-Item Env:HF_HUB_OFFLINE
```

إذا كانت النسخة المحفوظة غير مكتملة فسيفشل وضع Offline بدون تحميل؛ ارجعي للوضع العادي بدل حذف Cache.

## تسليم الفريق
الـAPI Contract الحالي في `handoff/step19/` منفصل عن هذا الـDemo Adapter. الربط النهائي يحتاج تصميم API مع أحمد، ID Mapping ومراجعة أسئلة مع هناء، وTranscripts حقيقية مع تقى.
