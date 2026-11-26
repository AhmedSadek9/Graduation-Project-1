# Step 39 — Context-Aware Paired Evaluation (Experimental)

## الملفات
- `step39_context_aware_runner.py`: بديل تجريبي منفصل عن Step32؛ يعيد استخدام تجهيز A/B/C/D، لكنه يمرّر طلبات الضمير إلى `SafeContextAdapter` من Step35.
- `test_step39_context_runner.py`: اختبارات على Fake Bridge فقط.

ضع الملفين في `WANAS_Team_Repo/Zenab/experiments/`، وتأكد أن ملفات Step32 وStep35 موجودة بجانبهما.

## تشغيل آمن الآن
من فولدر `Zenab`:

```powershell
python experiments/test_step39_context_runner.py
python experiments/step39_context_aware_runner.py
```

الاختبار الأول لا يحمل E5 ولا يستعمل Face/STT حقيقي؛ الأمر الثاني **Preflight فقط**.

## صيغة `conversation_context` عند إنشاء Pairings حقيقية
هذا العمود لازم يحتوي JSON (وليس جملة حرة)، مثال تعليمي فقط:

```json
{"referent_person_id":"person-001","session_id":"session-01","fresh":true}
```

- `session_id` يجب أن يطابق العمود الآخر في نفس الحالة.
- `referent_person_id` يجب أن يطابق *الهوية الحالية المعترف بها*، وليس اسم الشخص السابق.
- `fresh` يجب أن يُحسب من **Session Manager موثوق** اعتمادًا على وقت آخر حدث؛ مجرد كتابة `true` في CSV **ليست دليلًا على الحداثة**.
- إذا كان السؤال صريحًا، يمر للـBridge كما هو؛ يظل Face Gate مسؤولًا عن الرفض.
- Step35 يتعرف فقط على صياغات ضمائر ضيقة مثل `Tell me about him/her` و`Who is him/her`، ولا يعتبر حلًا عامًا لفهم السياق.

## بعد تأكيد الاتفاقيات مع أحمد وهناء والـSession Manager

لا تنفذ الأمر التالي حاليًا. لا يُستخدم إلا على Validation مرتبطة فعليًا وبعد مراجعة معنى الـFace Score، والـID Mapping، وFreshness Contract:

```powershell
python experiments/step39_context_aware_runner.py --execute --face-score-contract-confirmed --session-freshness-contract-confirmed
```

يخرج نتيجتين منفصلتين عن نتائج Step32 في `experiments/results/`.

## قيود علمية مهمة
- التصنيف في هذا الـRunner مبني على Step32 **مبدئي** ويحتاج فحص ملكية الـFacts عبر Step37/38 قبل اعتماد التقرير النهائي.
- لا توجد نتائج End-to-End فعلية بدون بيانات أحمد وتقى.
- لا تستخدمي Test القديم بوصفه Blind Held-out جديدًا.
- لا تعدلي `memory_integration.py` أو `e5_demo_bridge_step21.py` أو الـFrozen Threshold بسبب هذه الاختبارات.
- لا ترفعي Face Images أو التسجيلات أو بيانات المرضى الحقيقية إلى GitHub.
