# WANAS — Step 16: Freeze & Independent Test Preparation

المرحلة دي **تحضير فقط، مش تقييم نهائي**. السكربت لا يغيّر الداتا ولا يعيد تنزيل E5.

## مكان الملفات
ضع الملفين `step16_prepare_evaluation.py` و`step16_validate_blind_questions.py` داخل فولدر `experiments/` في `WANAS_Zinab_Evaluation_Updated`.

## 1) التحضير

```powershell
python experiments/step16_prepare_evaluation.py
```

النتيجة داخل `experiments/step16/`:

- `frozen_e5_config_step16.json`: snapshot لإعداد **E5 + question_only + fact_type_guard**، بالـthreshold المأخوذ من `multilingual_e5_small_report.json`، وSHA-256 للملفات. **اختيار مؤقت للتقييم المستقل، ليس اعتمادًا إنتاجيًا.**
- `blind_question_authoring_template.csv`: أربعون **مكانًا فارغًا** لأسئلة مستقبلية: 20 تستهدف حقائق معروفة و20 غير قابلة للإجابة. وجود السطر لا يعني وجود سؤال. يكتب الأسئلة شخص مستقل، ويراجعها شخص آخر قبل أول تقييم؛ لا نعيد استخدام الأسئلة القديمة أو أمثلتها المنسوخة.
- `toqa_handoff_draft.csv`: ثلاثة أمثلة لتقى — إجابة تجريبية صحيحة، غياب معلومة، هوية غير معروفة — مع IDs وسبب الرفض، كلها **synthetic/demo** وليست نتائج صوت حقيقية ولا إثبات إرسال.
- `prior_question_norms.json`: نصوص الأسئلة القديمة بعد التطبيع للمساعدة في كشف تكرار السؤال حرفيًا. لا يكتشف التسريب المعنوي أو إعادة الصياغة المتقاربة.

**هام:** السكربت لن يستبدل ملفات Step 16 لو كانت موجودة، لحماية بيانات المراجعين والتجميد الأول.

## 2) تعبئة الأسئلة المستقلة

افتح `blind_question_authoring_template.csv` باستخدام Excel أو محرر CSV. لا تغيّر `new_query_id`, `person_id`, `fact_source_id_for_author` أو `context_scenario` من غير مراجعة. لكل سؤال مكتوب املأ:

- `question`: سؤال جديد مستقل **لا ينقل صياغة سؤال قديم**.
- `expected_outcome`: `answerable` أو `unanswerable`.
- `acceptable_fact_ids`: مصفوفة JSON مثل `["person-001-F01"]` أو `[]`.
- `original_question_id`: هوية السؤال الأصلي؛ كل الـparaphrases لازم تبقى في نفس التقسيم.
- `independent_author`, `independent_reviewer` (شخصان مختلفان), `review_approval=approved`.
- `origin_id`: وصف مستقل موثّق لمصدر السؤال الجديد، وليس `source_xlsx` إلا لو السؤال مأخوذ منه فعلًا.

بالنسبة للأسئلة غير القابلة للإجابة لازم المُراجع يؤكد إن كل الـ120 Fact لا تدعم الإجابة. ومافيش أسئلة مصنفة Verified Real Patient قبل وجود موافقة Caregiver حقيقية.

## 3) المراجعة

```powershell
python experiments/step16_validate_blind_questions.py
```

اختبار الملف الفارغ **لازم يفشل** برسالة `NOT READY` — ده سلوك صحيح. بعد ملء الأسئلة ومراجعتها قد ينجح الـstructural check، لكن لا يثبت الاستقلال الدلالي أو قوة الاختبار. ما نختبرش على الأسئلة الجديدة ثم نغيّر القواعد بناءً على نتائجها ونسمّيها held-out.

## 4) ملاحظات للمشاركة والتسليم

- `question_only` اختيار baseline محافظ لأن تجربة `last_request` الصناعية ظهرت فيها إجابات خاطئة قبل الـGuard. Step 15 exploratory مش مُعتمد قبل اختبار مستقل.
- لازم بعد تثبيت الإعداد يتعمل **runner منفصل** يطبق E5 + guard على أسئلة المراجعة؛ السكربت ده لا يدّعي إنه حسب نتائج جديدة.
- تقرير الـloss من مخرجات Face Recognition/STT الحقيقية ما زال **Pending** إلى حين استلام بيانات أحمد وتقى، وتأكيد mapping من هناء.
