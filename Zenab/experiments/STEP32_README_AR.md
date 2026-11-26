# WANAS — Step 32: Paired A/B/C/D Evaluation Runner

## هدف الخطوة
بعد Step 31 (0 حالات مرتبطة)، جهزنا Runner حقيقي لكن **لن يختلق صورة أو STT أو نتيجة**. يقرأ `handoff/step29/CASE_MANIFEST_WORKING.csv` ويربط الحالات الموجودة في Validation فقط. لا يشغل الموديل على Manifest فارغ.

## النقل والتشغيل الآن
انقلي `step32_run_paired_evaluation.py` و`test_step32_runner.py` إلى `Zenab/experiments/`.

```powershell
cd "$HOME\Desktop\WANAS_Team_Repo\Zenab"
python experiments/test_step32_runner.py
python experiments/step32_run_paired_evaluation.py
```

الاختبار الأول يستخدم Fake Bridge **لمنطق التقييم فقط**. الثاني يعمل Preflight لا يشغل E5 ولا يغير ملفًا.

## بعد وصول بيانات الفريق
1. هناء وأحمد يراجعان أن الـ`expected_person_id` في الحالات مرتبط بصورة مناسبة، وأن الـ`face_predicted_person_id` نتيجة فعلية لا Ground Truth.
2. أحمد يؤكد رسميًا معنى `face_score` وهل مناسب للـ0..1 gate المستخدم في E5 Demo Bridge. إن كان Cosine/Score مقياسه مختلف، لا تشغلي B/D قبل تعديل عقد الربط باتفاق الفريق.
3. تقى تزود `actual_stt_transcript` مع `utterance_id`, `stt_model_version` وProvenance.
4. شغّلي `step30_validate_case_manifest.py` ثم `step31_run_readiness.py` ثم اختبارات Step32.
5. بعد اعتماد المصدر، وللتجربة المحلية **فقط**:

```powershell
python experiments/step32_run_paired_evaluation.py --execute --face-score-contract-confirmed
```

يخرج `experiments/results/step32_paired_validation_predictions.jsonl` و`step32_paired_validation_report.json` عند وجود حالات فعلية سليمة. يحسب Reliability Loss فقط على الحالات المشتركة A وD. استخدمي `python` من البيئة التي بها E5 cached؛ لا يلزم تنزيل جديد.

## حدود مهمة
- Step32 يستخدم **E5 Experimental Demo Adapter** بعتبة `0.8824205`، وليس الـRule-Based Production Bridge.
- الـ`conversation_context` محفوظ في النتائج لكنه **لا يُدخل إلى الـE5 question-only query**. لذلك هذا ليس اختبارًا نهائيًا لقدرة فهم السياق. تجنبي سؤالًا يعتمد على مرجع غامض مثل “Tell me about him” قبل تنفيذ سياق مؤكد.
- إشارة `--face-score-contract-confirmed` ليست إثباتًا آليًا؛ لا تُستخدم إلا بعد مراجعة الفريق.
- لا ترفعي صور وجوه حقيقية أو تسجيلات أو بيانات أشخاص حقيقية إلى GitHub. 
- الـ40 سؤالًا AI-assisted ما زالت بحاجة لمراجعة، والـTest القديم ليس Held-out جديدًا.
