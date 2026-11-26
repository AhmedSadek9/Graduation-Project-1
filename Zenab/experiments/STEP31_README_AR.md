# Step 31 — A/B/C/D Readiness Preflight

هذا **فحص جاهزية فقط**؛ لا يشغل E5 أو Face Recognition أو STT، ولا يحسب Reliability Loss.

1. انسخي `step31_run_readiness.py` إلى `Zenab/experiments/`.
2. من داخل `WANAS_Team_Repo/Zenab` شغلي:
   `python experiments/step31_run_readiness.py`
3. طالما `CASE_MANIFEST_WORKING.csv` فاضي، النتيجة الصحيحة: A=0, B=0, C=0, D=0 و`WAITING_FOR_REAL_CASE_PAIRINGS`.
4. بعد استلام صور حقيقية وربط الأسئلة، املئي `handoff/step29/CASE_MANIFEST_WORKING.csv` بالقيم الصحيحة وشغلي Step 30 ثم Step 31 مجددًا.
5. تجهيز Run A يتطلب هوية مرجعية موثقة وسؤالًا مرجعيًا وسياقًا وحالة صورة حقيقية مرتبطة. Run B يضيف `face_decision` الحقيقي من أحمد، وRun C يضيف `actual_stt_transcript` الحقيقي من تقى، وRun D يشترط الاثنين لنفس الحالة.
6. لا تضعي `expected_person_id` مكان `face_predicted_person_id`، ولا تملئي الـSTT بنص السؤال المرجعي بغرض تمرير الفحص. قائمة `ready` ليست دليلًا على أداء أو نجاح الاسترجاع.
7. `0.8824205` يظل Frozen Baseline و`0.80` Candidate تجريبي فقط؛ لا تعديل لهما في هذه الخطوة.
8. لا تضيفي صور وجوه حقيقية أو صوتيات أو بيانات مرضى إلى GitHub دون ترتيبات موافقة وخصوصية مناسبة.

**ملاحظة تقنية:** يتطلب هذا الفحص قيمة `face_score` رقمية بين 0 و1 عندما يكون القرار `recognized`، بحسب عقد الـDemo الحالي. لو مخرجات أحمد raw cosine على مقياس آخر، اتفقوا أولًا على Mapping وسياسة gating بدل تغيير القيمة أو تفسيرها كاحتمال.
