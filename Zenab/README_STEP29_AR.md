# WANAS Step 29 — البداية

1. انسخي `step29_prepare_case_pool.py` إلى `Zenab/experiments/` (لو داخل الـTeam Repo) أو `experiments/` (لو داخل نسخة WANAS_Zinab_Final).
2. افتحي PowerShell من جذر مشروع زينب، وشغلي `python experiments/step29_prepare_case_pool.py`.
3. هيظهر `handoff/step29/STEP29_QUESTION_POOL.csv` فيه الـ140 سؤال الموجودين أصلًا، و`CASE_MANIFEST_WORKING.csv` فاضي إلا من أسماء الأعمدة، و`STEP29_POOL_SUMMARY.json`.
4. **متحطيش `image_id` أو `utterance_id` من عندك.** لازم أرقام حقيقية من أحمد/تقى ومراجعة هناء، مع Session/Context موثقين.
5. الـ140 سؤال فيها 78 Train و26 Validation و36 Test حسب الـManifest الأصلية؛ الـTest القديم سبق استخدامه، فلا تقدميه كـIndependent Held-Out جديد.
6. الملف العامل مش هيتمسح لو شغلتي السكريبت مرة تانية؛ لكن Question Pool يُعاد توليده من المصدر. الـFrozen Threshold والـRetrieval لا يتغيران.
7. قبل GitHub، ارفعي فقط الـQuestion Pool والقالب والسكريبت والملخص (Fictional). لا ترفعي بيانات وجوه/صوت حقيقية أو identifiers حساسة.

بعد تجهيز الـQuestion Pool، الخطوة التالية هي **الربط بالحالات الفعلية** وكتابة Validator للـCase Manifest، وبعدها Runner لـA/B/C/D باستخدام Outputs حقيقية.
