# مصادر public-apis المستخدمة

تمت مراجعة [مستودع public-apis](https://github.com/public-apis/public-apis) واختيار ثلاثة مصادر مرتبطة مباشرة بمشروع التداول. المستودع المرجعي هو كتالوج مجتمعي، وليس طبقة ضمان لجودة البيانات أو استمرار الخدمة؛ لذلك يحتفظ المشروع بموصلات مستقلة وبإعدادات مفاتيح خارج Git.

| المصدر | الاستخدام في المشروع | القيد التشغيلي |
|---|---|---|
| [Twelve Data](https://twelvedata.com/docs) | شموع OHLCV داخل اليوم لزوجي EUR/USD وGBP/JPY عبر `time_series`، بفواصل مثل 5min. | يتطلب مفتاح API وتخضع الطلبات للخطة وحدود المعدل؛ يجب حفظ البيانات الخام وتاريخ التنزيل. |
| [Frankfurter](https://frankfurter.dev/v1/) | أسعار صرف مرجعية يومية وسلسلة زمنية لفحوص sanity-check والـ benchmark، وليس بديلاً لبيانات M1/M5. | بيانات مرجعية يومية وليست تنفيذية أو Tick/M5. |
| [GNews](https://docs.gnews.io/endpoints/search-endpoint) | سياق الأخبار قبل تمرير النص إلى FinBERT؛ يدعم البحث بالكلمات والتاريخ والصفحات. | يتطلب مفتاح API، والنتائج وحدود المحتوى تعتمد على الخطة؛ لا تُعامل الأخبار كإشارة سعرية مباشرة. |

تؤكد وثائق OANDA أن فواصل الشموع تشمل M1 وM5، ولذلك يبقى موصل OANDA التاريخي السابق خياراً مناسباً عند توافر بيانات اعتماد المستخدم. أما Yahoo Finance فليس أساساً مناسباً لطلب عامين من M1/M5؛ صفحة المساعدة الرسمية تربط تنزيل التاريخ باشتراك Gold ولا تضمن الفواصل الدقيقة المطلوبة.

## نتيجة تدريب تجريبية

على بيانات العينة الاصطناعية المحلية فقط، انتهى GridSearch الزمني إلى `learning_rate=0.07` و`max_depth=3` و`n_estimators=100`. أُنشئ مخطط `top10_features.png` خارج المستودع أثناء التحقق، وكان أعلى الخصائص الظاهرة `return_20` و`rsi_21` و`keltner_width`. هذه ليست نتيجة أداء سوق حقيقي ولا ينبغي استخدامها كدليل صلاحية.

## مصادر

1. [public-apis/public-apis — README](https://github.com/public-apis/public-apis)
2. [Twelve Data Documentation](https://twelvedata.com/docs)
3. [Frankfurter API Documentation](https://frankfurter.dev/v1/)
4. [GNews Search Endpoint](https://docs.gnews.io/endpoints/search-endpoint)
5. [OANDA Instrument Definitions](https://developer.oanda.com/rest-live-v20/instrument-df/)
6. [Yahoo Finance historical data help](https://help.yahoo.com/kb/download-historical-data-yahoo-finance-sln2311.html)

## تحقق حي

في 28 أغسطس 2026 أعاد endpoint `https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD,GBP,JPY` حمولة JSON صالحة بتاريخ آخر يوم عمل ظاهر في الاستجابة. استُخدم هذا التحقق للتأكد من مسار الاتصال فقط؛ لم تُستخدم الأسعار الناتجة لتقييم الاستراتيجية أو لفتح أي صفقة.
