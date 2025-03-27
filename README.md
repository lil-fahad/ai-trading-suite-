# AI Trading Suite - Super Final Edition

## الميزات:

- جلب بيانات الأسهم والخيارات من Polygon.io
- تحليل فني ذكي (SMA, MACD, Bollinger, وغيرها)
- توصيات شراء/بيع مع شرح GPT
- Backtesting للاستراتيجيات
- تصدير توصيات CSV
- تنبيهات تليجرام
- دعم للمؤشرات الاقتصادية (فائدة، تضخم، إلخ)

## تشغيل المشروع:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## إعداد:

- أنشئ ملف `.env` يحتوي على:
```
POLYGON_API_KEY=your_polygon_key
OPENAI_API_KEY=your_openai_key
```