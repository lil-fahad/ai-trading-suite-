
def explain_decision(pred_price, indicators, openai_api_key):
    import openai
    openai.api_key = openai_api_key

    prompt = f"""
You are a financial AI analyst. Based on the following indicators and predicted price, explain in simple terms why the model might suggest a buy or sell action.

Predicted Price: {pred_price}
Indicators:
- RSI: {indicators['rsi']}
- MACD: {indicators['macd']}
- SMA_10: {indicators['sma']}
- EMA_20: {indicators['ema']}

Provide a 2-3 sentence explanation of the decision.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "GPT explanation not available."
