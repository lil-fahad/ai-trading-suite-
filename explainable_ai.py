
import openai
import os

def explain_decision(pred_price, indicators):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
Given:
- Predicted price: {pred_price}
- Indicators: {indicators}
Explain the forecast in clear terms.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except:
        return "Explanation unavailable right now."
