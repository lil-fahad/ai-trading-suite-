
import pandas as pd

def save_predictions_to_csv(predictions, actuals, filename="forecast_output.csv"):
    df = pd.DataFrame({
        "Predicted": predictions,
        "Actual": actuals
    })
    df.to_csv(filename, index=False)
    return filename
