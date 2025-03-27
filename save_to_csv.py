
import pandas as pd

def save_predictions_to_csv(preds, actuals, filename="forecast.csv"):
    df = pd.DataFrame({"Predicted": preds, "Actual": actuals})
    df.to_csv(filename, index=False)
    return filename
