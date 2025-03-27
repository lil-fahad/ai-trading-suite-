
from sklearn.metrics import mean_absolute_error

def select_best_model(y_test, preds_dict):
    errors = {name: mean_absolute_error(y_test, pred) for name, pred in preds_dict.items()}
    best_model = min(errors, key=errors.get)
    return best_model, errors[best_model]
