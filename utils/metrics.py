from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_test, y_pred):
    try:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    except ValueError as e:
        mse = None
        r2 = None
        print(f"Error calculating metrics: {e}")
    return mse, r2
