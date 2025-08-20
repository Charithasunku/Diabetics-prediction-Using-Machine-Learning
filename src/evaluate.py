from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, report
