from src.preprocess import load_and_preprocess_data
from src.train import train_models
from src.evaluate import evaluate_model

def main():
    # Load data
    X, y = load_and_preprocess_data("data/diabetes.csv")

    # Train models
    gbc, rf, X_train, X_test, y_train, y_test = train_models(X, y)

    # Evaluate Gradient Boosting (best threshold 0.55)
    gbc_acc, gbc_report = evaluate_model(gbc, X_test, y_test, threshold=0.55)
    print("\n--- Gradient Boosting Classifier (Best Threshold 0.55) ---")
    print("Accuracy:", round(gbc_acc, 4))
    print("Report:\n", gbc_report)

    # Evaluate Random Forest (best threshold 0.48)
    rf_acc, rf_report = evaluate_model(rf, X_test, y_test, threshold=0.48)
    print("\n--- Random Forest Classifier (Best Threshold 0.48) ---")
    print("Accuracy:", round(rf_acc, 4))
    print("Report:\n", rf_report)

if __name__ == "__main__":
    main()
