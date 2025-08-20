from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def train_models(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gradient Boosting
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    return gbc, rf, X_train, X_test, y_train, y_test
