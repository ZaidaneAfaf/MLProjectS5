import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def main():
    print("🌸 Starting Iris ML Training...")
    
    # Chargement des données
    print("📊 Loading dataset...")
    df = pd.read_csv('Iris.csv')
    
    # Préparation des données
    X = df.drop(['Species'], axis=1)
    y = df['Species']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraînement
    print("🤖 Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy of the model: {accuracy:.2f}")
    
    # Sauvegarde du modèle
    os.makedirs('artifacts', exist_ok=True)
    model_path = 'artifacts/iris_model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"💾 Model saved as {model_path}")
    
    # Sauvegarde des métriques
    metrics = {
        'accuracy': accuracy,
        'model_type': 'RandomForestClassifier',
        'features': list(X.columns)
    }
    
    with open('artifacts/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("📈 Metrics saved!")
    return accuracy

if __name__ == "__main__":
    main()