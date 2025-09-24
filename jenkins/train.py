# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
import sys

def main():
    # Configuration de l'encodage pour Windows
    if sys.platform.startswith('win'):
        import codecs
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        except:
            pass  # Si ça échoue, on continue sans
    
    print("Starting Iris ML Training...")
    
    # Chargement des données
    print("Loading dataset...")
    
    # Vérifier plusieurs chemins possibles pour le dataset
    dataset_paths = [
        'Iris.csv',
        '../data/Iris.csv', 
        './data/Iris.csv'
    ]
    
    df = None
    for path in dataset_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"Dataset loaded from: {path}")
                break
        except Exception as e:
            continue
    
    if df is None:
        print("ERROR: Could not find Iris.csv dataset")
        print("Available files in current directory:")
        print(os.listdir('.'))
        if os.path.exists('../data'):
            print("Available files in ../data:")
            print(os.listdir('../data'))
        return None
    
    # Préparation des données
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Supprimer la colonne 'Id' si elle existe
    if 'Id' in df.columns:
        X = df.drop(['Id', 'Species'], axis=1)
    else:
        X = df.drop(['Species'], axis=1)
    
    y = df['Species']
    
    print(f"Features: {list(X.columns)}")
    print(f"Classes: {y.unique()}")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraînement
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the model: {accuracy:.2f}")
    
    # Sauvegarde du modèle
    os.makedirs('artifacts', exist_ok=True)
    model_path = 'iris_model.pkl'  # Sauvegarde dans le répertoire courant
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved as {model_path}")
    
    # Sauvegarde des métriques
    metrics = {
        'accuracy': accuracy,
        'model_type': 'RandomForestClassifier',
        'features': list(X.columns)
    }
    
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("Metrics saved!")
    print("Training completed successfully!")
    
    return accuracy

if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)
    else:
        print(f"Final accuracy: {result:.2f}")
        sys.exit(0)