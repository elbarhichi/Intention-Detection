import joblib
import sys

class SVCModel:
    def __init__(self):
        model_path = 'models/tfidf/best_svc_model.joblib'
        # Charger le modèle depuis le fichier
        self.model = joblib.load(model_path)

    def predict(self, X):
        # Faire des prédictions
        return self.model.predict(X)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_intent.py 'votre texte ici'")
        sys.exit(1)

    input_text = sys.argv[1]
    # Créer une instance de SVCModel
    model = SVCModel()
    # Appeler predict avec une liste contenant `input_text`
    intent = model.predict([input_text])
    print(f"Intention prédite : {intent[0]}")
