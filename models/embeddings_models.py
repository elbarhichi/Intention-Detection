import cloudpickle

class SVCModel:
    def __init__(self):
        with open('models/embeddings/embedding_svc_model.pkl', 'rb') as f:
            loaded_pipeline = cloudpickle.load(f)

        # Charger le modèle depuis le fichier
        self.model = loaded_pipeline

    def predict(self, X):
        # Faire des prédictions
        return self.model.predict(X)
