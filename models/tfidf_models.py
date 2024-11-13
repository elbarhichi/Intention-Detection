import joblib

class SVCModel:
    def __init__(self):
        model_path = 'models/tfidf/best_svc_model.joblib'
        # Charger le modèle depuis le fichier
        self.model = joblib.load(model_path)

    def predict(self, X):
        # Faire des prédictions
        return self.model.predict(X)
    


class LogisticRegressionModel:
    def __init__(self):
        model_path = 'models/tfidf/best_logistic_regression_model.joblib'
        # Charger le modèle depuis le fichier
        self.model = joblib.load(model_path)

    def predict(self, X):
        # Faire des prédictions
        return self.model.predict(X)
    
    


class RandomForestModel:
    def __init__(self):
        model_path = 'models/tfidf/best_random_forest_model.joblib'
        # Charger le modèle depuis le fichier
        self.model = joblib.load(model_path)

    def predict(self, X):
        # Faire des prédictions
        return self.model.predict(X)
    



class KNNModel:
    def __init__(self):
        model_path = 'models/tfidf/best_knn_model.joblib'
        # Charger le modèle depuis le fichier
        self.model = joblib.load(model_path)

    def predict(self, X):
        # Faire des prédictions
        return self.model.predict(X)
    

