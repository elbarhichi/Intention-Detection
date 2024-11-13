from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class BartModel_withDescription:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/bart-large-mnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
            "translate": "aider à traduire une phrase pour faciliter la communication en voyage",
            "travel_alert": "informer sur les alertes ou restrictions de voyage pour la destination",
            "flight_status": "donner le statut du vol, comme les retards ou annulations",
            "lost_luggage": "aider en cas de perte ou de bagages manquants",
            "travel_suggestion": "offrir des recommandations de destinations ou de voyage",
            "carry_on": "informer sur les règles de bagages à main, taille et poids",
            "book_hotel": "aider à réserver un hôtel pour le séjour",
            "book_flight": "aider à réserver un vol pour une destination",
            "out_of_scope": "une demande qui n'est pas liée au voyage ou à l'assistance"
        }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance touristique, ce message concerne probablement : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions




class BartModel_withParaphrasing:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/bart-large-mnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
                "translate": "traduction",
                "travel_alert": "alerte de voyage",
                "flight_status": "statut de vol",
                "lost_luggage": "bagages perdus",
                "travel_suggestion": "suggestion de voyage",
                "carry_on": "bagage à main",
                "book_hotel": "réservation d'hôtel",
                "book_flight": "réservation de vol",
                "out_of_scope": "hors sujet"
            }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance pour une agence de tourisme, ce message concerne probablement l'intention suivante : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions




class DistilBertModel:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/distilbert-base-uncased-mnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
            "translate": "aider à traduire une phrase pour faciliter la communication en voyage",
            "travel_alert": "informer sur les alertes ou restrictions de voyage pour la destination",
            "flight_status": "donner le statut du vol, comme les retards ou annulations",
            "lost_luggage": "aider en cas de perte ou de bagages manquants",
            "travel_suggestion": "offrir des recommandations de destinations ou de voyage",
            "carry_on": "informer sur les règles de bagages à main, taille et poids",
            "book_hotel": "aider à réserver un hôtel pour le séjour",
            "book_flight": "aider à réserver un vol pour une destination",
            "out_of_scope": "une demande qui n'est pas liée au voyage ou à l'assistance"
        }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance touristique, ce message concerne probablement : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions
    


class DistilBertModel_withParaphrasing:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/distilbert-base-uncased-mnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
                "translate": "traduction",
                "travel_alert": "alerte de voyage",
                "flight_status": "statut de vol",
                "lost_luggage": "bagages perdus",
                "travel_suggestion": "suggestion de voyage",
                "carry_on": "bagage à main",
                "book_hotel": "réservation d'hôtel",
                "book_flight": "réservation de vol",
                "out_of_scope": "hors sujet"
            }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance pour une agence de tourisme, ce message concerne probablement l'intention suivante : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions



class MDeBertaModel:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/mDeBERTa-v3-base-mnli-xnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
            "translate": "aider à traduire une phrase pour faciliter la communication en voyage",
            "travel_alert": "informer sur les alertes ou restrictions de voyage pour la destination",
            "flight_status": "donner le statut du vol, comme les retards ou annulations",
            "lost_luggage": "aider en cas de perte ou de bagages manquants",
            "travel_suggestion": "offrir des recommandations de destinations ou de voyage",
            "carry_on": "informer sur les règles de bagages à main, taille et poids",
            "book_hotel": "aider à réserver un hôtel pour le séjour",
            "book_flight": "aider à réserver un vol pour une destination",
            "out_of_scope": "une demande qui n'est pas liée au voyage ou à l'assistance"
        }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance touristique, ce message concerne probablement : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions



class MDeBertaModel_withParaphrasing:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/mDeBERTa-v3-base-mnli-xnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
                "translate": "traduction",
                "travel_alert": "alerte de voyage",
                "flight_status": "statut de vol",
                "lost_luggage": "bagages perdus",
                "travel_suggestion": "suggestion de voyage",
                "carry_on": "bagage à main",
                "book_hotel": "réservation d'hôtel",
                "book_flight": "réservation de vol",
                "out_of_scope": "hors sujet"
            }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance pour une agence de tourisme, ce message concerne probablement l'intention suivante : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions




class MDeBertaModelMulti:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
            "translate": "aider à traduire une phrase pour faciliter la communication en voyage",
            "travel_alert": "informer sur les alertes ou restrictions de voyage pour la destination",
            "flight_status": "donner le statut du vol, comme les retards ou annulations",
            "lost_luggage": "aider en cas de perte ou de bagages manquants",
            "travel_suggestion": "offrir des recommandations de destinations ou de voyage",
            "carry_on": "informer sur les règles de bagages à main, taille et poids",
            "book_hotel": "aider à réserver un hôtel pour le séjour",
            "book_flight": "aider à réserver un vol pour une destination",
            "out_of_scope": "une demande qui n'est pas liée au voyage ou à l'assistance"
        }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance touristique, ce message concerne probablement : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions




class MDeBertaModelMulti_withParaphrasing:
    def __init__(self):
        
        # Charger le modèl
        model_directory = "models/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    
        self.classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        # Définir les descriptions de labels pour le chatbot touristique en français
        self.label_descriptions = {
                "translate": "traduction",
                "travel_alert": "alerte de voyage",
                "flight_status": "statut de vol",
                "lost_luggage": "bagages perdus",
                "travel_suggestion": "suggestion de voyage",
                "carry_on": "bagage à main",
                "book_hotel": "réservation d'hôtel",
                "book_flight": "réservation de vol",
                "out_of_scope": "hors sujet"
            }


        # Inverser le dictionnaire pour mapper les descriptions de retour aux labels initiaux
        self.original_labels = {v: k for k, v in self.label_descriptions.items()}

    def predict(self, liste_text):

        predictions = []

        
        results = self.classifier(
            liste_text,
            candidate_labels=list(self.label_descriptions.values()),
            hypothesis_template="Dans un chatbot d'assistance pour une agence de tourisme, ce message concerne probablement l'intention suivante : {}."
        )

        for result in results:
            best_label_description = result['labels'][0]
            best_original_label = self.original_labels[best_label_description]
            predictions.append(best_original_label)

        return predictions



