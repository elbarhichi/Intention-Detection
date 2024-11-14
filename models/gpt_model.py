from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class GPTModel:
    def __init__(self):

        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
        self.prompt_system = {
            "role": "system",
            "content": """
Tu es un assistant spécialisé dans la classification des intentions des utilisateurs dans un chatbot d'assistance pour le domaine du tourisme.
Le chatbot doit classer les messages des utilisateurs dans une des intentions suivantes :

1. `translate` : L’utilisateur souhaite traduire une phrase dans une autre langue.
   - Exemple : Pouvez-vous me dire comment dire «je ne parle pas beaucoup espagnol», en espagnol
   - Exemple : Dites-moi comment dire: «C'est une belle matinée» en italien
   
2. `travel_alert` : L’utilisateur demande si sa destination de voyage est concernée par une alerte ou restriction de sécurité.
   - Exemple : La Corée du Nord a-t-elle des alertes de voyage que je devrais être conscientes
   - Exemple : Y a-t-il des alertes de voyage pour la Syrie

3. `flight_status` : L’utilisateur souhaite obtenir des informations sur le statut de son vol (retard, annulation, etc.).
   - Exemple : Y a-t-il des nouvelles sur le vol DL123
   - Exemple : Quand allons-nous commencer à monter à bord de mon vol prévu

4. `lost_luggage` : L’utilisateur signale la perte de ses bagages lors d'un voyage.
   - Exemple : Localiser les bagages perdus de Flight America Airlines à O'Hare
   - Exemple : Perte des bagages sur Flight America Airlines à O'Hare

5. `travel_suggestion` : L’utilisateur demande une recommandation de destination ou des suggestions de voyage.
   - Exemple : Veuillez suggérer des activités touristiques amusantes à Tokyo
   - Exemple : Si je vais à Evans, que dois-je faire

6. `carry_on` : L’utilisateur souhaite obtenir des informations sur les règles de bagages à main pour son vol.
   - Exemple : Voulez-vous me faire part des restrictions de bagage à main pour American Airlines
   - Exemple : Restrictions de cabine pour les émirats arabes unis

7. `book_hotel` : L’utilisateur souhaite réserver un hôtel pour son séjour.
   - Exemple : Réservez-moi une chambre du 11 au 15 novembre à Cali
   - Exemple : Trouvez-moi un endroit où séjourner à Cali du 11 au 15 novembre

8. `book_flight` : L’utilisateur souhaite réserver un vol vers une destination spécifique.
   - Exemple : Trouvez-moi des vols aller-retour de LAX vers SFOX
   - Exemple : Réservez un vol de Chicago à DC lundi et revenant mercredi

9. `out_of_scope` : L’utilisateur pose une question qui n’est pas liée aux services de voyage ou de tourisme proposés.


Pour chaque message utilisateur, analyse l'intention et réponds uniquement par le label correspondant parmi les options ci-dessus. Ne génère aucun texte additionnel en dehors du label.
"""
        }

    def predict(self, liste_text):

        y_pred = []
        for text in liste_text:
            user_prompt = {
                "role": "user",
                "content": f"Message de l'utilisateur : \"{text}\". Réponds uniquement par le label correspondant parmi les labels [translate, travel_alert, flight_status, lost_luggage, travel_suggestion, carry_on, book_hotel, book_flight, out_of_scope]."
            }
            
        
            # Appel à l'API Azure OpenAI pour obtenir la prédiction
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[self.prompt_system, user_prompt],
                temperature=0.0,  # Réduire la température pour plus de cohérence dans les réponses
                n=1
            )
            
            # Extraire le label de la réponse
            predicted_label = response.choices[0].message.content.strip()

            list_mots = predicted_label.split(' ')
            if len(list_mots) > 1:
                # extraire le label dans le text si le label est dans le text
                labels = ['translate', 'travel_alert', 'flight_status', 'lost_luggage', 'travel_suggestion', 'carry_on', 'book_hotel', 'book_flight']
            
                for label in labels:
                    if label in list_mots:
                        predicted_label = label
                        break
                    else:
                        predicted_label = 'out_of_scope'

            y_pred.append(predicted_label)

        return y_pred