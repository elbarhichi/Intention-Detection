
# Détection d'Intention pour un Chatbot Touristique

Ce projet vise à développer un modèle de détection d’intention pour un chatbot d’assistance dans le domaine du tourisme, permettant de classifier les messages utilisateurs selon des intentions prédéfinies (réserver un vol, obtenir des recommandations de voyage, signaler un bagage perdu, etc.).

## Contexte

Le projet se concentre sur le développement d'un système conversationnel intelligent pour gérer les interactions avec les utilisateurs via différents canaux (site web, messagerie instantanée, serveur vocal, application mobile). La détection d’intention est une étape clé pour déclencher des actions adaptées en fonction des besoins exprimés par les utilisateurs.

## Classes d'Intention

Le modèle est conçu pour reconnaître les intentions suivantes :

* `translate` : Traduction d'une phrase dans une autre langue
* `travel_alert` : Alerte de voyage concernant la destination
* `flight_status` : Statut du vol (retard, annulation, etc.)
* `lost_luggage` : Signalement de bagages perdus
* `travel_suggestion` : Recommandation de voyage
* `carry_on` : Informations sur les bagages à main
* `book_hotel` : Réservation d'un hôtel
* `book_flight` : Réservation d'un vol
* `out_of_scope` : Questions hors du domaine touristique

## Approches Utilisées

Le projet a exploré plusieurs approches pour détecter l'intention, notamment :

1. **LLM** : Utilisation des LLM préentrainés avec du prompt engineering
2. **Zero-Shot Classification** : Utilisation de modèles déjà entraînés pour une classification sans apprentissage spécifique.
3. **TF-IDF + Classificateurs** : Utilisation de TF-IDF et de modèles traditionnels de classification.
4. **Embeddings + Classificateurs** : Génération d'embeddings de phrases pour une meilleure représentation contextuelle.
5. **Fine-Tuning** : Ajustement fin d’un modèle pré-entraîné (CamemBERT) pour notre tâche spécifique.

## Installation

1. **Cloner le dépôt :**
   <pre class="!overflow-visible"><code class="!whitespace-pre hljs language-bash">git clone https://github.com/elbarhichi/Intention-Detection
   cd Intention-Detection
   </code></div></div></pre>
2. **Installer les dépendances :**
   Assurez-vous d'avoir Python 3.8+ et installez les bibliothèques nécessaires.
   <pre class="!overflow-visible"><code class="!whitespace-pre hljs language-bash">pip install -r requirements.txt
   </code></div></div></pre>

## Utilisation

Le script `predict_intent.py` permet de prédire l’intention d’un texte utilisateur en utilisant un modèle de classification pré-entraîné, stocké dans le répertoire `models/tfidf/best_svc_model.joblib`.

### Prérequis

Assurez-vous que le modèle `best_svc_model.joblib` est présent dans le dossier `models/tfidf/`. Ce modèle doit être un modèle entraîné et sauvegardé avec `joblib` en utilisant TF-IDF et SVC.

### Exemple d'Utilisation

Pour faire une prédiction, exécutez la commande suivante dans votre terminal, en remplaçant `Votre texte ici` par la phrase pour laquelle vous souhaitez obtenir une prédiction :

<pre class="!overflow-visible"><code class="!whitespace-pre hljs language-bash">python predict_intent.py "Votre texte ici"
</code></div></div></pre>

### Exemple d'Exécution

<pre class="!overflow-visible"><code class="!whitespace-pre hljs language-bash">python predict_intent.py "Je voudrais réserver un vol pour New York"
</code></div></div></pre>

### Résultat attendu

Le script renverra la prédiction de l’intention correspondant à la phrase donnée. Par exemple :

<pre class="!overflow-visible"><code class="!whitespace-pre hljs language-">Intention prédite : book_flight</code></div></div></pre>
