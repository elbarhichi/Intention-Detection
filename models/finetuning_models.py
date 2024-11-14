from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch


class FineTunedModel:

    def __init__(self):
        # Load the fine-tuned model and tokenizer
        self.model = CamembertForSequenceClassification.from_pretrained("models/fine_tuned_camembert_model")
        self.tokenizer = CamembertTokenizer.from_pretrained("models/fine_tuned_camembert_model")
        self.unique_labels = ['out_of_scope', 'book_flight', 'carry_on', 'flight_status', 'book_hotel', 'lost_luggage', 'translate', 'travel_alert', 'travel_suggestion']


    def predict(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

        # Use the model to get logits
        with torch.no_grad():  # No need to compute gradients for inference
            outputs = self.model(**tokens)
        logits = outputs.logits

        # Get the predicted class index for each input
        predicted_class_indices = torch.argmax(logits, dim=1)

        # Map the indices back to the label names
        predicted_labels = [self.unique_labels[idx] for idx in predicted_class_indices]

        return predicted_labels

