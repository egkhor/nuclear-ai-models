import os
os.environ["TRANSFORMERS_BACKEND"] = "tensorflow"
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import classification_report

# Simulate social media data on nuclear power
np.random.seed(42)
data = {
    'text': [
        "Nuclear power is safe and clean for Malaysia!",
        "Worried about radiation risks in nuclear plants.",
        "SMRs could power our data centers sustainably.",
        "Nuclear waste is a big problem, no way!",
        "Excited for Malaysia’s nuclear future."
    ] * 200,
    'sentiment': np.random.choice(['positive', 'negative', 'neutral'], 1000, p=[0.4, 0.3, 0.3])
}
df = pd.DataFrame(data)

# Load BERT-based sentiment classifier
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Predict sentiment
predictions = classifier(df['text'].tolist())
df['predicted'] = [p['label'].lower() for p in predictions]

# Evaluate model
print(classification_report(df['sentiment'], df['predicted']))

# Save predictions
df.to_csv('sentiment_predictions.csv', index=False)
