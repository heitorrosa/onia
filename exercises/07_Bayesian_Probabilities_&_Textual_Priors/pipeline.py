import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

def load_data():
    dataset = pd.read_csv("exercises/07_Bayesian_Probabilities_&_Textual_Priors/spam.csv", encoding='latin-1')
    dataset = dataset[['v1', 'v2']]
    dataset['v1'] = dataset['v1'].map({'ham': 0, 'spam': 1})

    return dataset['v2'], dataset['v1']

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=4096)),
        ('clf', MultinomialNB(class_prior=[0.866, 0.134]))
    ])
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
    
    y_pred = (probs >= best_threshold).astype(int)
    auprc = auc(recall, precision)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f"\n auprc: {auprc:.4f}")

    plt.figure(figsize=(10, 5))
    
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score Trajectory', color='tab:blue', linewidth=2)
    plt.axvline(x=best_threshold, linestyle='--', color='red', alpha=0.7, label=f'Best Threshold ({best_threshold:.2f})')
    
    plt.title('Bayesian Probability Threshold Mapping (Inference Decay)')
    plt.xlabel('Spam Probability Threshold')
    plt.ylabel('F1 Score (Harmonic Mean)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("exercises/07_Bayesian_Probabilities_&_Textual_Priors/threshold_mapping.png")