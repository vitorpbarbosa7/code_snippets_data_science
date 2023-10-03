import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate the example dataset (unbalanced)
np.random.seed(42)

# Number of samples for each class
n_samples = 40
n_positive = int(0.3 * n_samples)
n_negative = n_samples - n_positive

# Positive class scores (25% below 0.5)
positive_scores = np.concatenate([np.random.uniform(0, 0.5, int(0.25 * n_positive)),
                                  np.random.uniform(0.5, 1, int(0.75 * n_positive))])

# Negative class scores (25% above 0.5)
negative_scores = np.concatenate([np.random.uniform(0, 0.5, int(0.75 * n_negative)),
                                  np.random.uniform(0.5, 1, int(0.25 * n_negative))])

# Combine positive and negative scores to create the full dataset
scores = np.concatenate([positive_scores, negative_scores])
labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])

# Calculate evaluation metrics
true_positives = np.sum((scores >= 0.5) & (labels == 1))
false_positives = np.sum((scores >= 0.5) & (labels == 0))
true_negatives = np.sum((scores < 0.5) & (labels == 0))
false_negatives = np.sum((scores < 0.5) & (labels == 1))

accuracy = (true_positives + true_negatives) / len(scores)
recall = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1_score)

# Plot the boxplot
plt.figure(figsize=(6, 4))
plt.boxplot([positive_scores, negative_scores], labels=["Positive", "Negative"])
plt.xlabel("Class")
plt.ylabel("Scores")
plt.title("Boxplot of Scores for Positive and Negative Classes (Unbalanced)")
plt.show()

# Plot the PDF
plt.figure(figsize=(6, 4))
x = np.linspace(0, 1, 100)
plt.plot(x, norm.pdf(x, np.mean(positive_scores), np.std(positive_scores)), label="Positive")
plt.plot(x, norm.pdf(x, np.mean(negative_scores), np.std(negative_scores)), label="Negative")
plt.xlabel("Score")
plt.ylabel("Probability Density")
plt.title("PDF of Scores for Positive and Negative Classes (Unbalanced)")
plt.legend()
plt.show()

