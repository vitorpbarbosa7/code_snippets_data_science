import numpy as np

def log_loss(y_true, y_pred):
    # Make sure the input arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Number of samples
    n_samples = len(y_true)

    # Initialize the total log loss
    total_log_loss = 0.0

    # Loop through each sample
    for i in range(n_samples):
        # Get the true class probabilities and predicted class probabilities for this sample
        true_probs = y_true[i]
        pred_probs = y_pred[i]

        # Ensure that predicted probabilities are valid (between 0 and 1)
        if not np.all((0 <= pred_probs) & (pred_probs <= 1)):
            raise ValueError("Predicted probabilities must be in the range [0, 1].")

        # Normalize the predicted probabilities to sum to 1
        pred_probs /= np.sum(pred_probs)

        # Calculate the log loss for this sample
        sample_log_loss = -np.sum(true_probs * np.log(pred_probs))

        # Add the sample log loss to the total
        total_log_loss += sample_log_loss

    # Calculate the average log loss across all samples
    average_log_loss = total_log_loss / n_samples

    return average_log_loss

# Example usage:
# Define true class probabilities and predicted class probabilities for three classes
y_true = np.array([[1.0, 0.0, 0.0],  # Sample 1
                   [0.0, 1.0, 0.0],  # Sample 2
                   [0.0, 0.0, 1.0],  # Sample 3
                   [0.3, 0.4, 0.3]]) # Sample 4

y_pred = np.array([[0.9, 0.1, 0.0],  # Sample 1
                   [0.1, 0.8, 0.1],  # Sample 2
                   [0.0, 0.1, 0.9],  # Sample 3
                   [0.2, 0.5, 0.3]]) # Sample 4

# Calculate the Log Loss
loss = log_loss(y_true, y_pred)
print("Log Loss:", loss)

