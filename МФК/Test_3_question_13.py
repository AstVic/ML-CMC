import numpy as np


class KNN_classifier:
    def __init__(self, n_neighbors: int, **kwargs):
        self.K = n_neighbors

    def fit(self, x: np.array, y: np.array):
        self.X_train = x
        self.y_train = y
        self.is_fitted = True

    def predict(self, x: np.array):
        predictions = []
        for test_sample in x:
            distances = np.linalg.norm(self.X_train - test_sample, axis=1)
            nearest_indices = np.argsort(distances)[:self.K]
            nearest_labels = self.y_train[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)
        
        predictions = np.array(predictions)
        return predictions
