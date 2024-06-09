from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import Levenshtein

# To get matrix result per word
def calculate_similarity_matrix_for_word(dataset, model, num_to_char):
    sentence_accuracies = []
    sentence_precisions = []
    sentence_recalls = []
    sentence_f1_scores = []
    similar_words = []

    for data in dataset:
        yhat = model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75] * yhat.shape[0], greedy=False)[0][0].numpy()

        original_text = tf.strings.reduce_join(num_to_char(data[1][0])).numpy().decode('utf-8')
        # Check the shape of the decoded tensor
        if decoded.ndim == 1:
            # If it's a single character, convert it to a list
            predicted_text = num_to_char(decoded).numpy().decode('utf-8')
        else:
            # If it's a sequence, use reduce_join as before
            predicted_text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')
        print('~' * 100)
        print('Original:', original_text)
        print('Prediction:', predicted_text)
        print('~' * 100)

        # Tokenize sentences into words
        orig_words = original_text.split()
        pred_words = predicted_text.split()

        # Calculate word-level metrics for this sentence
        correct_words = sum(pred_word in orig_words[max(0, i - 1):min(len(orig_words), i + 2)]
                            for i, pred_word in enumerate(pred_words))
        sentence_accuracy = correct_words / max(len(orig_words), len(pred_words))
        sentence_accuracies.append(sentence_accuracy)

        true_positive = sum(1 for i, pred_word in enumerate(pred_words)
        if pred_word in orig_words[max(0, i - 1):min(len(orig_words), i + 2)])
        false_positive = sum(1 for pred_word in pred_words if pred_word not in orig_words)
        false_negative = sum(1 for orig_word in orig_words if orig_word not in pred_words)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0.0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        sentence_precisions.append(precision)
        sentence_recalls.append(recall)
        sentence_f1_scores.append(f1)

        # Find similar words in the predicted sentence
        for i, word in enumerate(pred_words):
            context_indices = range(max(0, i - 1), min(len(orig_words), i + 2))
            context_words = [orig_words[j] for j in context_indices]

            # Calculate similarity percentage using Levenshtein distance
            similarity_percentages = [
                (1 - Levenshtein.distance(word, context_word) / max(len(word), len(context_word))) * 100
                for context_word in context_words
            ]

            best_similarity = max(similarity_percentages)

            similar_words.append((word, context_words, best_similarity))

            # print(f"Predicted Word: '{word}', Similar Words: {context_words}, Similarity Percentage: {best_similarity:.2f}%")

    # Calculate overall sentence-level metrics
    overall_sentence_accuracy = np.mean(sentence_accuracies)
    overall_sentence_precision = np.mean(sentence_precisions)
    overall_sentence_recall = np.mean(sentence_recalls)
    overall_sentence_f1_score = np.mean(sentence_f1_scores)

    print(f"Overall Sentence Accuracy: {overall_sentence_accuracy:.4f}")
    print(f"Overall Sentence Precision: {overall_sentence_precision:.4f}")
    print(f"Overall Sentence Recall: {overall_sentence_recall:.4f}")
    print(f"Overall Sentence F1 Score: {overall_sentence_f1_score:.4f}")

    # Compute and display the confusion matrix
    compute_confusion_matrix_per_word(similar_words)

def compute_confusion_matrix_per_word(similar_words):
    if not similar_words:
        print("No similar words found.")
        return

    labels = sorted(set([word for word, _, _ in similar_words] + [word for _, context, _ in similar_words for word in context]))
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for pred_word, context_words, _ in similar_words:
        for context_word in context_words:
            pred_idx = label_to_index.get(pred_word, -1)
            context_idx = label_to_index.get(context_word, -1)
            if pred_idx != -1 and context_idx != -1:
                cm[pred_idx, context_idx] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Context Words')
    plt.ylabel('Predicted Word')
    plt.title('Confusion Matrix')
    plt.show()

#####################################################################################################

# To get matrix result per character
def calculate_similarity_matrix_for_char(dataset, model, num_to_char):
    true_labels= []
    predicted_labels = []

    for data in dataset:
        yhat = model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75] * yhat.shape[0], greedy=False)[0][0].numpy()

        original_text = tf.strings.reduce_join(num_to_char(data[1][0])).numpy().decode('utf-8')
        # Check the shape of the decoded tensor
        if decoded.ndim == 1:
            # If it's a single character, convert it to a list
            predicted_text = num_to_char(decoded).numpy().decode('utf-8')
        else:
            # If it's a sequence, use reduce_join as before
            predicted_text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')
        print('~' * 100)
        print('Original:', original_text)
        print('Prediction:', predicted_text)
        print('~' * 100)

  # Collect true and predicted labels for the confusion matrix
        min_length = min(len(original_text), len(predicted_text))
        true_labels.extend(list(original_text[:min_length]))
        predicted_labels.extend(list(predicted_text[:min_length]))
    compute_confusion_matrix_per_char(true_labels, predicted_labels)
    calculate_metrics(true_labels, predicted_labels)

def compute_confusion_matrix_per_char(true_labels, predicted_labels):
    if not true_labels or not predicted_labels:
        print("No labels collected for confusion matrix.")
        return

    true_labels_flat = true_labels
    predicted_labels_flat = predicted_labels

    labels = sorted(set(true_labels_flat + predicted_labels_flat))
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for true, pred in zip(true_labels_flat, predicted_labels_flat):
        true_idx = label_to_index.get(true, -1)
        pred_idx = label_to_index.get(pred, -1)
        if true_idx != -1 and pred_idx != -1:
            cm[true_idx, pred_idx] += 1

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_metrics(true_labels, predicted_labels):
    true_labels_flat = true_labels
    predicted_labels_flat = predicted_labels

    # Calculate overall accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(true_labels_flat, predicted_labels_flat)
    precision = precision_score(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=0)
    recall = recall_score(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=0)
    f1 = f1_score(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=0)

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1 Score: {f1:.4f}")

    # Calculate metrics for each character
    labels = sorted(set(true_labels_flat + predicted_labels_flat))
    print("\nMetrics for each character:")
    for label in labels:
        precision = precision_score(true_labels_flat, predicted_labels_flat, labels=[label], average='micro', zero_division=0)
        recall = recall_score(true_labels_flat, predicted_labels_flat, labels=[label], average='micro', zero_division=0)
        f1 = f1_score(true_labels_flat, predicted_labels_flat, labels=[label], average='micro', zero_division=0)
        support = true_labels_flat.count(label)
        print(f"Character: {label}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Support: {support}")
