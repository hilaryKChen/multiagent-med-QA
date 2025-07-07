import json
import pandas as pd
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_predictions(predictions_file: str) -> Dict[str, int]:
    """Load CrewAI predictions from file"""
    with open(predictions_file, 'r') as f:
        return json.load(f)

def evaluate_accuracy(true_labels: List[int], pred_labels: List[int]):
    """Calculate evaluation metrics"""


    correctness = [1 if true == pred else 0 for true, pred in zip(true_labels, pred_labels)]
    accuracy = sum(correctness) / len(correctness)
    return accuracy
    return {
        'accuracy': accuracy,
    }

def evaluate_confidence(true_labels: List[int], pred_labels: List[int], correct_confidence: List[float]):
    """Calculate confidence metrics"""
    # Calculate confidence metrics
    correctness = [1 if true == pred else 0 for true, pred in zip(true_labels, pred_labels)]
    correct_confidence = [conf for conf, correct in zip(correct_confidence, correctness) if correct == 1]
    # incorrect_confidence = [conf for conf, correct in zip(correct_confidence, correctness) if correct == 0]

    avg_correct_confidence = sum(correct_confidence) / len(correct_confidence) if correct_confidence else 0
    # avg_incorrect_confidence = sum(incorrect_confidence) / len(incorrect_confidence) if incorrect_confidence else 0
    
    return avg_correct_confidence


def evaluate_results(predictions_file: str) -> Dict[str, float]:
    """Evaluate predictions against ground truth"""

    # predictions = load_predictions(predictions_file) # bug1

    predictions = pd.read_csv(predictions_file) # fix bug1
    
    # Align predictions with ground truth
    if 'cop' in predictions.columns:
        true_labels = predictions['cop'].tolist()
    else:
        true_labels = predictions['data'].apply(lambda x: eval(x)['Correct Option']).tolist()
    pred_labels = predictions['correct_answer'].tolist()
    print("处理前：",len(pred_labels), len(true_labels))
    for pred_label, true_label in zip(pred_labels, true_labels):
        if pred_label == -1:
            pred_labels.remove(pred_label)
            true_labels.remove(true_label)
    print("处理后：",len(pred_labels), len(true_labels))
    correct_confidence = predictions['correct_prob'].tolist() ##################

    # Calculate metrics
    metrics = {'accuracy': 0.0,
               'correct_confidence': 0.0, #####################
            #    'incorrect_confidence': 0.0,
               }
    metrics['accuracy'] = evaluate_accuracy(true_labels, pred_labels)
    metrics['correct_confidence']= evaluate_confidence(true_labels, pred_labels, correct_confidence) #################### 

    return metrics


def compute_subject_wise_accuracy(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute accuracy for each subject in the dataset.

    Args:
        results_df: DataFrame containing the evaluation results with columns:
                    ['subject_name', 'cop', 'correct_answer']

    Returns:
        A dictionary where keys are subject names and values are accuracy scores.
    """
    # Initialize a dictionary to store subject-wise accuracy
    subject_wise_accuracy = {}

    # Group the dataset by subject_name
    grouped = results_df.groupby('subject_name')

    # Iterate through each group
    for subject, group in grouped:
        # Extract true labels (cop) and predicted labels (correct_answer)
        true_labels = group['cop'].tolist()
        pred_labels = group['correct_answer'].tolist()

        # Compute accuracy for the current subject
        correctness = [1 if true == pred else 0 for true, pred in zip(true_labels, pred_labels)]
        accuracy = sum(correctness) / len(correctness) if len(correctness) > 0 else 0.0

        # Store the accuracy in the dictionary
        subject_wise_accuracy[subject] = accuracy

    return subject_wise_accuracy



def visualize_subject_wise_accuracy(subject_wise_accuracy: Dict[str, float], output_path: str = "subject_wise_accuracy.png"):
    """
    Visualize subject-wise accuracy as a bar chart.

    Args:
        subject_wise_accuracy: A dictionary where keys are subject names and values are accuracy scores.
        output_path: Path to save the bar chart image.
    """
    # Convert the dictionary to a sorted list of tuples for visualization
    sorted_accuracy = sorted(subject_wise_accuracy.items(), key=lambda x: x[1], reverse=True)
    subjects, accuracies = zip(*sorted_accuracy)

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(subjects), y=list(accuracies), palette="viridis")

    # Add labels and title
    plt.xlabel("Subjects", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Subject-Wise Accuracy", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1

    # Annotate bars with accuracy values
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=10)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Subject-wise accuracy chart saved to {output_path}")

# def evaluate_qa_results(results_df: pd.DataFrame) -> Dict[str, float]:
#     """
#     Evaluate MultiAgentQA results
    
#     Args:
#         results_df: DataFrame with columns ['id', 'predicted_answer', 'correct_answer', 'question']
#     """
#     true_labels = results_df['correct_answer'].tolist()
#     pred_labels = results_df['predicted_answer'].apply(lambda x: x if isinstance(x, str) else '').tolist()
    
#     metrics = {
#         'accuracy': sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels),
#         'precision': precision_score(true_labels, pred_labels, average='macro', zero_division=0),
#         'recall': recall_score(true_labels, pred_labels, average='macro', zero_division=0),
#         'f1_score': f1_score(true_labels, pred_labels, average='macro', zero_division=0)
#     }
    
#     # Generate confusion matrix
#     labels = ['A', 'B', 'C', 'D']
#     cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
#     # Plot confusion matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.savefig('confusion_matrix.png')
#     plt.close()
    
#     # Calculate per-option metrics
#     per_option_metrics = {}
#     for idx, option in enumerate(labels):
#         option_mask = [t == option for t in true_labels]
#         if sum(option_mask) > 0:
#             per_option_metrics[option] = {
#                 'precision': precision_score(true_labels, pred_labels, average=None, zero_division=0)[idx],
#                 'recall': recall_score(true_labels, pred_labels, average=None, zero_division=0)[idx],
#                 'count': sum(option_mask)
#             }
    
#     metrics['per_option'] = per_option_metrics
#     return metrics

def generate_error_analysis(results_df: pd.DataFrame) -> Dict:
    """
    Analyze error patterns in predictions
    """
    if 'cop' in results_df.columns:
        error_cases = results_df[results_df['correct_answer'] != results_df['cop']]
    else: 
        error_cases = results_df[results_df['correct_answer'] != results_df['data'].apply(lambda x: eval(x)['Correct Option']).tolist()]

    # Convert tuple keys to strings for JSON compatibility
    # common_error_patterns = {
    #     f"{k[0]} -> {k[1]}": v
    #     for k, v in error_cases.groupby(['cop', 'correct_answer']).size().to_dict().items()
    # }

    analysis = {
        'total_errors': len(error_cases),
        'error_rate': len(error_cases) / len(results_df),
        # 'common_error_patterns': error_cases.groupby(['cop', 'correct_answer']).size().to_dict(),
        # 'common_error_patterns': common_error_patterns,
        'sample_errors': error_cases.head(5).to_dict('records')
    }
    
    return analysis

def save_evaluation_report(metrics: Dict, error_analysis: Dict, output_path: str):
    """
    Save detailed evaluation report
    """
    report = {
        'metrics': metrics,
        'error_analysis': error_analysis,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    # Configuration
    dataset_path = "path/to/medmcqa/dataset.json"
    predictions_file = "path/to/crewai_predictions.json"

    # Run evaluation
    metrics = evaluate_results(predictions_file)

    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"F1-score: {metrics['f1_score']:.4f}")

