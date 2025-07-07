# from multiagent_qa import MultiAgentMedicalQA
import sys
import os
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from medicalMCQA import MedicalMCQACrew
from evaluation.eval_helper import evaluate_results, generate_error_analysis, save_evaluation_report, compute_subject_wise_accuracy, visualize_subject_wise_accuracy
import pandas as pd
import argparse
from datetime import datetime
from evaluation.evaluate import evaluate_medmcqa, evaluate_medqa
from datasets import load_dataset
from typing import Dict
from baseline import BaselineModel

os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # Replace with your actual OpenAI API key


def parse_args():
    parser = argparse.ArgumentParser(description='Run Medical QA System')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'],
                      help='Dataset split to use')
    # parser.add_argument('--knowledge-base', type=str, required=True,
    #                   help='Path to knowledge base file')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Number of questions to process in one batch')
    parser.add_argument('--dataset', type=str, default='medmcqa', choices=['medmcqa', 'medqa'],)
    parser.add_argument('--sample-size', type=int, default=300,
                      help='Number of samples to evaluate'),
    parser.add_argument('--model', type=str, default='crew', choices=['crew', 'baseline'],)
    
    

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = os.path.join(args.output_dir, args.dataset, args.model)
    os.makedirs(output_dir, exist_ok=True)

    eval_report_path = os.path.join(output_dir, f'eval_report_{args.split}_{args.batch_size}_{args.sample_size}_{timestamp_start}.json')

    # Instantiate the QA crew
    QAcrew = MedicalMCQACrew(question=None)
    # Instantiate baseline model
    baseline_model = BaselineModel("")
    
    timestamp_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.dataset == 'medmcqa':
        # Process MedMCQA dataset
        evaluate_medmcqa(args=args, crew=QAcrew, baseline=baseline_model)
    elif args.dataset == 'medqa':
        # Process MedQA dataset
        evaluate_medqa(args=args, crew=QAcrew, baseline=baseline_model)

    # timestamp_end = datetime.now().strftime('%Y%m%d_%H%M%S')
    time_spent = datetime.now() - datetime.strptime(timestamp_start, '%Y%m%d_%H%M%S')
    print(f"Time spent on evaluation: {time_spent}")

    # Load test_results
    if args.dataset == 'medmcqa':
        test_results = os.path.join(args.output_dir, f'medmcqa_results_{args.split}_{args.model}_{args.batch_size}_{args.sample_size}.csv')
    elif args.dataset == 'medqa':
        test_results = os.path.join(args.output_dir, f'medqa_results_{args.split}_{args.model}_{args.batch_size}_{args.sample_size}.csv')

    # Evaluate results
    results_df = pd.read_csv(test_results)
    metrics = evaluate_results(test_results)
    error_analysis = generate_error_analysis(results_df)
    # Compute subject-wise accuracy
    if args.dataset == 'medmcqa':
        subject_wise_accuracy = compute_subject_wise_accuracy(results_df)
        visualize_subject_wise_accuracy(subject_wise_accuracy, os.path.join(output_dir, f'subject_wise_accuracy_{timestamp_start}.png'))

    metrics['time_spent'] = str(time_spent)

    if args.dataset == 'medmcqa':
        metrics['subject_wise_accuracy'] = subject_wise_accuracy

    metrics['total_questions'] = len(results_df) # fix bug3

    # Save evaluation report
    save_evaluation_report(metrics, error_analysis, eval_report_path)
    print(f"Evaluation report saved to: {eval_report_path}")

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Dataset: {args.dataset}")
    print(f"Total Questions: {metrics['total_questions']}") # bug3
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct Confidence: {metrics['correct_confidence']:.2%}") #################3
    # print(f"Incorrect Confidence: {metrics['incorrect_confidence']:.2%}")
    # print(f"Macro F1-score: {metrics['f1_score']:.2%}")
    # print(f"Error Rate: {error_analysis['error_rate']:.2%}")

if __name__ == "__main__":
    main()



