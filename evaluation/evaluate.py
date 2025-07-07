import pandas as pd
# from datasets.medmcqa.dataset import MCQADataset
# from datasets.medmcqa.model import MCQAModel
import os
from tqdm import tqdm
from datasets import load_dataset
from medicalMCQA import MedicalMCQACrew
import json


def evaluate_medmcqa(
    args,
    crew,
    baseline
):
    dataset = load_dataset("openlifescienceai/medmcqa")[args.split]

    # Sample a subset of the dataset #change 0424
    sample_size = args.sample_size  # Define the number of instances to sample
    dataset = dataset.shuffle(seed=8).select(range(sample_size))  # Shuffle and select the first `sample_size` examples
    # dataset = dataset.select(range(sample_size))

    all_answers = []
    correct_answers = []
    correct_probs = []
    correct_rationales = []
    composite_confidences = []
    evidence_coverages = []
    clinical_alignments = []

    # Define a mapping from letters to numbers
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    batch_index = 0
    question_index = 0
    for start_idx in range(0, len(dataset), args.batch_size):
        
        batch = dataset.select(range(start_idx, min(start_idx + args.batch_size, len(dataset))))
        questions = []
        answers = []
        for item in batch:
            question = f"""Question: {item['question']}\nOptions:\nA) {item['opa']}\nB) {item['opb']}\nC) {item['opc']}\nD) {item['opd']}\nSubject: {item['subject_name']}\nTopic: {item['topic_name']}"""                             # format of input question
            questions.append(question)

        print(f"Questions passed to crew: {questions}")

        # crew_output_path = os.path.join(args.output_dir, f'crew_medmcqa_outputs_{args.split}_{batch_index}_{args.sample_size}.json')
        # crew = MedicalMCQACrew(question=questions, output_file=crew_output_path)
        # crew.question = questions
        baseline.question = questions
        try:
            if args.model == 'crew':
                # answers = crew.run()  # test1
                for question in questions:
                    crew_output_path = os.path.join(args.output_dir, f'crew_medmcqa_outputs_{args.split}_{question_index}_{args.sample_size}.json')
                    crew = MedicalMCQACrew(question=question, output_file=crew_output_path)
                    crew.run()
                    with open(crew_output_path, 'r', encoding='utf-8') as f:
                        answer = json.load(f)
                        answers.append(answer)
                    question_index += 1
                # output = crew.run() ################
                # answers = json.load(crew_output_path)
            elif args.model == 'baseline':
                answers = baseline.batch_process(questions, 
                                                output_file="baseline_medmcqa_results.json", #########
                                                delay=1.0)

            # # Generate pseudo answers for testing
            # answers = []
            # for question in questions:
            #     pseudo_answer = {
            #         "options": [
            #             {
            #                 "code": "A",
            #                 "probability": 0.60,
            #                 "rationale": "Myocardial Infarction (MI) is supported by robust diagnostic measures (ECG changes, elevated troponin levels) and guidelines from reputable organizations (AHA, ESC), indicating high urgency for intervention."
            #             },
            #             {
            #                 "code": "B",
            #                 "probability": 0.30,
            #                 "rationale": "Congestive Heart Failure (CHF) presents as moderate risk, supported by patient history and echocardiogram findings along with essential biomarker assessment, indicating a need for careful management but lower urgency than MI."
            #             },
            #             {
            #                 "code": "C",
            #                 "probability": 0.10,
            #                 "rationale": "Pulmonary Embolism (PE) is considered low risk and necessitates further imaging studies for a definitive diagnosis, highlighting uncertainty in treatment approaches."
            #             }
            #         ],
            #         "correct_answer": "A",
            #         "analysis_metrics": {
            #             "composite_confidence": 0.95,
            #             "evidence_coverage": 0.95,
            #             "clinical_alignment": 0.90
            #         }
            #     }
            #     answers.append(pseudo_answer)

            for answer in answers:
                # Extract fields from the JSON-like output
                correct_answer = answer.get("correct_answer", "")
                correct_answers.append(answer_mapping.get(correct_answer, -1))  # Convert letter to number
                ################
                options = answer.get("options", [])
                correct_prob = options[answer_mapping.get(correct_answer, -1)].get("probability", 0.0)
                if isinstance(correct_prob, str) and correct_prob.endswith('%'):
                    correct_prob = float(correct_prob.strip('%')) / 100.0
                else:
                    correct_prob = float(correct_prob) / 100.0
                correct_probs.append(correct_prob)
                ################
                correct_rationale = options[answer_mapping.get(correct_answer, -1)].get("rationale", "")
                correct_rationales.append(correct_rationale)

                ################3
                # if args.model == 'crew': 
                    # metrics = answer.get("analysis_metrics", {}) 
                    # composite_confidences.append(metrics.get("composite_confidence", 0.0)) 
                    # evidence_coverages.append(metrics.get("evidence_coverage", 0.0)) 
                    # clinical_alignments.append(metrics.get("clinical_alignment", 0.0)) 
                # elif args.model == 'baseline':
                #     composite_confidences.append(-1)
                #     evidence_coverages.append(-1)
                #     clinical_alignments.append(-1)
                ###############
            all_answers.extend(answers)
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return

        batch_index += 1

    # append all_answers as a 'predictions' column to the dataset
    results_df = dataset.to_pandas() # fix bug2
    

    # results_df['predictions'] = all_answers # fix bug2

    # Append the extracted fields as new columns
    results_df['correct_answer'] = correct_answers
    results_df['correct_prob'] = correct_probs ##################3
    results_df['correct_rationale'] = correct_rationales
    # results_df['composite_confidence'] = composite_confidences ##################
    # results_df['evidence_coverage'] = evidence_coverages #################
    # results_df['clinical_alignment'] = clinical_alignments ##################3

    # results = dataset.add_column("predictions", all_answers) # bug2

    # save results to a CSV file
    results_path = os.path.join(args.output_dir, f'medmcqa_results_{args.split}_{args.model}_{args.batch_size}_{args.sample_size}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

def evaluate_medqa(
    args,
    crew,
    baseline
):
    dataset = load_dataset("openlifescienceai/medqa", split=args.split)
    sample_size = args.sample_size
    dataset = dataset.shuffle(seed=42).select(range(sample_size))

    all_answers = []
    correct_answers = []
    correct_probs = []
    correct_rationales = []
    composite_confidences = []
    evidence_coverages = []
    clinical_alignments = []

    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    for start_idx in range(0, len(dataset), args.batch_size):
        batch = dataset.select(range(start_idx, min(start_idx + args.batch_size, len(dataset))))
        questions = []
        for item in batch:
            data = item['data']
            question = f"""Question: {data['Question']}
Options:
A) {data['Options']['A']}
B) {data['Options']['B']}
C) {data['Options']['C']}
D) {data['Options']['D']}"""
            questions.append(question)

        crew = MedicalMCQACrew(question=questions)
        # crew.question = questions
        baseline.question = questions
        try:
            if args.model == 'crew':
                answers = crew.run()  # test1
            elif args.model == 'baseline':
                answers = baseline.batch_process(questions, 
                                                output_file="baseline_medqa_results.json", #########
                                                delay=1.0)
            for answer in answers:
                # Extract fields from the JSON-like output
                correct_answer = answer.get("correct_answer", "")
                correct_answers.append(correct_answer)  # Convert letter to number
                options = answer.get("options", [])
                correct_prob = options[answer_mapping.get(correct_answer, -1)].get("probability", 0.0)
                if isinstance(correct_prob, str) and correct_prob.endswith('%'):
                    correct_prob = float(correct_prob.strip('%')) / 100.0
                else:
                    correct_prob = float(correct_prob) / 100.0
                correct_probs.append(correct_prob)
                correct_rationale = options[answer_mapping.get(correct_answer, -1)].get("rationale", "")
                correct_rationales.append(correct_rationale)

                if args.model == 'crew': 
                    metrics = answer.get("analysis_metrics", {})
                    composite_confidences.append(metrics.get("composite_confidence", 0.0))
                    evidence_coverages.append(metrics.get("evidence_coverage", 0.0))
                    clinical_alignments.append(metrics.get("clinical_alignment", 0.0))
                elif args.model == 'baseline':
                    composite_confidences.append(-1)
                    evidence_coverages.append(-1)
                    clinical_alignments.append(-1)
            all_answers.extend(answers)

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return

    results_df = dataset.to_pandas()
    results_df['correct_answer'] = correct_answers
    results_df['correct_prob'] = correct_probs
    results_df['correct_rationale'] = correct_rationales
    results_df['composite_confidence'] = composite_confidences
    results_df['evidence_coverage'] = evidence_coverages
    results_df['clinical_alignment'] = clinical_alignments

    results_path = os.path.join(args.output_dir, f'medqa_results_{args.split}_{args.model}_{args.batch_size}_{args.sample_size}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

# def evaluate_medqa(
#     args,
#     crew,
#     baseline
# ):
#     dataset = load_dataset("openlifescienceai/medqa", split=args.split)

#     sample_size = args.sample_size
#     dataset = dataset.shuffle(seed=42).select(range(sample_size))

#     all_answers = []
#     correct_answers = []
#     correct_probs = []
#     correct_rationales = []
#     composite_confidences = []
#     evidence_coverages = []
#     clinical_alignments = []

#     answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

#     for start_idx in range(0, len(dataset), args.batch_size):
#         batch = dataset.select(range(start_idx, min(start_idx + args.batch_size, len(dataset))))
#         questions = []
#         for item in batch:
#             question = f"""
#             Question: {item['question']}
#             Options:
#             A) {item['opa']}
#             B) {item['opb']}
#             C) {item['opc']}
#             D) {item['opd']}
#             Subject: {item.get('subject_name', 'N/A')}
#             Topic: {item.get('topic_name', 'N/A')}
#             """
#             questions.append(question)

#         crew.question = questions
#         baseline.question = questions
#         try:
#             if args.model == 'crew':
#                 answers = crew.run()
#             elif args.model == 'baseline':
#                 answers = baseline.batch_process(questions, output_file="baseline_medqa_results.json", delay=1.0)

#             for answer in answers:
#                 correct_answer = answer.get("correct_answer", "")
#                 correct_answers.append(answer_mapping.get(correct_answer, -1))
#                 options = answer.get("options", [])
#                 correct_index = answer_mapping.get(correct_answer, -1)
#                 correct_prob = options[correct_index].get("probability", 0.0) if 0 <= correct_index < len(options) else 0.0
#                 if isinstance(correct_prob, str) and correct_prob.endswith('%'):
#                     correct_prob = float(correct_prob.strip('%')) / 100.0
#                 correct_probs.append(correct_prob)
#                 correct_rationale = options[correct_index].get("rationale", "") if 0 <= correct_index < len(options) else ""
#                 correct_rationales.append(correct_rationale)

#                 if args.model == 'crew':
#                     metrics = answer.get("analysis_metrics", {})
#                     composite_confidences.append(metrics.get("composite_confidence", 0.0))
#                     evidence_coverages.append(metrics.get("evidence_coverage", 0.0))
#                     clinical_alignments.append(metrics.get("clinical_alignment", 0.0))
#                 else:
#                     composite_confidences.append(-1)
#                     evidence_coverages.append(-1)
#                     clinical_alignments.append(-1)

#             all_answers.extend(answers)

#         except Exception as e:
#             print(f"Error processing batch: {str(e)}")
#             return

#     results_df = dataset.to_pandas()
#     results_df['correct_answer'] = correct_answers
#     results_df['correct_prob'] = correct_probs
#     results_df['correct_rationale'] = correct_rationales
#     results_df['composite_confidence'] = composite_confidences
#     results_df['evidence_coverage'] = evidence_coverages
#     results_df['clinical_alignment'] = clinical_alignments

#     results_path = os.path.join(args.output_dir, f'medqa_results_{args.split}_{args.model}_{args.sample_size}.csv')
#     results_df.to_csv(results_path, index=False)
#     print(f"Results saved to: {results_path}")

# def evaluate_medqa(args, 
#                    crew,
#                    baseline):
    # dataset = load_dataset("medqa")[args.split]  # Load the MedQA dataset

    # sample_size = args.sample_size
    # dataset = dataset.shuffle(seed=42).select(range(sample_size))  # Shuffle and select a subset

    # all_answers = []
    # correct_answers = []
    # correct_probs = []
    # correct_rationales = []

    # for start_idx in range(0, len(dataset), args.batch_size):
    #     batch = dataset.select(range(start_idx, min(start_idx + args.batch_size, len(dataset))))
    #     questions = []
    #     for item in batch:
    #         question = f"""
    #         Question: {item['question']}
    #         Options:
    #         A) {item['option_a']}
    #         B) {item['option_b']}
    #         C) {item['option_c']}
    #         D) {item['option_d']}
    #         Subject: {item['subject']}
    #         Topic: {item['topic']}
    #         """
    #         questions.append(question)
        
    #     try:
    #         if args.model == 'crew':
    #             answers = crew.run()  # test1
    #         elif args.model == 'baseline':
    #             answers = baseline.batch_process(questions, 
    #                                             output_file="baseline_medical_qa_results.json", #########
    #                                             delay=1.0)

    #         for answer in answers:
    #             correct_answer = answer.get("correct_answer", "")
    #             correct_answers.append(correct_answer)
    #             options = answer.get("options", [])
    #             correct_prob = options[answer.get("correct_index", -1)].get("probability", 0.0)
    #             if isinstance(correct_prob, str) and correct_prob.endswith('%'):
    #                 correct_prob = float(correct_prob.strip('%')) / 100.0
    #             else:
    #                 correct_prob = float(correct_prob)
    #             correct_probs.append(correct_prob)
    #             correct_rationale = options[answer.get("correct_index", -1)].get("rationale", "")
    #             correct_rationales.append(correct_rationale)

    #         all_answers.extend(answers)
    #     except Exception as e:
    #         print(f"Error processing batch: {str(e)}")
    #         return

    # results_df = dataset.to_pandas()
    # results_df['correct_answer'] = correct_answers
    # results_df['correct_prob'] = correct_probs
    # results_df['correct_rationale'] = correct_rationales

    # results_path = os.path.join(args.output_dir, f'medqa_results_{args.split}_{args.model}_{args.sample_size}.csv')
    # results_df.to_csv(results_path, index=False)
    # print(f"Results saved to: {results_path}")


