import os
import json
import sys
import io
from openai import OpenAI
import time
from tqdm import tqdm 
from typing import List, Dict
import pandas as pd

# 编码UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# 初始化GPT-4o-mini配置
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-iY1j43dZqk8b9ReDULM6zzYN3GJ85eHLfFUi6srBUaEZDRJm" # 你的openai密匙

class BaselineModel:
    def __init__(self, question):
        self.question = question
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"]
        )
        self.prompt = """
            You are a medical expert. Analyze the question and provide ONLY the final JSON output in English. 
            Do not include any analysis process or explanations outside the JSON structure.
            Question: {question}
            Required JSON format:
            {{
                "options": [
                    {{"code": "A", "probability": "XX.XX", "rationale": "Brief explanation in English"}},
                    {{"code": "B", "probability": "XX.XX", "rationale": "Brief explanation in English"}},
                    {{"code": "C", "probability": "XX.XX", "rationale": "Brief explanation in English"}},
                    {{"code": "D", "probability": "XX.XX", "rationale": "Brief explanation in English"}}
                ],
                "correct_answer": "X"
            }}
            Rules:
            1. Only output the JSON structure, nothing else
            2. All text must be in English
            3. Probabilities must be between 0.00 and 100.00
            4. Rationale should be concise and professional
            5. Correct_answer should be the option with highest probability
            Example question: "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma", "A)":"Hyperplasia","B)":"Hyperophy","C)":"Atrophy","D)":"Dyplasia"
            Example answer: {{"options": [{{"code": "A", "probability": 30.00, "rationale": "..."}},{{"code": "B", "probability": 20.00, "rationale": "..."}},{{"code": "C", "probability": 90.00, "rationale": "..."}},{{"code": "D", "probability": 40.00, "rationale": "..."}}],"correct_answer": "C"}}
            """
    
    def run(self):
        # 构建完整的 prompt
        full_prompt = self.prompt.format(question=self.question)
        
        # 调用 API
        response = self.client.chat.completions.create(
            model=os.environ["OPENAI_MODEL_NAME"], 
            messages=[
                {"role": "system", "content": "You are a medical expert specialized in clinical diagnosis."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            presence_penalty=0.0,  # 不鼓励模型引入新主题
            frequency_penalty=0.0,  # 不惩罚频繁使用的词
            top_p=0.9  # 控制输出的确定性
        )
        
        result_text = response.choices[0].message.content
        cleaned_result = result_text.replace('\n', '').replace('\\n', '').strip()
        return cleaned_result


    def batch_process(self, questions: List[str], output_file: str = "results.json", delay: float = 1.0):
        """批量处理多个问题
        Args:
            questions: 问题列表
            output_file: 输出文件名
            delay: 每次请求之间的延迟(秒)
        """
        results = []
        errors = []

        for i, question in enumerate(tqdm(questions, desc="Processing questions")):
            try:
                result_str = self.run()
                result_dict = json.loads(result_str)
                print(f"Question {i + 1}: \n Result:\n {json.dumps(result_dict, indent=2, ensure_ascii=False)}")
                results.append(result_dict)
                
                # 定期保存结果
                if (i + 1) % 10 == 0:
                    self._save_results(results, output_file)
                    
                time.sleep(delay)  # API 请求间隔
                
            except Exception as e:
                errors.append({
                    "question": question,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"\nError processing question: {str(e)}")
                result_str = f'{{"options": [{{"code": "A", "probability": 0.00, "rationale": "Error occurred"}}], "correct_answer": "X"}}'
                result_dict = json.loads(result_str)
                results.append(result_dict)
        # 最终保存所有结果
        self._save_results(results, output_file)
        
        # 如果有错误，保存错误日志
        if errors:
            error_file = f"errors_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
        return results    

    def _save_results(self, results: List[Dict], filename: str):
        """保存结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
      
    
if __name__ == "__main__":
    # # 从CSV文件读取问题
    # def load_questions_from_csv(filepath: str) -> List[str]:
    #     df = pd.read_csv(filepath)
    #     return df['question'].tolist()  # 假设CSV文件中有一列名为'question'
    
    # # 从JSON文件读取问题
    # def load_questions_from_json(filepath: str) -> List[str]:
    #     with open(filepath, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     return data['questions']  # 假设JSON文件中有一个'questions'数组
    
    # 示例使用
    questions = [
        """
        A 45-year-old man presents with epigastric pain relieved by eating.
        Endoscopy shows duodenal ulcers. Which combination is MOST appropriate?
        A) PPIs + H. pylori eradication therapy
        B) H2 blockers + antacids
        C) Sucralfate + antibiotics
        D) NSAIDs + misoprostol
        """,
        """
        Which vitamin is supplied from only animal source:
        A) Vitamin C
        B) Vitamin B7
        C) Vitamin B12
        D) Vitamin D
        """
        # 添加更多问题...
    ]
    
    # 批量处理
    model = BaselineModel("")  # 初始化时传入空字符串
    results = model.batch_process(
        questions=questions,
        output_file="baseline_medical_qa_results.json",
        delay=1.0  # 每次请求间隔1秒
    )
    
    # 打印统计信息
    print(f"\nProcessed {len(results)} questions")
    print(f"Results saved to baseline_medical_qa_results.json")
   