from crewai import Agent, Task, Crew
import os
import json
import sys
import io
from tools.RagTool import rag_tool
from pydantic import BaseModel
from datasets import load_dataset
import datetime
# 编码UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# 初始化GPT-4o-mini配置
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-iY1j43dZqk8b9ReDULM6zzYN3GJ85eHLfFUi6srBUaEZDRJm" # 你的openai密匙



class MedicalMCQACrew:
    # 初始化
    def __init__(self, question, output_file="results.json"):
        # 初始化问题
        self.question = question

        self.output_file = output_file

        # 初始化Agents
        self.question_analyst = self._create_question_analyst()
        self.retrieval_specialist = self._create_retrieval_specialist()
        self.synthesis_engine = self._create_synthesis_engine()
        self.verification_expert = self._create_verification_expert()
        self.orchestrator = self._create_orchestrator()

        # 初始化Tasks
        self.tasks = self._build_tasks()
    

    # 定义分析问题Agent
    def _create_question_analyst(self):
        return Agent(
            role='Senior Medical NLP Analyst',
            goal='Analyze medical questions with probabilistic modeling',
            backstory='Expert in clinical decision analysis and Bayesian reasoning',
            verbose=True,
            allow_delegation=False
        )

    # 定义搜数据库Agent
    def _create_retrieval_specialist(self):
        return Agent(
            role='Evidence Retrieval Specialist',
            goal='Retrieve the most relevant medical evidence',
            backstory='Equipped with access to PubMed Summarizations and various Wikipedia medical pages', #这改成相关数据库
            tools=[],
            verbose=True,
            allow_delegation=False
            )

    # 定义下结论Agent
    def _create_synthesis_engine(self):
        return Agent(
            role='Medical Synthesis Expert',
            goal='Generate accurate and coherent medical answers',
            backstory='Experienced in clinical reasoning and medical communication',
            verbose=True,
            allow_delegation=False
        )

    # 定义验证Agent
    def _create_verification_expert(self):
        return Agent(
            role='Clinical Verification Specialist',
            goal='Ensure medical accuracy and compliance',
            backstory='Board-certified medical auditor with quality assurance expertise',
            verbose=True,
            allow_delegation=False
        )

    # 定义协作Agent
    def _create_orchestrator(self):
        return Agent(
            role='Chief Clinical Coordinator',
            goal='Orchestrate multi-agent collaboration',
            backstory='Experienced in clinical workflow optimization',
            verbose=True,
            allow_delegation=False
        )


    # 定义Tasks
    def _build_tasks(self):

        class Outcome(BaseModel):
            options: list[dict]
            correct_answer: str
            analysis_metrics: dict

        # 分析问题Task
        analysis_task = Task(
            description=f"""
                Analyze the medical question: {self.question}
                Then perform the following:
                    1. Identify key clinical entities (symptoms, signs, findings...)
                    2. Classify question type (diagnostic/therapeutic/preventive...)
                Then conduct probability distribution analysis on each question option:
                    1. Use Bayesian inference to assess prior probabilities
                    2. Generate probability distribution (0-100) 
                    3. Specify the confidence score (0-1)
                """,
            agent=self.question_analyst,
            expected_output="Analysis Results with confidence scores"
        )

        # 搜数据库Task
        retrieval_task = Task(
            description="""
                Conduct evidence retrieval based on probabilities on each question option:
                    1. Prioritize evidence for high-probability options
                    2. Validate exclusion criteria for low-probability options
                    3. Ensure coverage of critical evidence for all options
                    4. Generate evidence weighting scores (0-1)
                """,
            agent=self.retrieval_specialist,
            context=[analysis_task],
            expected_output="Retrieval results with evidence weighting scores"
        )

        # 下结论Task
        synthesis_task = Task(
            description="""
                Generate probability-enhanced answer:
                    1. Integrate evidence and probability distribution
                    2. Formulate clinical reasoning chain
                    3. Output answer (which question option(s)?) with statistical metrics
                    4. Flag high-risk options
                """,
            agent=self.synthesis_engine,
            context=[retrieval_task],
            expected_output="Synthesis results with probabilities"
        )

        # 验证Task
        verification_task = Task(
            description="""
                Validate probability analysis:
                    1. Verify probability-evidence consistency
                    2. Validate statistical methods
                    3. Calibrate probability values (0-100)
                    4. Generate final confidence scores (0-1)
                    5. Output final answer (Which question option(s)?) with statistical metrics
                """,
            agent=self.verification_expert,
            context=[synthesis_task],
            expected_output="Final answer with calibrated probabilities"
        )

        # 协作Task
        orchestration_task = Task(
            description="""
                For each question, Integrate and format final report:
                    1. Analyze the question requirements:
                        a) If the question asks for the "correct" or "most appropriate" option:
                           - Assign higher probability to the most correct/appropriate option
                           - The option with highest probability will be the correct answer
                        b) If the question asks for the "incorrect" or "least appropriate" option:
                           - Assign higher probability to the most correct/appropriate option
                           - The option with lowest probability will be the correct answer
                        c) If the question asks for "except" or "not":
                           - Assign higher probability to the options that match the criteria
                           - The option with lowest probability will be the correct answer
                        d) If the question asks for "all of the following are true EXCEPT":
                           - Assign higher probability to the true statements
                           - The option with lowest probability will be the correct answer
                    2. Generate probability distribution:
                        a) For questions asking for correct answers:
                           - Highest probability for the most correct option
                           - This option will be marked as correct_answer
                        b) For questions asking for incorrect answers:
                           - Highest probability for the most correct option
                           - The option with lowest probability will be marked as correct_answer
                    3. Output standardized JSON structure specifying
                        a) probability of each question option (A, B, C...)
                        b) final answer (A/B/C...) - this will be:
                           - The option with highest probability for correct answer questions
                           - The option with lowest probability for incorrect answer questions
                        c) analysis metrics
                        d) in the format like:
                            {
                                "options": [
                                    {"code": "A", "probability": 0.00, "rationale": "..."},
                                    {"code": "B", "probability": 0.00, "rationale": "..."},
                                    ...
                                ],
                                "correct_answer": "A",
                                "analysis_metrics": {
                                    "composite_confidence": 0.00,
                                    "evidence_coverage": 0.00,
                                    "clinical_alignment": 0.00
                                }
                            }
                        e) Ensure "code" is the question option (A, B, C...)
                        f) Ensure "probability" is a .00 percentage (0-100)
                        g) Ensure "correct_answer" is:
                           - The option with highest probability for correct answer questions
                           - The option with lowest probability for incorrect answer questions
                        h) Ensure "composite_confidence", "evidence_coverage", and "clinical_alignment" are .00 between 0-1
                """,
            agent=self.orchestrator,
            context=[verification_task],
            expected_output="A list of Standardized JSON result",
            output_json=Outcome,
            output_file=self.output_file,
        )

        return [analysis_task, retrieval_task, synthesis_task, verification_task, orchestration_task]

    # 运行
    def run(self):
        crew = Crew(
            agents=[self.question_analyst,
                    self.retrieval_specialist,
                    self.synthesis_engine,
                    self.verification_expert,
                    self.orchestrator
            ],
            tasks=self.tasks,
            verbose=True
        )


        # Prepare inputs as a dictionary
        #inputs = [{"question": q} for q in self.question]  # Convert list of questions to a list of dictionaries


        # crew_output = crew.kickoff()
        #print(f"Inputs passed to crew.kickoff_for_each: {inputs}")
        crew_output = crew.kickoff()
        print(f"结果存入成功，结果文件为：{self.output_file}")
        # print(f"Orchestrator raw output: {crew_output}")
        return crew_output
        #return self._get_raw_json(crew_output)
        result = self._get_raw_json(crew_output)
        result = json.dumps(result, indent=2, ensure_ascii=False)

        return result

    # 直接提取并返回标准JSON
    def _get_raw_json(self, crew_output):
        try:
            processed_output = []

            for output in crew_output:
            
                if hasattr(output, 'json_dict') and output.json_dict:
                    processed_output.append(output.json_dict)
                else:
                    raw_data = output.raw if hasattr(output, 'raw') else output
                    if "```json" in raw_data:
                        json_str = raw_data.split("```json")[1].split("```")[0].strip()
                        processed_output.append(json.loads(json_str))
                    else:
                        processed_output.append(json.loads(raw_data))
            return processed_output
        except Exception as e:
            return {
                "error": str(e),
                "raw_output": crew_output,
            }


# 示例
# if __name__ == "__main__":
    # sample_question = ["""
    # A 45-year-old man presents with epigastric pain relieved by eating.
    # Endoscopy shows duodenal ulcers. Which combination is MOST appropriate?
    # A) PPIs + H. pylori eradication therapy
    # B) H2 blockers + antacids
    # C) Sucralfate + antibiotics
    # D) NSAIDs + misoprostol
    # """]
    # dataset = load_dataset("openlifescienceai/medmcqa")["validation"]

    # # 获取数据集的前5个元素
    # dataset = dataset.select(range(5))
    # print(f"数据集大小: {len(dataset)}")
    
    # for i in range(5):  # 处理前2个问题
    #     item = dataset[i]
    #     question = f"""Question: {item['question']}\nOptions:\nA) {item['opa']}\nB) {item['opb']}\nC) {item['opc']}\nD) {item['opd']}\nSubject: {item['subject_name']}\nTopic: {item['topic_name']}"""
    #     print(item['cop'])
    #     print(question)
    #     # 为每个问题创建唯一的输出文件名
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_file = f"result_{i}.json"
        


    #     print(f"正在处理第 {i+1} 个问题...")
    #     mcqa_crew = MedicalMCQACrew([question], output_file)
    #     result = mcqa_crew.run()
    #     print(f"问题 {i+1} 处理完成，结果已保存到 {output_file}")
    
    
    
    
    
    '''sample_question = ['\n Question: Amount of heat that is required to change boiling water into vapor is referred to as\n Options:\n A) Latent Heat of vaporization\n B) Latent Heat of sublimation\n C) Latent Heat of condensation\n D) Latent heat of fusion\n Subject: Dental\n Topic: None\n ']
    mcqa_crew = MedicalMCQACrew(sample_question)
    result = mcqa_crew.run()
    print(result)'''
    #print(json.dumps(result, indent=2, ensure_ascii=False))