import json
import requests
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.datasets import BaseDataset
from opencompass.utils import TextType
from typing import List, Optional
import numpy as np
from tqdm import tqdm

class MultipleChoiceDataset(BaseDataset):
    """自定义单项选择题数据集类"""
    
    def __init__(self, path: str, **kwargs):
        super().__init__(path=path, **kwargs)
        self.load_data(path)
    
    def load_data(self, path: str):
        """从JSONL文件加载数据"""
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __getitem__(self, index):
        item = self.data[index]
        # 构建问题文本
        question = item['question']
        options = item['options']
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        prompt = f"{question}\n{options_text}\n请选择正确答案的字母:"
        
        return {
            'prompt': prompt,
            'options': options,
            'answer': item['answer'],
            'id': index
        }
    
    def __len__(self):
        return len(self.data)

class VLLMAPI:
    """vLLM API客户端类"""
    
    def __init__(self, api_url: str, model_name: str, max_tokens: int = 10):
        self.api_url = api_url
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str) -> str:
        """调用vLLM API生成回复"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": 0.1  # 低温度以获得更确定的输出
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['text'].strip()
        except Exception as e:
            print(f"API调用错误: {e}")
            return ""

class MultipleChoiceEvaluator(BaseEvaluator):
    """单项选择题评估器"""
    
    def __init__(self):
        super().__init__()
    
    def score(self, predictions: List[str], references: List[str]) -> dict:
        """计算准确率"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考结果长度不一致")
        
        correct = 0
        for pred, ref in zip(predictions, references):
            # 提取预测中的第一个字母作为选项
            pred_option = self.extract_option(pred)
            if pred_option and pred_option.upper() == ref.upper():
                correct += 1
        
        accuracy = correct / len(predictions) if predictions else 0
        return {"accuracy": accuracy}
    
    def extract_option(self, text: str) -> Optional[str]:
        """从文本中提取选项字母"""
        # 查找文本中的第一个大写字母(A-D)
        for char in text:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()
        return None

def main():
    # 配置参数
    data_path = "path/to/your/dataset.jsonl"  # 替换为你的数据集路径
    api_url = "http://localhost:8000/v1/completions"  # vLLM API地址
    model_name = "your-model-name"  # 模型名称
    
    # 加载数据集
    dataset = MultipleChoiceDataset(path=data_path)
    
    # 初始化vLLM客户端
    vllm_client = VLLMAPI(api_url=api_url, model_name=model_name)
    
    # 初始化评估器
    evaluator = MultipleChoiceEvaluator()
    
    # 进行推理和评估
    predictions = []
    references = []
    
    print("开始评估模型...")
    for item in tqdm(dataset):
        prompt = item['prompt']
        references.append(item['answer'])
        
        # 获取模型回复
        response = vllm_client.generate(prompt)
        predictions.append(response)
    
    # 计算得分
    scores = evaluator.score(predictions, references)
    print(f"评估结果: {scores}")
    
    # 可选: 保存详细结果
    results = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        results.append({
            "id": i,
            "prediction": pred,
            "reference": ref,
            "correct": evaluator.extract_option(pred) == ref.upper()
        })
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "details": results}, f, ensure_ascii=False, indent=2)
    
    print("评估完成! 结果已保存到 evaluation_results.json")

if __name__ == "__main__":
    main()