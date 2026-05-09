import json
import re
from dataclasses import dataclass

# =====================================================================
# 前置定义：定义解析结果的数据结构
# =====================================================================
@dataclass
class ModelStep:
    """存放解析后的大模型步骤信息的容器"""
    thought: str
    action: str
    action_input: dict
    raw_response: str


# =====================================================================
# LLM 输出解析辅助函数：扒掉 Markdown 外衣
# =====================================================================
def _strip_json_fence(raw_response: str) -> str:
    """
    大模型经常喜欢在 JSON 外面白白加上 ```json 和 ``` 的 Markdown 代码块标记。
    这个函数的作用就是把这些外衣脱掉，只提取里面纯净的 JSON 字符串。
    """
    text = raw_response.strip() # strip()去除字符串两头的空白字符
    
    # 尝试匹配 ```json ... ```，re.DOTALL 表示 . 可以匹配换行符
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
        
    # 如果没写 json，只是写了 ``` ... ``` 也兼容处理
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
        
    # 如果大模型很乖，直接输出了裸的 JSON，就原样返回
    return text


# =====================================================================
# LLM 输出解析辅助函数：严格提取单 JSON 对象
# =====================================================================
def _load_single_json_object(text: str) -> dict[str, object]:
    """
    不仅要把文本解析成字典，还要防止大模型“话痨”在 JSON 后面追加废话。
    """
    # json.JSONDecoder().raw_decode 会返回解析出的对象和解析停止的索引位置 (end)
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

    # 检查 JSON 解析完之后，后面还有没有多余的字符串
    remainder = text[end:].strip()
    if remainder:
        # 去掉换行和制表符等空白字符
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            # 强硬报错：如果后面还有字（比如 "Here is your answer:"), 直接抛报错
            raise ValueError("Model response must contain only one JSON object.")
            
    # 判断解析出来的是不是一个字典 (dict)
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
        
    return payload


# =====================================================================
# 核心解析函数：将文本转化为大模型的“思维步骤”
# =====================================================================
def parse_model_step(raw_response: str) -> ModelStep:
    """
    将大模型返回的文本，经过脱衣、提取、校验，最终变成一个规范的 ModelStep 对象。
    """
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    # 提取三大核心要素：思考过程，工具名称、工具参数
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    
    # 类型安全校验（防御性编程：永远不要完全信任大模型的输出格式）
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        print("=====================================================================")
        print(f"action_input 是一个 {type(action_input)} 类型")
        print(action_input)
        print("=====================================================================")
        raise ValueError("action_input must be a JSON object.")

    print("=====================================================================")
    print(f"thought: {thought}")
    print(f"action: {action}")
    print(f"action_input: {action_input}")
    print("=====================================================================")
    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


# =====================================================================
# 独立测试模块：展示代码运行效果
# =====================================================================
if __name__ == "__main__":
    print("🚀 开始测试 LLM JSON 解析器...\n" + "="*50)

    # test_case_1 =  "```json\n{\"thought\":\"Based on the knowledge.md file, severe thrombosis corresponds to Thrombosis = 2 in the Examination table. I need to join Patient and Examination tables to get patients with severe thrombosis, returning their ID, SEX, and Diagnosis.\",\"action\":\"execute_python\",\"action_input\":\"import json\\n\\n# Read the JSON files\\nwith open('json/Patient.json', 'r') as f:\\n    patient_data = json.load(f)\\n\\nwith open('json/Examination.json', 'r') as f:\\n    exam_data = json.load(f)\\n\\n# Create a dictionary of patients by ID\\npatient_dict = {p['ID']: p for p in patient_data['records']}\\n\\n# Find patients with severe thrombosis (Thrombosis = 2)\\nsevere_thrombosis_patients = []\\nfor exam in exam_data['records']:\\n    if exam.get('Thrombosis') == 2:\\n        pid = exam['ID']\\n        if pid in patient_dict:\\n            p = patient_dict[pid]\\n            severe_thrombosis_patients.append({\\n                'ID': pid,\\n                'SEX': p.get('SEX', ''),\\n                'Diagnosis': p.get('Diagnosis', '')\\n            })\\n\\nprint(f'Found {len(severe_thrombosis_patients)} patients with severe thrombosis')\\nfor p in severe_thrombosis_patients[:10]:\\n    print(p)\\n\\n# Prepare answer table\\ncolumns = ['ID', 'SEX', 'Diagnosis']\\nrows = [[p['ID'], p['SEX'], p['Diagnosis']] for p in severe_thrombosis_patients]\\nprint(f'Total rows: {len(rows)}')\"}\n```"
    # print("▶ 测试用例 1: 标准 Markdown 格式")
    # step1 = parse_model_step(test_case_1)
    # print(f"✅ 解析成功! Action: {step1.action}, Thought: {step1.thought}\n")

    test_case_2 = "```json\n{\"thought\":\"I need to run Python code to process the JSON files and find patients with severe thrombosis (Thrombosis = 2).\",\"action\":\"execute_python\",\"action_input\":{\"code\":\"import json\\n\\n# Read the JSON files\\nwith open('json/Patient.json', 'r') as f:\\n    patient_data = json.load(f)\\n\\nwith open('json/Examination.json', 'r') as f:\\n    exam_data = json.load(f)\\n\\n# Create a dictionary of patients by ID\\npatient_dict = {p['ID']: p for p in patient_data['records']}\\n\\n# Find patients with severe thrombosis (Thrombosis = 2)\\nsevere_thrombosis_patients = []\\nfor exam in exam_data['records']:\\n    if exam.get('Thrombosis') == 2:\\n        pid = exam['ID']\\n        if pid in patient_dict:\\n            p = patient_dict[pid]\\n            severe_thrombosis_patients.append({\\n                'ID': pid,\\n                'SEX': p.get('SEX', ''),\\n                'Diagnosis': p.get('Diagnosis', '')\\n            })\\n\\nprint(f'Found {len(severe_thrombosis_patients)} patients with severe thrombosis')\\nfor p in severe_thrombosis_patients[:10]:\\n    print(p)\\n\\n# Prepare answer table\\ncolumns = ['ID', 'SEX', 'Diagnosis']\\nrows = [[p['ID'], p['SEX'], p['Diagnosis']] for p in severe_thrombosis_patients]\\nprint(f'Total rows: {len(rows)}')\"}}\n```"
    print("▶ 测试用例 2: 包含多行代码的 JSON")
    step2 = parse_model_step(test_case_2)
    print(f"✅ 解析成功! Action: {step2.action}, Thought: {step2.thought}\n")

    


