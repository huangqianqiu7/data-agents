import os
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv  # 新增：导入 dotenv

# ==========================================
# 核心配置区 (Configuration)
# ==========================================
# 新增：显式加载当前目录下的 .env 文件
load_dotenv() 

RUN_DIR = r"C:\Users\18155\learn python\Agent\kddcup2026-data-agents-starter-kit-master\artifacts\runs\example_run_id"
ANS_DIR = r"C:\Users\18155\learn python\Agent\kddcup2026-data-agents-starter-kit-master\data\public\output_ans"
OUTPUT_REPORT = r"evaluation_report.csv"

# 阿里千问 API 配置 (兼容 OpenAI SDK)
API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.5-flash" # 可视需求切换为 qwen-max 或 qwen-turbo

# ==========================================
# 模块 1：数据预处理 (Data Preprocessing)
# ==========================================
def clean_and_format_csv(file_path: str) -> str:
    """
    读取 CSV 并进行确定性本地清洗：处理空值、数值精度。
    返回清洗后的 JSON 字符串，便于 LLM 理解。
    """
    try:
        # 读取数据，全量当做字符串处理以防止 pandas 自动推断导致的精度丢失，随后定向处理数值
        df = pd.read_csv(file_path)
        
        # 处理空值：归一化为空字符串
        df = df.fillna("")
        
        # 处理数值：遍历所有列，尝试将数值对象保留两位小数
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{round(float(x), 2):.2f}" if is_numeric(x) else x
            )
            
        # 转换为 records 格式的 JSON 字符串送给大模型（忽略了索引，等同于无序列表）
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def is_numeric(val):
    """判断是否为可转换为浮点数的数值类型（排除空字符串）"""
    if val == "": return False
    try:
        float(val)
        return True
    except ValueError:
        return False

# ==========================================
# 模块 2：大语言模型判定 (LLM Evaluation)
# ==========================================
def evaluate_with_llm(client: OpenAI, pred_data: str, gold_data: str) -> dict:
    """
    调用千问大模型对比两份数据的一致性。
    """
    system_prompt = """
    你是一个严格的数据评估专家。你的任务是比较两组通过 CSV 转换而来的 JSON 格式数据（预测数据和标准答案）是否完全一致。
    
    【核心评估规则】
    1. 忽略表头：JSON 数据中的键名（原CSV列名）不参与评分，仅用于辅助你理解数据结构和提高可读性。
    2. 行列无序匹配：列表的行顺序、以及单行内键值对的列顺序均不影响评分。你的评分必须基于“列值向量的无序匹配”。
    3. 姓名格式宽容度 (Name Fields)：对于姓名数据，预测文件将其拆分成“名字 (First Name)”和“姓氏 (Last Name)”两列，或者将其合并成一列“全名 (Full Name)”，这两种形式均可接受。只要语义和拼写与标准答案一致，即视为完全正确。
    4. 数值与空值 (Numeric & Nulls)：系统已在后台将所有的空值 (NULL, null, NaN) 归一化为了空字符串 `""`，并将所有数值标准化并四舍五入保留到了小数点后两位。你只需在此基础上进行严格的字符串字面值比对。
    
    【输出格式约束】
    你必须且只能输出严格的 JSON 格式数据，不要包含任何 Markdown 标记（如 ```json），不要输出任何其他解释性文字。JSON 必须包含以下两个字段：
    {
        "status": "一致" 或 "不一致",
        "reason": "如果一致填'Pass'；如果不一致，请精确指出哪一行（给出标识特征）的哪一列出现了不匹配。"
    }
    """
    
    user_prompt = f"【标准答案 (Gold)】\n{gold_data}\n\n【预测数据 (Prediction)】\n{pred_data}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.01, # 极低温度保证输出确定性
        )
        
        # 提取并解析 JSON 结果
        result_text = response.choices[0].message.content.strip()
        # 清理可能携带的 markdown 代码块标签
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        return json.loads(result_text)
    except Exception as e:
        return {"status": "评估异常", "reason": f"API调用或解析失败: {str(e)}"}

# ==========================================
# 模块 3：主调度引擎 (Main Engine)
# ==========================================
def main():
    if not API_KEY:
        print("致命错误: 未检测到 QWEN_API_KEY 环境变量。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []

    run_path = Path(RUN_DIR)
    ans_path = Path(ANS_DIR)

    if not run_path.exists():
        print(f"目标目录不存在: {run_path}")
        return

    print("开始扫描并评估任务文件...")
    
    # 遍历 run_id 下的所有 task 文件夹
    for task_dir in run_path.iterdir():
        if task_dir.is_dir() and task_dir.name.startswith("task_"):
            task_name = task_dir.name
            pred_file = task_dir / "prediction.csv"
            gold_file = ans_path / task_name / "gold.csv"

            # 1. 存在性校验
            if not pred_file.exists():
                results.append({
                    "Task": task_name,
                    "Status": "结果生成失败",
                    "Reason": "prediction.csv 不存在"
                })
                print(f"[{task_name}] 结果生成失败")
                continue
                
            if not gold_file.exists():
                results.append({
                    "Task": task_name,
                    "Status": "评估失败",
                    "Reason": "对应的 gold.csv 标准答案不存在"
                })
                print(f"[{task_name}] 缺失标准答案")
                continue

            # 2. 本地数据读取与清洗
            pred_cleaned = clean_and_format_csv(pred_file)
            gold_cleaned = clean_and_format_csv(gold_file)

            if "Error" in pred_cleaned or "Error" in gold_cleaned:
                results.append({
                    "Task": task_name,
                    "Status": "文件读取错误",
                    "Reason": "CSV解析异常"
                })
                continue

            # 3. LLM 评估比对
            print(f"[{task_name}] 数据读取成功，提交大模型评估中...")
            eval_result = evaluate_with_llm(client, pred_cleaned, gold_cleaned)
            
            results.append({
                "Task": task_name,
                "Status": eval_result.get("status", "未知异常"),
                "Reason": eval_result.get("reason", "无")
            })
            print(f"[{task_name}] 评估完成: {eval_result.get('status')}")

    # ==========================================
    # 模块 4：结果聚合导出
    # ==========================================
    if results:
        df_report = pd.DataFrame(results)
        df_report.to_csv(OUTPUT_REPORT, index=False, encoding='utf-8-sig')
        print(f"\n全部任务处理完毕。总计处理 {len(results)} 个任务。")
        print(f"测评报告已成功导出至: {os.path.abspath(OUTPUT_REPORT)}")
    else:
        print("\n未发现任何有效的 task 文件夹，无报告生成。")

if __name__ == "__main__":
    main()