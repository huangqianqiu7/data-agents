import os
import re
import sys
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 新增：导入 tqdm 用于显示进度条
import collections
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

# Windows 默认 GBK 终端无法输出 emoji（📊 🎯 等），统一切到 UTF-8。
# Python 3.7+ 才有 reconfigure；非 console / 已是 UTF-8 时静默跳过。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

# ==========================================
# 核心配置区 (Configuration)
# ==========================================
load_dotenv() 

# v5 之后 dabench-lc run-* 的 run_id 默认为 UTC 时间戳 (YYYYMMDDTHHMMSSZ)，
# 所以 RUNS_ROOT 只指到 runs/ 这一层，脚本启动时自动挑最新一个。
RUNS_ROOT = Path(r"C:\Users\18155\learn python\Agent\kddcup2026-data-agents-starter-kit-master\artifacts\runs")
ANS_DIR = r"C:\Users\18155\learn python\Agent\kddcup2026-data-agents-starter-kit-master\data\public\output_ans"
INPUT_DIR = r"C:\Users\18155\learn python\Agent\kddcup2026-data-agents-starter-kit-master\data\public\input"

# 严格匹配 UTC 时间戳格式（忽略旧的 example_run_id_* 与手工名称）
_UTC_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")

# 获取当前脚本所在目录，并设置 Comparison 文件夹路径
SCRIPT_DIR = Path(__file__).parent
COMPARISON_DIR = SCRIPT_DIR / "Comparison"

MAX_WORKERS = 10
LLM_REQUEST_TIMEOUT_SECONDS = 120

API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.5-flash" 

# ==========================================
# §6 评分机制相关常量（rules.zh.md §6.3 / §6.5）
# ==========================================
# §6.3 公式 Score = max(0, Recall − λ·(Extra/Pred)) 中的 λ。
# rules.zh.md 未公开 λ 取值；先用 0.1 保守值，跑出来与 LLM Status 对照不合理可调。
LAMBDA_PENALTY = 0.1

# §6.5 步骤 1：以下字面量（不区分大小写）→ 统一为空字符串 ""。
NULL_LITERALS = frozenset({"", "null", "none", "nan", "nat", "<na>"})

# ==========================================
# 模块 1：数据预处理 (保持不变)
# ==========================================
def clean_and_format_csv(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        df = df.fillna("")
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{round(float(x), 2):.2f}" if is_numeric(x) else x
            )
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def is_numeric(val):
    if val == "": return False
    try:
        float(val)
        return True
    except ValueError:
        return False

# ==========================================
# 模块 1.5：自动定位最新 run 目录（新增）
# ==========================================
def resolve_latest_run_dir() -> Path:
    """在 RUNS_ROOT 下挑一个最新的 UTC 时间戳子目录。

    UTC ISO 时间戳字符串的字典序与时间先后顺序一致，所以直接 max() 即可。
    遇到以下情况招 FileNotFoundError，调用方负责处理：
      - RUNS_ROOT 不存在
      - RUNS_ROOT 下没有任何 ^\\d{8}T\\d{6}Z$ 格式的子目录
    """
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"RUNS_ROOT 不存在: {RUNS_ROOT}")
    candidates = [
        d for d in RUNS_ROOT.iterdir()
        if d.is_dir() and _UTC_RUN_ID_RE.match(d.name)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"在 {RUNS_ROOT} 下未找到任何 UTC 时间戳格式 (YYYYMMDDTHHMMSSZ) 的 run 目录。"
            "请先跑一次 dabench-lc run-task / run-benchmark 生成产物。"
        )
    return max(candidates, key=lambda d: d.name)

# ==========================================
# 模块 1.6：§6.5 单元格标准化（rules.zh.md §6.5）
# ==========================================
# 严格按表格顺序：null → numeric → date / datetime → string fallback。
# 关键设计：
#   1) Decimal + ROUND_HALF_UP（不用 float / Python round() — banker rounding 不一致）
#   2) 日期 / datetime 用 strptime 固定格式 + datetime.fromisoformat 兜底，
#      不用 dateutil free-form parse，避免 "Tuesday" / "yes" 这类字符串被误识别为日期
#   3) 通过原字符串是否含 ":" 区分纯日期 vs datetime
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d")
_DATETIME_FALLBACK_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
)


def _is_null_literal(s: str) -> bool:
    """§6.5 步骤 1：strip().lower() ∈ NULL_LITERALS。"""
    return s.strip().lower() in NULL_LITERALS


def _try_decimal(s: str) -> str | None:
    """§6.5 步骤 2：Decimal 解析 → quantize 到 0.01 ROUND_HALF_UP。

    成功返回 ``"X.XX"`` 字符串；失败 / 非有限数 / 类型错误 → ``None``。
    """
    try:
        d = Decimal(s.strip())
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _try_parse_date(s: str) -> str | None:
    """§6.5 步骤 3：纯日期 → ISO 8601 ``YYYY-MM-DD``。"""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _try_parse_datetime(s: str) -> str | None:
    """§6.5 步骤 4：datetime → 带 tz 转 UTC ``Z``，不带 tz 保留 ISO 字符串。"""
    dt: datetime | None = None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        for fmt in _DATETIME_FALLBACK_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt_utc = dt.astimezone(timezone.utc)
        # Python isoformat 给 "+00:00"，统一替换为 'Z'
        return dt_utc.isoformat().replace("+00:00", "Z")
    return dt.isoformat()


def normalize_cell(value) -> str:
    """按 §6.5 表格规则标准化单个单元格。

    优先级：null literal → Decimal → 日期 / datetime → 字符串 fallback。
    字符串分支区分大小写（§6.5 警告），仅去掉 ``\\r\\n`` 与首尾空白。
    """
    if value is None:
        return ""
    s = str(value)
    if _is_null_literal(s):
        return ""
    n = _try_decimal(s)
    if n is not None:
        return n
    s_stripped = s.strip()
    if ":" in s_stripped:
        dt = _try_parse_datetime(s_stripped)
        if dt is not None:
            return dt
    else:
        d = _try_parse_date(s_stripped)
        if d is not None:
            return d
    # 字符串 fallback：去 \r\n + 首尾空白
    return s.replace("\r", "").replace("\n", "").strip()


# ==========================================
# 模块 1.7：§6.2/§6.3 列签名 + 评分（rules.zh.md §6.2 / §6.3）
# ==========================================
def column_signature(series: "pd.Series") -> tuple:
    """对一列 cell 标准化后排序，作为列签名（§6.2）。

    忽略列名、忽略行序、保留重复（多重集语义靠后续 Counter 体现）。
    """
    return tuple(sorted(normalize_cell(v) for v in series))


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _is_first_name_header(name: str) -> bool:
    return _normalize_header(name) in {"firstname", "givenname", "forename"}


def _is_last_name_header(name: str) -> bool:
    return _normalize_header(name) in {"lastname", "surname", "familyname"}


def _joined_name_signature(first: "pd.Series", last: "pd.Series") -> tuple:
    values = []
    for first_value, last_value in zip(first, last, strict=False):
        first_text = normalize_cell(first_value)
        last_text = normalize_cell(last_value)
        values.append(" ".join(part for part in (first_text, last_text) if part).strip())
    return tuple(sorted(values))


def logical_column_signatures(df: "pd.DataFrame") -> list[tuple]:
    used: set[str] = set()
    signatures: list[tuple] = []
    columns = list(df.columns)
    first_cols = [c for c in columns if _is_first_name_header(c)]
    last_cols = [c for c in columns if _is_last_name_header(c)]

    for first_col, last_col in zip(first_cols, last_cols, strict=False):
        signatures.append(_joined_name_signature(df[first_col], df[last_col]))
        used.add(first_col)
        used.add(last_col)

    for col in columns:
        if col not in used:
            signatures.append(column_signature(df[col]))

    return signatures


def compute_task_score(
    pred_csv: Path,
    gold_csv: Path,
    lambda_penalty: float = LAMBDA_PENALTY,
) -> dict:
    """按 §6.2/§6.3 计算单个任务 Score。

    返回字典固定包含：``score`` / ``recall`` / ``matched_cols`` / ``gold_cols``
    / ``pred_cols`` / ``extra_cols``；失败场景额外带 ``error`` 键。

    特例：
      - ``pred_csv`` 不存在 → score=0（§6.6 文件缺失给 0 分）
      - 占位 ``result\\r\\n`` → 解析为 1 列 0 行，签名空，正常走算法 → score=0
      - ``gold_cols == 0`` → recall=0 兜底
      - ``pred_cols == 0`` → 惩罚项分母 0 → 视作 0
      - 公式 ``Score = max(0, Recall − λ·Extra/Pred)``，下界 0
    """
    if not pred_csv.exists():
        return {
            "score": 0.0,
            "recall": 0.0,
            "matched_cols": 0,
            "gold_cols": 0,
            "pred_cols": 0,
            "extra_cols": 0,
            "error": "prediction.csv 不存在",
        }
    try:
        # dtype=str + keep_default_na=False + na_filter=False 防止 pandas 把
        # "NaN" / "null" 等字面值自动转 float NaN，破坏 §6.5 步骤 1 的判定。
        # skip_blank_lines=False 防止单列 CSV 中的空 cell 行被静默丢弃。
        read_kwargs = dict(
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
        )
        pred_df = pd.read_csv(pred_csv, **read_kwargs)
        gold_df = pd.read_csv(gold_csv, **read_kwargs)
    except Exception as exc:  # noqa: BLE001 - 任何 IO/parse 异常都记 error 给 0 分
        return {
            "score": 0.0,
            "recall": 0.0,
            "matched_cols": 0,
            "gold_cols": 0,
            "pred_cols": 0,
            "extra_cols": 0,
            "error": f"CSV 解析失败: {type(exc).__name__}: {exc}",
        }

    pred_counter = collections.Counter(logical_column_signatures(pred_df))
    gold_counter = collections.Counter(logical_column_signatures(gold_df))
    matched = sum(
        min(pred_counter[sig], gold_counter[sig]) for sig in gold_counter
    )
    pred_cols = sum(pred_counter.values())
    gold_cols = sum(gold_counter.values())
    extra = pred_cols - matched
    recall = matched / gold_cols if gold_cols > 0 else 0.0
    penalty = (
        lambda_penalty * (extra / pred_cols) if pred_cols > 0 else 0.0
    )
    score = max(0.0, recall - penalty)
    return {
        "score": round(score, 4),
        "recall": round(recall, 4),
        "matched_cols": matched,
        "gold_cols": gold_cols,
        "pred_cols": pred_cols,
        "extra_cols": extra,
    }


# ==========================================
# 模块 1.8：§6 覆盖率检查（rules.zh.md §6 + 用户需求）
# ==========================================
def check_full_coverage(
    input_dir: Path, run_dir: Path
) -> tuple[bool, list[str]]:
    """检查 ``input_dir`` 下每个 ``task_*`` 都在 ``run_dir/<task_id>/`` 下有 ``prediction.csv``。

    用户语义："全部输入文件都有对应的输出文件（哪怕最后生成结果失败也算）"
    → 文件存在即视为已覆盖（不检查内容）。

    Returns:
      ``(is_complete, missing)``，``missing`` 已排序。
      ``input_dir`` 不存在或不含任何 ``task_*`` 子目录时视为完整 ``(True, [])``。
    """
    if not input_dir.exists():
        return True, []
    expected = sorted(
        d.name
        for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    )
    missing = [
        t for t in expected
        if not (run_dir / t / "prediction.csv").exists()
    ]
    return (not missing, missing)


# ==========================================
# 模块 2：大语言模型判定 (保持不变)
# ==========================================
def evaluate_with_llm(client: OpenAI, pred_data: str, gold_data: str) -> dict:
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
            temperature=0.01, 
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        return json.loads(result_text)
    except Exception as e:
        return {"status": "评估异常", "reason": f"API调用或解析失败: {str(e)}"}

# ==========================================
# 模块 2.5：读取任务难度 (新增)
# ==========================================
def get_task_difficulty(task_name: str) -> str:
    """从 input 目录下对应 task 的 task.json 中读取 difficulty 字段"""
    task_json_path = Path(INPUT_DIR) / task_name / "task.json"
    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("difficulty", "unknown")
    except Exception:
        return "unknown"

# ==========================================
# 模块 3：单任务执行原子逻辑（LLM Status + §6 算法 Score 并存）
# ==========================================
def _build_score_record(score_info: dict) -> dict:
    """把 ``compute_task_score`` 的结果展开为 CSV 列字段（中文列名）。"""
    return {
        "得分": score_info["score"],
        "召回率": score_info["recall"],
        "匹配列数": score_info["matched_cols"],
        "标答列数": score_info["gold_cols"],
        "预测列数": score_info["pred_cols"],
        "多余列数": score_info["extra_cols"],
        "评分错误": score_info.get("error", ""),
    }


def process_single_task(task_dir: Path, ans_path: Path, client: OpenAI) -> dict:
    task_name = task_dir.name
    difficulty = get_task_difficulty(task_name)
    pred_file = task_dir / "prediction.csv"
    gold_file = ans_path / task_name / "gold.csv"

    # §6 评分先算（compute_task_score 内部已涵盖文件缺失 / 解析异常分支，
    # 失败给 0 分 + error；因此每条结果都有 Score 字段，DataFrame 列对齐）。
    score_info = compute_task_score(pred_file, gold_file)
    score_record = _build_score_record(score_info)

    if not pred_file.exists():
        return {
            "任务": task_name, "难度": difficulty,
            "一致性": "结果生成失败", "原因": "prediction.csv 不存在",
            **score_record,
        }

    if not gold_file.exists():
        return {
            "任务": task_name, "难度": difficulty,
            "一致性": "评估失败", "原因": "对应的 gold.csv 标准答案不存在",
            **score_record,
        }

    pred_cleaned = clean_and_format_csv(pred_file)
    gold_cleaned = clean_and_format_csv(gold_file)

    if "Error" in pred_cleaned or "Error" in gold_cleaned:
        return {
            "任务": task_name, "难度": difficulty,
            "一致性": "文件读取错误", "原因": "CSV解析异常",
            **score_record,
        }

    # 移除打印，避免干扰进度条
    tqdm.write(f"开始评估: {task_name}")
    eval_result = evaluate_with_llm(client, pred_cleaned, gold_cleaned)
    tqdm.write(
        f"完成评估: {task_name} -> "
        f"{eval_result.get('status', '未知异常')} | 得分={score_record['得分']}"
    )

    return {
        "任务": task_name,
        "难度": difficulty,
        "一致性": eval_result.get("status", "未知异常"),
        "原因": eval_result.get("reason", "无"),
        **score_record,
    }

# ==========================================
# 模块 5：处理输出文件命名 (新增)
# ==========================================
def get_output_filepath(run_dir_name: str) -> Path:
    """生成与 *run_dir_name* 同名的 CSV 报告路径。

    首次输出为 ``<run_dir_name>.csv``；同名冲突时依次加
    ``-v2`` / ``-v3`` / ... 后缀，直到找到未占用的名字。
    例：``20260508T065149Z.csv`` → ``20260508T065149Z-v2.csv``。
    """
    if not COMPARISON_DIR.exists():
        COMPARISON_DIR.mkdir(parents=True)

    base_path = COMPARISON_DIR / f"{run_dir_name}.csv"
    if not base_path.exists():
        return base_path

    version = 2
    while True:
        candidate = COMPARISON_DIR / f"{run_dir_name}-v{version}.csv"
        if not candidate.exists():
            return candidate
        version += 1

# ==========================================
# 模块 4：并发主调度引擎 (集成 tqdm 和统计)
# ==========================================
def main():
    if not API_KEY:
        print("致命错误: 未检测到 LLM_API_KEY 环境变量。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []

    try:
        run_path = resolve_latest_run_dir()
    except FileNotFoundError as exc:
        print(f"无法定位 run 目录: {exc}")
        return
    print(f"自动选中最新 run: {run_path.name}")
    ans_path = Path(ANS_DIR)

    task_dirs = [d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("task_")]
    total_tasks = len(task_dirs)
    
    print(f"扫描完毕。共发现 {total_tasks} 个任务，启用 {MAX_WORKERS} 并发进行处理...\n")

    # 使用 tqdm 包装 as_completed 实现进度条
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_single_task, task_dir, ans_path, client): task_dir.name 
            for task_dir in task_dirs
        }
        
        # 使用 tqdm 创建进度条，设置总任务数
        with tqdm(total=total_tasks, desc="评估进度", unit="task") as pbar:
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    fallback_score = _build_score_record({
                        "score": 0.0, "recall": 0.0,
                        "matched_cols": 0, "gold_cols": 0,
                        "pred_cols": 0, "extra_cols": 0,
                        "error": "系统异常",
                    })
                    results.append({
                        "任务": task_name,
                        "难度": "unknown",
                        "一致性": "系统异常",
                        "原因": str(exc),
                        **fallback_score,
                    })
                finally:
                    # 无论成功失败，更新进度条
                    pbar.update(1)

    # ==========================================
    # 结果聚合导出与统计 (新增统计和动态文件逻辑)
    # ==========================================
    if results:
        df_report = pd.DataFrame(results).sort_values(by="任务")
        
        output_filepath = get_output_filepath(run_path.name)
        df_report.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*40)
        print("📊 任务评估统计结果")
        print("="*40)
        
        # 统计「一致性」分布
        status_counts = collections.Counter(res["一致性"] for res in results)
        for status, count in status_counts.items():
            print(f"- {status}: {count} 个")
        
        # 按难度分组统计
        print("\n" + "-"*40)
        print("📈 按难度分组统计")
        print("-"*40)
        difficulty_groups = collections.defaultdict(list)
        for res in results:
            difficulty_groups[res["难度"]].append(res["一致性"])
        
        for diff in sorted(difficulty_groups.keys()):
            statuses = difficulty_groups[diff]
            total = len(statuses)
            consistent = sum(1 for s in statuses if s == "一致")
            print(f"\n【{diff}】共 {total} 个任务，一致 {consistent} 个，正确率 {consistent/total*100:.1f}%")
            diff_status_counts = collections.Counter(statuses)
            for status, count in diff_status_counts.items():
                print(f"  - {status}: {count} 个")

        # ==========================================
        # §6 评分（rules.zh.md §6.3 公式 + §6.7 总分）
        # ==========================================
        print("\n" + "-"*40)
        print(f"🎯 §6 评分（λ={LAMBDA_PENALTY}，公式: Score = max(0, Recall − λ·Extra/Pred)）")
        print("-"*40)

        all_scores = [r.get("得分", 0.0) for r in results]
        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # 按难度分组的 mean 得分
        diff_score_groups = collections.defaultdict(list)
        for res in results:
            diff_score_groups[res["难度"]].append(res.get("得分", 0.0))
        for diff in sorted(diff_score_groups.keys()):
            ds = diff_score_groups[diff]
            print(f"  【{diff}】mean Score = {sum(ds)/len(ds):.4f} ({len(ds)} 任务)")

        print(f"\n  全部已评 {len(all_scores)} 任务的 mean Score = {mean_score:.4f}")

        # 覆盖闸门：所有 INPUT 任务都要在 run 下有 prediction.csv
        is_complete, missing = check_full_coverage(Path(INPUT_DIR), run_path)
        print("\n" + "="*40)
        if is_complete:
            print(f"✅ 全部输入任务都已生成 prediction.csv")
            print(f"🏁 最终得分 (mean Score) = {mean_score:.4f}")
        else:
            print(f"⚠️ 覆盖不完整：{len(missing)} 个输入任务缺 prediction.csv，按用户要求不出最终得分")
            for t in missing[:10]:
                print(f"   - {t}")
            if len(missing) > 10:
                print(f"   ... 还有 {len(missing) - 10} 个未列出")

        print("\n" + "="*40)
        print(f"总计处理 {len(results)} 个任务。")
        print(f"测评报告已成功导出至: {output_filepath.resolve()}")
    else:
        print("\n未发现任何有效的 task 文件夹，无报告生成。")

if __name__ == "__main__":
    main()