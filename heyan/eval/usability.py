"""可用性评估（文档 3.3(2)：面向低识字率、60 岁以上用户的可用性验证）。

这一层回答的问题是："这套三步流程，田里的老人真的用得起来吗？"
光有模型精度不够，方案把可用性列为与精度并列的验收维度，所以这里提供：

  1. SUS（System Usability Scale）—— 10 题标准量表，业界可比；
  2. TAM（Technology Acceptance Model）—— 感知有用 / 感知易用 / 使用意愿，
     衡量"愿不愿意继续用"，这对推广期产品比 SUS 更敏感；
  3. 任务绩效 —— 三步流程（拍照→识别→听结果）的完成时间、出错次数、
     是否需要旁人协助。老年用户的真实障碍几乎都体现在"协助次数"上；
  4. 适老化检查清单 —— 字号、触控目标、对比度、图标优先等硬性 UI 指标，
     与 `heyan.config.UI_MIN_FONT_PX` / `UI_TOUCH_TARGET_PX` 对齐。

量表同时以两种方式使用：
  * 命令行 `heyan usability` 做现场访谈式录入（农技站走访时用纸笔+录入）；
  * Web 界面"设置 → 帮助改进"里直接填写，答案 POST 回 `/api/usability`。
两者共用同一份题目与计分逻辑，保证口径一致。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import UI_MIN_FONT_PX, UI_TOUCH_TARGET_PX

LIKERT_MIN, LIKERT_MAX = 1, 5
LIKERT_LABELS_ZH = {1: "很不同意", 2: "不同意", 3: "说不清", 4: "同意", 5: "很同意"}
LIKERT_LABELS_EN = {1: "Strongly disagree", 2: "Disagree", 3: "Neutral",
                    4: "Agree", 5: "Strongly agree"}

# --------------------------------------------------------------------------
# SUS：10 题标准版（奇数题为正向表述）
# --------------------------------------------------------------------------
SUS_ITEMS: List[Dict[str, str]] = [
    {"id": "S1", "positive": True,
     "zh": "我愿意经常使用这个系统", "en": "I think that I would like to use this system frequently"},
    {"id": "S2", "positive": False,
     "zh": "这个系统比我想象的复杂", "en": "I found the system unnecessarily complex"},
    {"id": "S3", "positive": True,
     "zh": "这个系统很容易上手", "en": "I thought the system was easy to use"},
    {"id": "S4", "positive": False,
     "zh": "我需要别人教我才会用", "en": "I would need the support of a technical person to use it"},
    {"id": "S5", "positive": True,
     "zh": "各个功能配合得很顺", "en": "I found the various functions in this system were well integrated"},
    {"id": "S6", "positive": False,
     "zh": "系统里有些地方前后不一致", "en": "I thought there was too much inconsistency in this system"},
    {"id": "S7", "positive": True,
     "zh": "大多数人能很快学会用", "en": "I would imagine that most people would learn to use this system very quickly"},
    {"id": "S8", "positive": False,
     "zh": "用起来觉得笨拙、不顺手", "en": "I found the system very cumbersome to use"},
    {"id": "S9", "positive": True,
     "zh": "我用起来很有信心", "en": "I felt very confident using the system"},
    {"id": "S10", "positive": False,
     "zh": "我得先学很多东西才能用", "en": "I needed to learn a lot of things before I could get going"},
]

# --------------------------------------------------------------------------
# TAM：感知有用 / 感知易用 / 使用意愿
# --------------------------------------------------------------------------
TAM_ITEMS: Dict[str, List[Dict[str, str]]] = {
    "perceived_usefulness": [
        {"id": "PU1", "zh": "用它能让我的作物少受灾", "en": "Using it would protect my crops from loss"},
        {"id": "PU2", "zh": "它能帮我更快决定打不打药", "en": "It would help me decide on spraying sooner"},
        {"id": "PU3", "zh": "它说的话我能听懂、用得上", "en": "Its spoken advice is understandable and useful"},
        {"id": "PU4", "zh": "总的来说它对我的种地有帮助", "en": "Overall it is useful for my farming"},
    ],
    "perceived_ease_of_use": [
        {"id": "PE1", "zh": "拍照这一步很简单", "en": "Taking the photo is simple"},
        {"id": "PE2", "zh": "不用别人帮我也能操作", "en": "I can operate it without anyone helping me"},
        {"id": "PE3", "zh": "字够大、按钮够大，看得清按得准", "en": "The text and buttons are large enough to see and press"},
        {"id": "PE4", "zh": "学会用它不费力气", "en": "Learning to use it is effortless"},
    ],
    "behavioral_intention": [
        {"id": "BI1", "zh": "我以后还愿意用它", "en": "I intend to keep using it"},
        {"id": "BI2", "zh": "我愿意介绍给邻居用", "en": "I would recommend it to my neighbours"},
        {"id": "BI3", "zh": "没有网络我也敢用它", "en": "I would trust it even without internet"},
    ],
}

# 任务绩效：对应文档的三步流程 + 查记录
TASKS: List[Dict[str, str]] = [
    {"id": "T1", "zh": "拍一张叶子并看到识别结果", "en": "Photograph a leaf and see the result"},
    {"id": "T2", "zh": "听一遍语音播报", "en": "Listen to the spoken result once"},
    {"id": "T3", "zh": "找到上一次拍的记录", "en": "Find the previous record"},
]

# 适老化硬性 UI 检查项（与 config 常量同源）
UI_CHECKS: List[Dict[str, Any]] = [
    {"id": "font_min", "zh": f"最小字号 ≥ {UI_MIN_FONT_PX}px", "target": UI_MIN_FONT_PX},
    {"id": "touch_min", "zh": f"触控目标 ≥ {UI_TOUCH_TARGET_PX}px", "target": UI_TOUCH_TARGET_PX},
    {"id": "icon_first", "zh": "主要操作以图标为主、文字为辅", "target": None},
    {"id": "three_steps", "zh": "主流程不超过三步", "target": 3},
    {"id": "offline", "zh": "全程无需联网", "target": None},
    {"id": "voice", "zh": "结果可语音播报", "target": None},
]


# --------------------------------------------------------------------------
# 计分
# --------------------------------------------------------------------------

def _clean(answers: Sequence[Any], n: int) -> List[int]:
    """把原始作答规整为 1~5 的整数；None/非法值视为缺失（不计分）。"""
    out: List[int] = []
    for a in answers[:n]:
        if a is None:
            continue
        try:
            v = int(round(float(a)))
        except (TypeError, ValueError):
            continue
        out.append(min(LIKERT_MAX, max(LIKERT_MIN, v)))
    return out


def score_sus(answers: Sequence[Any]) -> Dict[str, Any]:
    """SUS 标准计分：正向题 (x-1)、负向题 (5-x)，求和后 ×2.5，得 0~100。"""
    vals = _clean(answers, len(SUS_ITEMS))
    if len(vals) != len(SUS_ITEMS):
        return {"score": None, "complete": False, "answered": len(vals),
                "total": len(SUS_ITEMS)}
    raw = 0
    for item, v in zip(SUS_ITEMS, vals):
        raw += (v - 1) if item["positive"] else (LIKERT_MAX - v)
    score = raw * 2.5
    return {
        "score": round(score, 1),
        "complete": True,
        "answered": len(vals),
        "total": len(SUS_ITEMS),
        "adjective": sus_adjective(score),
        "percentile_hint": sus_percentile_hint(score),
        "learnability": bangor_learnability(vals),
        "usable_threshold": 68.0,
        "meets_threshold": score >= 68.0,
    }


def sus_adjective(score: float) -> str:
    """Bangor 形容词评级，便于向非专业读者解释分数。"""
    if score >= 85.6:
        return "excellent"
    if score >= 72.8:
        return "good"
    if score >= 52.0:
        return "ok"
    if score >= 39.0:
        return "poor"
    return "awful"


def sus_percentile_hint(score: float) -> str:
    if score >= 80:
        return "前 25%"
    if score >= 68:
        return "中等偏上"
    if score >= 50:
        return "中等偏下"
    return "后 25%"


def bangor_learnability(vals: Sequence[int]) -> Optional[float]:
    """SUS 第 4、10 题构成的可学习性子量表（0~100）。"""
    if len(vals) < 10:
        return None
    # 第 4、10 题均为负向
    sub = [(LIKERT_MAX - vals[3]), (LIKERT_MAX - vals[9])]
    return round(sum(sub) / len(sub) * 25.0, 1)


def score_tam(answers: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    """TAM 各维度取均值（1~5），并给出 0~100 归一化便于横向比较。"""
    out: Dict[str, Any] = {"dimensions": {}, "complete": True}
    for dim, items in TAM_ITEMS.items():
        vals = _clean(answers.get(dim, []), len(items))
        if len(vals) != len(items):
            out["complete"] = False
            out["dimensions"][dim] = {"mean": None, "answered": len(vals), "total": len(items)}
            continue
        mean = sum(vals) / len(vals)
        out["dimensions"][dim] = {
            "mean": round(mean, 2),
            "normalized": round((mean - 1) / (LIKERT_MAX - 1) * 100, 1),
            "answered": len(vals),
            "total": len(items),
        }
    dims = [d["mean"] for d in out["dimensions"].values() if d.get("mean") is not None]
    out["overall_mean"] = round(sum(dims) / len(dims), 2) if dims else None
    return out


def score_tasks(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """任务绩效汇总：完成率、平均耗时、协助次数、出错次数。"""
    done = [t for t in trials if t.get("completed")]
    times = [float(t["time_s"]) for t in done if t.get("time_s") is not None]
    assists = sum(int(t.get("assistance", 0) or 0) for t in trials)
    errors = sum(int(t.get("errors", 0) or 0) for t in trials)
    return {
        "tasks": len(trials),
        "completed": len(done),
        "completion_rate": round(len(done) / len(trials), 3) if trials else None,
        "mean_time_s": round(sum(times) / len(times), 2) if times else None,
        "max_time_s": round(max(times), 2) if times else None,
        "total_assistance": assists,
        "total_errors": errors,
        "needs_no_assistance": assists == 0,
    }


# --------------------------------------------------------------------------
# 会话与汇总
# --------------------------------------------------------------------------

@dataclass
class Session:
    participant_id: str = ""
    language: str = "zh"
    age_band: str = ""
    device: str = ""
    sus: List[Optional[int]] = field(default_factory=list)
    tam: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    ui_checks: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def scores(self) -> Dict[str, Any]:
        return {
            "sus": score_sus(self.sus),
            "tam": score_tam(self.tam),
            "tasks": score_tasks(self.tasks),
            "ui_checks_passed": sum(1 for v in self.ui_checks.values() if v),
            "ui_checks_total": len(self.ui_checks),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scores"] = self.scores()
        return d


@dataclass
class UsabilityStudy:
    sessions: List[Session] = field(default_factory=list)

    def add(self, session: Session) -> Session:
        self.sessions.append(session)
        return session

    def aggregate(self) -> Dict[str, Any]:
        sus_scores = [s.scores()["sus"]["score"] for s in self.sessions
                      if s.scores()["sus"]["score"] is not None]
        tam_means = [s.scores()["tam"]["overall_mean"] for s in self.sessions
                     if s.scores()["tam"]["overall_mean"] is not None]
        task_stats = [s.scores()["tasks"] for s in self.sessions if s.tasks]
        times = [t["mean_time_s"] for t in task_stats if t.get("mean_time_s") is not None]
        assists = sum(t["total_assistance"] for t in task_stats)
        completed = sum(t["completed"] for t in task_stats)
        total_tasks = sum(t["tasks"] for t in task_stats)
        return {
            "n_sessions": len(self.sessions),
            "sus_mean": round(sum(sus_scores) / len(sus_scores), 1) if sus_scores else None,
            "sus_min": min(sus_scores) if sus_scores else None,
            "sus_max": max(sus_scores) if sus_scores else None,
            "sus_meets_threshold": (sum(sus_scores) / len(sus_scores) >= 68.0)
            if sus_scores else None,
            "tam_mean": round(sum(tam_means) / len(tam_means), 2) if tam_means else None,
            "task_completion_rate": round(completed / total_tasks, 3) if total_tasks else None,
            "task_mean_time_s": round(sum(times) / len(times), 2) if times else None,
            "total_assistance": assists,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate": self.aggregate(),
            "sessions": [s.to_dict() for s in self.sessions],
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "UsabilityStudy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        study = cls()
        for item in raw.get("sessions", []):
            item.pop("scores", None)
            study.sessions.append(Session(**item))
        return study


def load_or_new(path: Optional[Path | str]) -> UsabilityStudy:
    p = Path(path) if path else None
    if p and p.exists():
        try:
            return UsabilityStudy.load(p)
        except Exception:
            pass
    return UsabilityStudy()


# --------------------------------------------------------------------------
# 供界面渲染的题目清单
# --------------------------------------------------------------------------

def questionnaire(language: str = "zh") -> Dict[str, Any]:
    """返回界面可直接渲染的量表结构（题目 + 5 级选项文案）。"""
    labels = LIKERT_LABELS_ZH if not language.startswith("en") else LIKERT_LABELS_EN
    return {
        "likert": [{"value": v, "label": labels[v]} for v in range(LIKERT_MIN, LIKERT_MAX + 1)],
        "sus": [{"id": i["id"], "text": i.get(language, i["zh"])} for i in SUS_ITEMS],
        "tam": {dim: [{"id": i["id"], "text": i.get(language, i["zh"])} for i in items]
                for dim, items in TAM_ITEMS.items()},
        "tasks": [{"id": t["id"], "text": t.get(language, t["zh"])} for t in TASKS],
        "ui_checks": [{"id": c["id"], "text": c["zh"]} for c in UI_CHECKS],
    }


# --------------------------------------------------------------------------
# 命令行交互式录入
# --------------------------------------------------------------------------

def _ask_likert(prompt: str) -> Optional[int]:
    while True:
        raw = input(f"{prompt}  (1-5, 回车跳过): ").strip()
        if not raw:
            return None
        try:
            v = int(raw)
        except ValueError:
            print("  请输入 1 到 5 的数字")
            continue
        if LIKERT_MIN <= v <= LIKERT_MAX:
            return v
        print("  请输入 1 到 5 的数字")


def run_console_session(language: str = "zh", out: Optional[Path | str] = None,
                        participant_id: str = "") -> Dict[str, Any]:
    """现场访谈式录入一份会话并存档。"""
    if not participant_id:
        participant_id = f"P{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"\n=== 可用性访谈 {participant_id} ===")
    print("说明：1=很不同意 … 5=很同意。请按用户原话记录，不要替用户判断。\n")

    session = Session(participant_id=participant_id, language=language)
    session.age_band = input("年龄段（如 60-69）: ").strip()
    session.device = input("使用设备（如 低端安卓机）: ").strip()

    print("\n-- SUS --")
    for item in SUS_ITEMS:
        v = _ask_likert(f"{item['id']}. {item.get(language, item['zh'])}")
        if v is None:
            print("  跳过该题将导致 SUS 不完整，仍继续")
            session.sus.append(None)
        else:
            session.sus.append(v)

    print("\n-- TAM --")
    for dim, items in TAM_ITEMS.items():
        vals: List[Optional[int]] = []
        for item in items:
            v = _ask_likert(f"{item['id']}. {item.get(language, item['zh'])}")
            vals.append(v)
        session.tam[dim] = vals

    print("\n-- 任务绩效 --")
    for task in TASKS:
        print(f"{task['id']}. {task.get(language, task['zh'])}")
        completed = input("  完成了吗？(y/n): ").strip().lower() in ("y", "yes", "是")
        time_s = input("  耗时秒数（可空）: ").strip()
        assistance = input("  旁人协助次数（可空, 默认0）: ").strip()
        errors = input("  出错次数（可空, 默认0）: ").strip()
        session.tasks.append({
            "id": task["id"], "completed": completed,
            "time_s": float(time_s) if time_s else None,
            "assistance": int(assistance) if assistance else 0,
            "errors": int(errors) if errors else 0,
        })

    print("\n-- 适老化检查（观察项）--")
    for check in UI_CHECKS:
        ok = input(f"{check['id']}. {check['zh']}  达标？(y/n): ").strip().lower() in ("y", "yes", "是")
        session.ui_checks[check["id"]] = ok

    session.notes = input("\n补充观察（可空）: ").strip()

    out_path = Path(out) if out else Path("artifacts/usability")
    if out_path.suffix:
        target = out_path
    else:
        target = out_path / f"session-{participant_id}.json"
    study = UsabilityStudy(sessions=[session])
    saved = study.save(target)

    report = session.to_dict()
    report["saved_to"] = str(saved)
    print(f"\n已存档 -> {saved}")
    return report
