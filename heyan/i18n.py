"""界面文案的多语言表（文档 3.3(1)：多语言语音支持 + 图标为主文字为辅）。

这份表同时服务两个消费方，所以必须是唯一来源：

  1. 界面渲染 —— `t(key, lang)` 直接给出屏幕上显示的字；
  2. 离线语音包 —— `heyan.tts.phrases` 按同样的 key 生成 `ui_<key>` 的 slug，
     构建期预渲染成 wav，运行期零合成。

两处共用一张表，就不会出现"屏幕上写粤语、喇叭里念普通话"这类分裂。

关于方言文案：客家话与潮汕话没有统一的书面正字规范，下面的写法按粤东地区
常见的汉字借音方案给出，能让母语者读懂，但**建议由当地农技站组织母语者校订**。
`DIALECT_REVIEW_NEEDED` 里列出了需要复核的语言，界面设置页会如实标注。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .tts.languages import SPOKEN_ORDER, fallback_chain, is_supported

# 需要母语者校订的方言文案
DIALECT_REVIEW_NEEDED = ("hak", "teochew")

DEFAULT_LANG = "zh"

# key -> {lang: text}
STRINGS: Dict[str, Dict[str, str]] = {
    # ---- 品牌与首屏 ----
    "app_name": {
        "zh": "禾眼", "yue": "禾眼", "hak": "禾眼", "teochew": "禾眼", "en": "HeYan",
    },
    "app_tagline": {
        "zh": "拍一张，就知道作物怎么了",
        "yue": "影一張，就知作物點解唔對路",
        "hak": "拍一張，就知作物做麼個",
        "teochew": "拍一張，就知作物怎呢個",
        "en": "One photo tells you what is wrong with your crop",
    },
    "offline_ready": {
        "zh": "不用联网，随时能用",
        "yue": "唔使上網，隨時用得",
        "hak": "毋使上網，隨時用得",
        "teochew": "免用上網，隨時会用",
        "en": "Works offline, anytime",
    },

    # ---- 三步流程：拍照 ----
    "tap_to_shoot": {
        "zh": "点这里拍照",
        "yue": "撳呢度影相",
        "hak": "撳這下拍照",
        "teochew": "點這塊拍照",
        "en": "Tap to take a photo",
    },
    "step_capture": {
        "zh": "第一步 拍照", "yue": "第一步 影相", "hak": "第一步 拍照",
        "teochew": "第一步 拍照", "en": "Step 1 Photo",
    },
    "step_recognize": {
        "zh": "第二步 识别", "yue": "第二步 識別", "hak": "第二步 辨識",
        "teochew": "第二步 識別", "en": "Step 2 Recognize",
    },
    "step_listen": {
        "zh": "第三步 听结果", "yue": "第三步 聽結果", "hak": "第三步 聽結果",
        "teochew": "第三步 聽結果", "en": "Step 3 Listen",
    },
    "choose_from_album": {
        "zh": "从相册选", "yue": "喺相簿揀", "hak": "相簿肚揀",
        "teochew": "相簿內揀", "en": "Choose from album",
    },
    "aim_hint": {
        "zh": "把叶子放在框中间，靠远一点拍清楚",
        "yue": "將葉放喺格仔中間，行近啲影清楚",
        "hak": "葉仔擺到框箇中間，行近滴拍清楚",
        "teochew": "葉囝放佇框中央，走近滴拍清楚",
        "en": "Center the leaf in the frame and move close",
    },

    # ---- 三步流程：识别 ----
    "analyzing": {
        "zh": "正在识别，请稍候",
        "yue": "識別緊，請稍等",
        "hak": "辨識緊，請等一下",
        "teochew": "識別中，請稍等",
        "en": "Recognizing, please wait",
    },

    # ---- 三步流程：结果 ----
    "result_ready": {
        "zh": "识别完成", "yue": "結果出咗", "hak": "結果出來了",
        "teochew": "結果出來了", "en": "Result ready",
    },
    "result_name": {
        "zh": "这是什么", "yue": "呢個係乜嘢", "hak": "這係做麼個",
        "teochew": "這係乜個", "en": "What it is",
    },
    "result_severity": {
        "zh": "有多严重", "yue": "幾嚴重", "hak": "幾嚴重",
        "teochew": "若嚴重", "en": "How severe",
    },
    "result_actions": {
        "zh": "该怎么做", "yue": "應該點做", "hak": "應該愛按怎做",
        "teochew": "應該愛怎呢做", "en": "What to do",
    },
    "confidence": {
        "zh": "把握程度", "yue": "把握程度", "hak": "把握程度",
        "teochew": "把握程度", "en": "Confidence",
    },
    "replay_voice": {
        "zh": "再听一次", "yue": "再聽一次", "hak": "再聽一次",
        "teochew": "再聽一次", "en": "Play again",
    },
    "retake": {
        "zh": "再拍一次", "yue": "再影一次", "hak": "再拍一張",
        "teochew": "再拍一次", "en": "Take another",
    },
    "save_record": {
        "zh": "存起来", "yue": "儲存落嚟", "hak": "存起來",
        "teochew": "存起來", "en": "Save record",
    },
    "low_confidence": {
        "zh": "看不清楚，请靠近一点再拍一次",
        "yue": "睇唔清楚，請行近啲再影一次",
        "hak": "看毋清楚，請行近滴再拍一次",
        "teochew": "睇唔清楚，請走近滴再拍一次",
        "en": "Not clear. Please move closer and try again.",
    },
    "healthy_note": {
        "zh": "没发现问题，继续保持",
        "yue": "冇睇到問題，繼續保持",
        "hak": "麼個問題都無，繼續保持",
        "teochew": "無發現問題，繼續保持",
        "en": "No problem found. Keep it up.",
    },

    # ---- 底部导航 ----
    "nav_home": {
        "zh": "拍照识别", "yue": "影相識別", "hak": "拍照辨識",
        "teochew": "拍照識別", "en": "Scan",
    },
    "nav_history": {
        "zh": "记录", "yue": "紀錄", "hak": "紀錄", "teochew": "記錄", "en": "Records",
    },
    "nav_settings": {
        "zh": "设置", "yue": "設定", "hak": "設定", "teochew": "設定", "en": "Settings",
    },

    # ---- 记录页 ----
    "history_title": {
        "zh": "识别记录", "yue": "識別紀錄", "hak": "辨識紀錄",
        "teochew": "識別記錄", "en": "Recognition records",
    },
    "history_empty": {
        "zh": "还没有记录，去拍一张吧",
        "yue": "未有紀錄，去影一張啦",
        "hak": "還無紀錄，去拍一張啊",
        "teochew": "還無記錄，去拍一張啊",
        "en": "No records yet. Take a photo.",
    },
    "history_filter_all": {
        "zh": "全部", "yue": "全部", "hak": "全部", "teochew": "全部", "en": "All",
    },
    "delete": {
        "zh": "删除", "yue": "刪除", "hak": "刪除", "teochew": "刪除", "en": "Delete",
    },
    "export_records": {
        "zh": "导出数据", "yue": "匯出資料", "hak": "匯出資料",
        "teochew": "匯出資料", "en": "Export data",
    },
    "export_usb": {
        "zh": "导出到U盘", "yue": "匯出去USB", "hak": "匯出到USB",
        "teochew": "匯出到USB", "en": "Export to USB",
    },
    "export_download": {
        "zh": "下载文件", "yue": "下載檔案", "hak": "下載檔案",
        "teochew": "下載檔案", "en": "Download file",
    },
    "records_count": {
        "zh": "共 {n} 条记录", "yue": "一共 {n} 條紀錄", "hak": "共下 {n} 條紀錄",
        "teochew": "共 {n} 條記錄", "en": "{n} records in total",
    },

    # ---- 设置页 ----
    "language": {
        "zh": "播报语言", "yue": "播報語言", "hak": "播報語言",
        "teochew": "播報語言", "en": "Voice language",
    },
    "text_size": {
        "zh": "字体大小", "yue": "字體大細", "hak": "字體大小",
        "teochew": "字體大小", "en": "Text size",
    },
    "voice_speed": {
        "zh": "播报速度", "yue": "播報速度", "hak": "播報速度",
        "teochew": "播報速度", "en": "Voice speed",
    },
    "auto_voice": {
        "zh": "出结果就自动播报", "yue": "出結果就自動播報",
        "hak": "出結果就自動播報", "teochew": "出結果就自動播報",
        "en": "Speak result automatically",
    },
    "size_standard": {
        "zh": "标准", "yue": "標準", "hak": "標準", "teochew": "標準", "en": "Standard",
    },
    "size_large": {
        "zh": "大", "yue": "大", "hak": "大", "teochew": "大", "en": "Large",
    },
    "size_huge": {
        "zh": "特大", "yue": "特大", "hak": "特大", "teochew": "特大", "en": "Extra large",
    },
    "model_info": {
        "zh": "模型信息", "yue": "模型資訊", "hak": "模型資訊",
        "teochew": "模型資訊", "en": "Model info",
    },
    "dialect_review_note": {
        "zh": "该方言文字待母语者校订，语音暂用相近语言播报",
        "yue": "呢種方言文字待母語者校訂，語音暫用相近語言播報",
        "hak": "這方言文字還愛母語者校訂，語音暫用相近語言播報",
        "teochew": "這方言文字待母語者校訂，語音暫用相近語言播報",
        "en": "Dialect text pending native review; voice falls back to a related language.",
    },

    # ---- 系统状态 ----
    "sys_no_camera": {
        "zh": "打不开摄像头，请从相册选择照片",
        "yue": "鏡頭開唔到，請喺相簿揀相",
        "hak": "鏡頭開毋到，請在相簿揀相",
        "teochew": "鏡頭開袂起，請佇相簿揀相",
        "en": "Camera unavailable. Please pick a photo.",
    },
    "sys_error": {
        "zh": "识别失败，请再试一次",
        "yue": "識別失敗，請再試一次",
        "hak": "辨識失敗，請再試一次",
        "teochew": "識別失敗，請再試一次",
        "en": "Recognition failed. Please try again.",
    },
    "sys_saved": {
        "zh": "记录已保存", "yue": "記錄已儲存", "hak": "紀錄已存好",
        "teochew": "記錄已存好", "en": "Record saved",
    },
    "sys_exported": {
        "zh": "数据已导出", "yue": "資料已匯出", "hak": "資料已匯出",
        "teochew": "資料已匯出", "en": "Data exported",
    },
    "sys_voice_unavailable": {
        "zh": "本机没有离线语音，已用文字显示",
        "yue": "呢部機冇離線語音，已用文字顯示",
        "hak": "這機無離線語音，已用文字顯示",
        "teochew": "這機無離線語音，已用文字顯示",
        "en": "No offline voice on this device. Showing text.",
    },
    "sys_dialect_fallback": {
        "zh": "本地语音暂不支持该方言，已用相近语言播报",
        "yue": "本地語音未支援呢種方言，已用相近語言播報",
        "hak": "本地語音還未支援這方言，已用相近語言播報",
        "teochew": "本地語音未支援這方言，已用相近語言播報",
        "en": "This dialect has no offline voice yet. Using a related one.",
    },
    "sys_no_model": {
        "zh": "还没有安装识别模型，请先在电脑上构建模型包",
        "yue": "未有安裝識別模型，請先在電腦度建模型包",
        "hak": "還無安裝辨識模型，請先在電腦建模型包",
        "teochew": "還未安裝識別模型，請先在電腦建模型包",
        "en": "No model installed. Build a bundle on a PC first.",
    },
    "sys_ready": {
        "zh": "准备好了，不用联网", "yue": "準備好，唔使上網",
        "hak": "準備好，毋使上網", "teochew": "準備好，免用上網",
        "en": "Ready. No internet needed.",
    },

    # ---- 推广层（保险 / 补贴 / 农资）----
    "insurance_claimable": {
        "zh": "可作为保险理赔证据", "yue": "可作保險理賠證據",
        "hak": "可作保險理賠證據", "teochew": "可作保險理賠證據",
        "en": "Usable as insurance claim evidence",
    },
    "agro_input_hint": {
        "zh": "农资站可备货", "yue": "農資站可備貨", "hak": "農資站可備貨",
        "teochew": "農資站可備貨", "en": "Agri-supply stocking hint",
    },
    "subsidy_hint": {
        "zh": "可作补贴申报记录", "yue": "可作補貼申報紀錄",
        "hak": "可作補貼申報紀錄", "teochew": "可作補貼申報記錄",
        "en": "Usable for subsidy filing",
    },
}


def t(key: str, lang: str = DEFAULT_LANG, **fmt) -> str:
    """取一条界面文案。缺失时按语言回退链找，最后退回中文，再退回 key 本身。"""
    table = STRINGS.get(key)
    if table is None:
        return key
    chain = fallback_chain(lang) if is_supported(lang) else [lang, DEFAULT_LANG]
    for code in chain:
        text = table.get(code)
        if text:
            return text.format(**fmt) if fmt else text
    text = table.get(DEFAULT_LANG, key)
    return text.format(**fmt) if fmt and "{" in text else text


def table(lang: str = DEFAULT_LANG) -> Dict[str, str]:
    """整份界面文案，供前端一次拉走（避免逐条请求）。"""
    return {key: t(key, lang) for key in STRINGS}


def ui_keys() -> List[str]:
    return list(STRINGS)


def ui_phrases(lang: str = DEFAULT_LANG) -> Dict[str, str]:
    """兼容旧接口：返回 key -> 文案。"""
    return table(lang)


def coverage() -> Dict[str, Dict[str, object]]:
    """每种语言的文案覆盖率，用于交付前自查。"""
    out: Dict[str, Dict[str, object]] = {}
    for lang in SPOKEN_ORDER:
        have = sum(1 for k in STRINGS if STRINGS[k].get(lang))
        out[lang] = {"keys": have, "total": len(STRINGS),
                     "complete": have == len(STRINGS),
                     "review_needed": lang in DIALECT_REVIEW_NEEDED}
    return out
