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

    # ---- 作物与胁迫类型（结果页/记录页的筛选与标签） ----
    "crop_rice": {
        "zh": "水稻", "yue": "水稻", "hak": "水稻", "teochew": "水稻", "en": "Rice",
    },
    "crop_peanut": {
        "zh": "花生", "yue": "花生", "hak": "花生", "teochew": "花生", "en": "Peanut",
    },
    "crop_vegetable": {
        "zh": "蔬菜", "yue": "蔬菜", "hak": "蔬菜", "teochew": "蔬菜", "en": "Vegetable",
    },
    "crop_none": {
        "zh": "其他", "yue": "其他", "hak": "其他", "teochew": "其他", "en": "Other",
    },
    "stress_healthy": {
        "zh": "正常", "yue": "正常", "hak": "正常", "teochew": "正常", "en": "Healthy",
    },
    "stress_disease": {
        "zh": "病害", "yue": "病害", "hak": "病害", "teochew": "病害", "en": "Disease",
    },
    "stress_nutrient": {
        "zh": "缺肥", "yue": "缺肥", "hak": "缺肥", "teochew": "缺肥", "en": "Nutrient",
    },
    "stress_water": {
        "zh": "缺水", "yue": "缺水", "hak": "缺水", "teochew": "缺水", "en": "Water",
    },
    "stress_invalid": {
        "zh": "拍得不清楚", "yue": "影得唔清楚", "hak": "拍得毋清楚",
        "teochew": "拍袂清楚", "en": "Unclear photo",
    },

    # ---- 拍照页补充 ----
    "flip_camera": {
        "zh": "翻转镜头", "yue": "反轉鏡頭", "hak": "反轉鏡頭",
        "teochew": "反轉鏡頭", "en": "Flip camera",
    },
    "recent_captures": {
        "zh": "最近拍的", "yue": "最近影嘅", "hak": "最近拍个",
        "teochew": "最近拍个", "en": "Recent",
    },

    # ---- 结果页补充 ----
    "view_detail": {
        "zh": "看细节", "yue": "睇細節", "hak": "看細節", "teochew": "看細節",
        "en": "Details",
    },
    "candidates": {
        "zh": "其他可能", "yue": "其他可能", "hak": "其他可能",
        "teochew": "其他可能", "en": "Other possibilities",
    },
    "photo": {
        "zh": "照片", "yue": "相片", "hak": "相片", "teochew": "相片", "en": "Photo",
    },
    "playing": {
        "zh": "正在播报", "yue": "播報緊", "hak": "播報緊", "teochew": "播報中",
        "en": "Speaking",
    },
    "stop_voice": {
        "zh": "停止播报", "yue": "停止播報", "hak": "停止播報",
        "teochew": "停止播報", "en": "Stop",
    },
    "record_saved": {
        "zh": "已经存好了", "yue": "已經存好咗", "hak": "已經存好了",
        "teochew": "已經存好了", "en": "Saved",
    },
    "view_record": {
        "zh": "查看记录", "yue": "睇紀錄", "hak": "看紀錄", "teochew": "看記錄",
        "en": "View record",
    },

    # ---- 记录页 ----
    "load_more": {
        "zh": "加载更多", "yue": "載多啲", "hak": "載多啲", "teochew": "載多啲",
        "en": "Load more",
    },
    "no_records_match": {
        "zh": "没有符合条件的记录", "yue": "冇符合條件嘅紀錄",
        "hak": "無符合條件个紀錄", "teochew": "無符合條件个記錄",
        "en": "No matching records",
    },
    "record_detail": {
        "zh": "记录详情", "yue": "紀錄詳情", "hak": "紀錄詳情",
        "teochew": "記錄詳情", "en": "Record detail",
    },
    "followup": {
        "zh": "处理跟进", "yue": "處理跟進", "hak": "處理跟進",
        "teochew": "處理跟進", "en": "Follow-up",
    },
    "action_taken": {
        "zh": "做了什么", "yue": "做咗乜嘢", "hak": "做了麼個",
        "teochew": "做了乜個", "en": "Action taken",
    },
    "product_used": {
        "zh": "用了什么药肥", "yue": "用咗乜嘢藥肥", "hak": "用了麼個藥肥",
        "teochew": "用了乜個藥肥", "en": "Product used",
    },
    "effect": {
        "zh": "有没有效", "yue": "有冇效", "hak": "有無效", "teochew": "有效無",
        "en": "Effect",
    },
    "effect_better": {
        "zh": "好转了", "yue": "好轉咗", "hak": "好轉了", "teochew": "好轉了",
        "en": "Better",
    },
    "effect_same": {
        "zh": "没变化", "yue": "冇變化", "hak": "無變化", "teochew": "無變化",
        "en": "No change",
    },
    "effect_worse": {
        "zh": "更严重了", "yue": "更嚴重咗", "hak": "更嚴重了", "teochew": "更嚴重了",
        "en": "Worse",
    },
    "consent_title": {
        "zh": "数据分享授权", "yue": "資料分享授權", "hak": "資料分享授權",
        "teochew": "資料分享授權", "en": "Data sharing consent",
    },
    "consent_insurance": {
        "zh": "给保险公司", "yue": "畀保險公司", "hak": "分保險公司",
        "teochew": "分保險公司", "en": "To insurer",
    },
    "consent_subsidy": {
        "zh": "给补贴部门", "yue": "畀補貼部門", "hak": "分補貼部門",
        "teochew": "分補貼部門", "en": "To subsidy office",
    },
    "consent_supplier": {
        "zh": "给农资站", "yue": "畀農資站", "hak": "分農資站", "teochew": "分農資站",
        "en": "To agri-supply",
    },
    "consent_research": {
        "zh": "给农技研究", "yue": "畀農技研究", "hak": "分農技研究",
        "teochew": "分農技研究", "en": "To research",
    },
    "consent_note": {
        "zh": "授权可以随时取消，不授权也能正常用",
        "yue": "授權隨時可以取消，唔授權都照用得",
        "hak": "授權隨時會當取消，毋授權也做得用",
        "teochew": "授權隨時會當取消，毋授權也會用",
        "en": "Consent can be revoked anytime. The app works without it.",
    },
    "confirm_delete": {
        "zh": "确定删除这条记录？", "yue": "確定刪除呢條紀錄？",
        "hak": "確定刪除這條紀錄？", "teochew": "確定刪除這條記錄？",
        "en": "Delete this record?",
    },
    "no_image": {
        "zh": "没有留照片", "yue": "冇留相片", "hak": "無留相片", "teochew": "無留相片",
        "en": "No photo kept",
    },
    "yes": {
        "zh": "确定", "yue": "確定", "hak": "確定", "teochew": "確定", "en": "OK",
    },
    "no": {
        "zh": "取消", "yue": "取消", "hak": "取消", "teochew": "取消", "en": "Cancel",
    },

    # ---- 导出与对接 ----
    "export_purpose": {
        "zh": "用途", "yue": "用途", "hak": "用途", "teochew": "用途", "en": "Purpose",
    },
    "purpose_statistics": {
        "zh": "统计分析", "yue": "統計分析", "hak": "統計分析",
        "teochew": "統計分析", "en": "Statistics",
    },
    "purpose_insurance": {
        "zh": "保险理赔", "yue": "保險理賠", "hak": "保險理賠",
        "teochew": "保險理賠", "en": "Insurance claim",
    },
    "purpose_subsidy": {
        "zh": "补贴申报", "yue": "補貼申報", "hak": "補貼申報",
        "teochew": "補貼申報", "en": "Subsidy filing",
    },
    "purpose_agri": {
        "zh": "农资备货", "yue": "農資備貨", "hak": "農資備貨",
        "teochew": "農資備貨", "en": "Agri-supply stocking",
    },
    "export_format": {
        "zh": "格式", "yue": "格式", "hak": "格式", "teochew": "格式", "en": "Format",
    },
    "export_both": {
        "zh": "两种都要", "yue": "兩種都要", "hak": "兩樣愛", "teochew": "兩樣愛",
        "en": "Both",
    },
    "export_channel": {
        "zh": "送到哪里", "yue": "送去邊度", "hak": "送到哪位",
        "teochew": "送到塊底", "en": "Destination",
    },
    "channel_local": {
        "zh": "留在本机", "yue": "留喺呢部機", "hak": "留在這機",
        "teochew": "留佇這機", "en": "Keep on device",
    },
    "channel_bluetooth": {
        "zh": "蓝牙发送", "yue": "藍牙發送", "hak": "藍牙發送",
        "teochew": "藍牙發送", "en": "Bluetooth",
    },
    "export_anonymize": {
        "zh": "隐去姓名电话", "yue": "隱去姓名電話", "hak": "隱去姓名電話",
        "teochew": "隱去姓名電話", "en": "Hide name and phone",
    },
    "check_integrity": {
        "zh": "校验数据", "yue": "校驗資料", "hak": "校驗資料",
        "teochew": "校驗資料", "en": "Verify data",
    },
    "integrity_ok": {
        "zh": "数据完整，没有被改过", "yue": "資料完整，冇畀人改過",
        "hak": "資料完整，無被人改過", "teochew": "資料完整，無被人改過",
        "en": "Data intact, not tampered with",
    },
    "integrity_bad": {
        "zh": "有记录校验不通过，请联系农技站",
        "yue": "有紀錄校驗唔通過，請聯絡農技站",
        "hak": "有紀錄校驗毋過，請聯絡農技站",
        "teochew": "有記錄校驗袂過，請聯絡農技站",
        "en": "Some records failed verification. Contact the station.",
    },
    "usb_found": {
        "zh": "找到 U 盘：{n}", "yue": "搵到 USB：{n}", "hak": "尋到 U 盤：{n}",
        "teochew": "尋著 U 盤：{n}", "en": "USB drives found: {n}",
    },
    "usb_none": {
        "zh": "没有插 U 盘", "yue": "未有挿 USB", "hak": "無挿 U 盤",
        "teochew": "無挿 U 盤", "en": "No USB drive",
    },
    "export_failed": {
        "zh": "导出没成功", "yue": "匯出冇成功", "hak": "匯出無成功",
        "teochew": "匯出無成功", "en": "Export failed",
    },
    "download_now": {
        "zh": "下载这个文件", "yue": "下載呢個檔案", "hak": "下載這個檔案",
        "teochew": "下載這個檔案", "en": "Download file",
    },
    "data_links": {
        "zh": "数据与对接", "yue": "資料同對接", "hak": "資料同對接",
        "teochew": "資料佮對接", "en": "Data and links",
    },
    "view_adapters": {
        "zh": "对接通道", "yue": "對接通道", "hak": "對接通道",
        "teochew": "對接通道", "en": "Channels",
    },
    "view_schemas": {
        "zh": "数据格式", "yue": "資料格式", "hak": "資料格式",
        "teochew": "資料格式", "en": "Data formats",
    },

    # ---- 设置页补充 ----
    "farmer_profile": {
        "zh": "农户信息（可留空）", "yue": "農戶資料（可以留空）",
        "hak": "農戶資料（可以留空）", "teochew": "農戶資料（會使留空）",
        "en": "Farmer info (optional)",
    },
    "profile_alias": {
        "zh": "称呼", "yue": "稱呼", "hak": "稱呼", "teochew": "稱呼", "en": "Name",
    },
    "profile_farmer_id": {
        "zh": "编号", "yue": "編號", "hak": "編號", "teochew": "編號", "en": "ID",
    },
    "profile_village": {
        "zh": "村", "yue": "村", "hak": "村", "teochew": "村", "en": "Village",
    },
    "profile_town": {
        "zh": "镇", "yue": "鎮", "hak": "鎮", "teochew": "鎮", "en": "Town",
    },
    "profile_county": {
        "zh": "县", "yue": "縣", "hak": "縣", "teochew": "縣", "en": "County",
    },
    "profile_plot": {
        "zh": "地块", "yue": "地塊", "hak": "地塊", "teochew": "地塊", "en": "Plot",
    },
    "speed_slow": {
        "zh": "慢", "yue": "慢", "hak": "慢", "teochew": "慢", "en": "Slow",
    },
    "speed_normal": {
        "zh": "正常", "yue": "正常", "hak": "正常", "teochew": "正常", "en": "Normal",
    },
    "speed_fast": {
        "zh": "快", "yue": "快", "hak": "快", "teochew": "快", "en": "Fast",
    },
    "voice_test": {
        "zh": "试听一句", "yue": "試聽一句", "hak": "試聽一句",
        "teochew": "試聽一句", "en": "Test voice",
    },
    "voice_engine": {
        "zh": "语音引擎", "yue": "語音引擎", "hak": "語音引擎",
        "teochew": "語音引擎", "en": "Voice engine",
    },
    "voice_pack": {
        "zh": "预置语音包", "yue": "預置語音包", "hak": "預置語音包",
        "teochew": "預置語音包", "en": "Bundled voice pack",
    },
    "voice_ready_langs": {
        "zh": "能发声的语言", "yue": "能發聲嘅語言", "hak": "能發聲个語言",
        "teochew": "能發聲个語言", "en": "Voiced languages",
    },
    "voice_missing_langs": {
        "zh": "还缺真人录音的语言", "yue": "還欠真人錄音嘅語言",
        "hak": "還欠真人錄音个語言", "teochew": "還欠真人錄音个語言",
        "en": "Needs native recordings",
    },
    "review_needed_badge": {
        "zh": "待校订", "yue": "待校訂", "hak": "待校訂", "teochew": "待校訂",
        "en": "Review",
    },
    "install_app": {
        "zh": "装到手机桌面", "yue": "裝到手機桌面", "hak": "裝到手機桌面",
        "teochew": "裝到手機桌面", "en": "Install to home screen",
    },
    "about_title": {
        "zh": "关于", "yue": "關於", "hak": "關於", "teochew": "關於", "en": "About",
    },

    # ---- 验收指标 ----
    "rail_status": {
        "zh": "运行状态", "yue": "運行狀態", "hak": "運行狀態",
        "teochew": "運行狀態", "en": "Status",
    },
    "rail_budget": {
        "zh": "验收指标", "yue": "驗收指標", "hak": "驗收指標",
        "teochew": "驗收指標", "en": "Acceptance",
    },
    "rail_voice": {
        "zh": "语音", "yue": "語音", "hak": "語音", "teochew": "語音", "en": "Voice",
    },
    "budget_pass": {
        "zh": "达标", "yue": "達標", "hak": "達標", "teochew": "達標", "en": "Pass",
    },
    "budget_fail": {
        "zh": "未达标", "yue": "未達標", "hak": "未達標", "teochew": "未達標",
        "en": "Fail",
    },
    "metric_size": {
        "zh": "模型体积", "yue": "模型體積", "hak": "模型體積",
        "teochew": "模型體積", "en": "Model size",
    },
    "metric_latency": {
        "zh": "识别耗时", "yue": "識別耗時", "hak": "識別耗時",
        "teochew": "識別耗時", "en": "Latency",
    },
    "metric_memory": {
        "zh": "运行内存", "yue": "運行記憶體", "hak": "運行記憶體",
        "teochew": "運行記憶體", "en": "Memory",
    },
    "metric_accuracy": {
        "zh": "量化精度损失", "yue": "量化精度損失", "hak": "量化精度損失",
        "teochew": "量化精度損失", "en": "Quantization loss",
    },

    # ---- 通用 ----
    "close": {
        "zh": "关闭", "yue": "關閉", "hak": "關閉", "teochew": "關閉", "en": "Close",
    },
    "loading": {
        "zh": "正在加载", "yue": "載入緊", "hak": "載入緊", "teochew": "載入中",
        "en": "Loading",
    },
    "retry": {
        "zh": "再试一次", "yue": "再試一次", "hak": "再試一次",
        "teochew": "再試一次", "en": "Try again",
    },
    "records_saved_n": {
        "zh": "本机已存 {n} 条", "yue": "呢部機已存 {n} 條",
        "hak": "這機已存 {n} 條", "teochew": "這機已存 {n} 條",
        "en": "{n} saved on device",
    },
    "model_ready": {
        "zh": "模型已就绪", "yue": "模型已就緒", "hak": "模型已就緒",
        "teochew": "模型已就緒", "en": "Model ready",
    },
    "backend": {
        "zh": "推理后端", "yue": "推理後端", "hak": "推理後端",
        "teochew": "推理後端", "en": "Inference backend",
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
