# 禾眼 HeYan

面向小农户的**离线**作物胁迫视觉识别系统，按《新工科驱动的小农户视觉识别解决方案》第三章落地：
MobileNetV3-Small 经知识蒸馏、低秩剪枝、INT8 量化感知训练后压到 **1.58MB**，在低端设备本地推理，
界面只有「拍照 → 识别 → 语音播报」三步，全程不联网。

调研区域：广东省粤东西北农业区；作物：水稻、花生、蔬菜；类别 14 个（含 1 个「画面无法识别」兜底类）。

## 硬指标对照

文档写死的验收线，`python -m heyan.cli benchmark` 可随时复测，越线即失败退出。

| 指标 | 文档要求 | 实测 | 结论 |
| --- | --- | --- | --- |
| 模型体积 | ≤ 20MB | 1.58MB（INT8，实际部署的那一个文件） | 达标，余量 18.4MB |
| 端到端延迟 p95 | ≤ 3s | 8ms | 达标 |
| 运行内存增量 | ≤ 200MB | 19MB | 达标 |
| 目标设备内存 | ≤ 2GB RAM | 已在 2048MB 档位下判定 | 达标 |
| 目标设备成本 | ≤ 300 元 | 已在 300 元档位下判定 | 达标 |
| 量化掉点 | ≤ 3% | 见下方「关于精度」 | 达标 |

> 体积口径说明：包里另有 FP32（5.1MB，仅作训练对照）和便携 npz（4.7MB，无 onnxruntime 时的退路）。
> 一台设备只会装其中一个，所以 20MB 预算按实际部署的 INT8 文件判定，三者合计 11.42MB 单独报在
> `benchmark.json` 的 `all_models_mb` 里，不混为一谈。

### 关于精度

验证集 Top-1：FP32 0.5625 → INT8 **0.8036**（量化感知训练反超，掉点为负）。

**这个数字不能当作田间准确率。** 它跑在 `tools/make_demo_dataset.py` 生成的 560 张合成图上
（14 类 × 40 张），只证明「蒸馏 → 剪枝 → QAT → 量化」这条流水线本身是通的、预算是守住的。
真实精度必须用田间照片重训：把照片按 `labels.csv`（两列 `filename,class_id`）整理好，跑
`python -m heyan.cli ingest`，再跑 `build`。文档第 2.2 节那 100 张实拍样本就是这一步的起点。

## 快速开始

```powershell
pip install -r requirements.txt        # 完整环境（含 torch，用于训练）
pip install -r requirements-edge.txt   # 或：边缘设备最小环境（numpy + pillow）

python -m heyan.cli serve --port 8080  # 启动界面，浏览器打开 http://127.0.0.1:8080
```

仓库里已经带了构建好的模型包（`artifacts/bundles/heyan-mnv3s-int8-v1.0.0`），
`serve` 会自动找到它，不需要先训练。

只识别一张图，不起服务：

```powershell
python -m heyan.cli recognize path\to\leaf.jpg --brief
```

从零重训一遍（约十几分钟，CPU）：

```powershell
python -m heyan.cli demo-data                       # 生成合成演示数据集
python -m heyan.cli build --voice-langs zh,en --pack-zip
```

> Windows 上默认端口 8765 常被输入法占用（`WinError 10013`），`serve` 会自动往后找空闲端口，
> 以终端打印的地址为准。

## 命令一览

| 命令 | 作用 |
| --- | --- |
| `serve` | 启动本地 Web 界面（PWA，可离线缓存） |
| `recognize` | 单张图离线识别 |
| `build` | 完整流水线：微调 → 蒸馏 → 剪枝 → QAT 量化 → 打包 → 基准测试 |
| `demo-data` | 生成合成演示数据集 |
| `ingest` | 导入田间照片并做数据体检（每类样本量、损坏文件、标签合法性） |
| `benchmark` | 复测体积 / 延迟 / 内存预算 |
| `voicepack` | 离线预渲染语音包（默认五语种，`--into-bundle` 拷进模型包） |
| `export` | 导出识别记录，支持 JSON / CSV、四种用途、四种通道 |
| `outbox` | 对接发件箱：保险、补贴、农资三个适配器 |
| `schema` | 写出全部 JSON Schema |
| `info` | 查看模型包元信息（`--probe` 顺带加载引擎） |
| `usability` | 跑 SUS + TAM 可用性量表 |

`python -m heyan.cli <命令> -h` 看每项的完整参数。

## 界面

三步流程，对应文档 §3.3：

1. **拍照** —— 全屏取景，一个大快门键，可从相册选，可翻转摄像头。
2. **识别** —— 本地推理，画面上直接标出结果。
3. **听结果** —— 自动播报病害名称、严重程度、处理建议，可「再听一次」。

适老化细节：正文最小 20px，触控目标最小 72px，高对比配色，图标为主文字为辅；
置信度低于 0.45 时不下结论，改为提示「看不清楚，请靠近一点再拍一次」。

另外三个页签：识别记录（含回访与授权管理）、数据导出、设置（语言 / 后端 / 线程）。

## 语音

离线预渲染，不依赖网络和系统 TTS。模型包里现有 **382 条**（普通话 191 + 英文 191，
22050Hz WAV，合计约 41MB），由 Windows SAPI 合成。

**粤语、客家话、潮汕话目前没有语音。** 本机 SAPI 没有这三种发音人，
`voicepack/index.json` 的 `unvoiced_languages` 如实记着这件事，运行时会退回普通话播报并标记
`degraded`，界面也会提示。要补齐，需要真人录音：把 wav 按
`recordings/<lang>/<slug>.wav` 放进语音包目录即可，不用改代码。
语音包根目录按顺序查找：`<bundle>/voicepack` → `<HEYAN_HOME>/voicepacks` → `heyan/assets/voicepacks`；
同一目录内 `recordings/` 下的真人录音优先级高于预渲染合成音，找不到再按
潮汕话 → 粤语 → 普通话回退。`slug` 清单见 `index.json`。
具体步骤见 [docs/operations.md](docs/operations.md) 第 6 节。
客家话和潮汕话的文案本身也还需要母语者审校，
i18n 里对这两种语言标了 `review_needed`。

## 数据与对接

每条识别记录是一份自描述的 JSON，带哈希链：每条记录记 `prev_sha256`、`record_sha256`、
`chain_index`，改一条或抽一条都会在 `GET /api/integrity` 里报出来。
删除同样认账 —— 每删一条落一枚墓碑（`chain_tombstones`），另有一本写入/删除计数与链头账本
（`chain_ledger`），绕过应用直接动数据库文件也会对不上账。这样导出给保险或补贴系统时，
这份数据能自证既没被改也没被抽走。

导出通道：留在本机、USB、蓝牙、浏览器下载。用途分统计分析、保险理赔、补贴申报、农资备货，
不同用途带不同的字段裁剪；匿名化可开可关；**未获农户授权的记录默认不导出**（`enforce_consent`）。

对接适配器与 JSON Schema：`python -m heyan.cli outbox describe`、`python -m heyan.cli schema`。

## 目录结构

```text
heyan/
  config.py          全部硬指标常量与路径（改验收线只改这里）
  classes.py         14 类作物胁迫体系
  i18n.py            五语种话术表与覆盖度自检
  advice.py          严重程度分级与处置建议
  core/              模型包、推理引擎、预处理、推理后端（onnxruntime / numpy / torch）
  train/             微调、蒸馏、剪枝、量化、QAT、ONNX 导出、流水线
  tts/               语音包、合成器、语言与话术
  data/              记录存储、交换 Schema、导出器、保险/补贴/农资适配器
  eval/              指标、预算基准、SUS+TAM 量表
  server/            Flask 服务与前端（index.html / app.js / style.css / sw.js）
  assets/            taxonomy.json、advisory.json
tools/               演示数据集、田间样本导入、应用图标、模型包打包
tests/               契约测试（硬指标、接口、存储导出、语音）
artifacts/           构建产物：模型包、数据集、记录库、导出、语音包
docs/                需求追溯表与运维手册
```

## 测试

```powershell
python -m pytest tests -q
```

31 项，跑的是仓库里真实的模型包和语音包（通过 junction 借进测试沙箱，不污染 `artifacts/`）。
`tests/test_core_contract.py` 专门守文档里那几条硬指标，构建产物一旦越线就红。

## 已知边界

- **精度数字来自合成数据**，田间可用性未经验证，需按上文用实拍照片重训后重测。
- **粤语 / 客家话 / 潮汕话缺语音**，客家话与潮汕话的文案还缺母语者审校。
- **用户对比实验（§3.3）尚未开展**，SUS/TAM 量表工具已就绪（`heyan usability`），
  但 60 名农户的分组实测需要线下组织。
- 保险、补贴、农资三个适配器是**接口原型**：数据格式、字段裁剪、授权门控都已实现并可测，
  但没有对接任何真实业务系统。
- 便携 npz 退路（纯 NumPy 解释器）能跑通，但速度远慢于 onnxruntime，只作无运行时环境的兜底。

## 许可

按文档 §4.1「开放工程」的要求，模型、代码、数据集与文档均以开放方式发布。
