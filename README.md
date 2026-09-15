# 禾眼 HeYan

面向小农户的**离线**作物胁迫视觉识别系统，按《新工科驱动的小农户视觉识别解决方案》第三章落地：
MobileNetV3-Small 经知识蒸馏、低秩剪枝、INT8 量化感知训练后压到 **1.74MB**，在低端设备本地推理，
界面只有「拍照 → 识别 → 语音播报」三步，全程不联网。

调研区域：广东省粤东西北农业区；作物：水稻、花生、蔬菜、玉米；类别 17 个（含 1 个「画面无法识别」兜底类）。

## 硬指标对照

文档写死的验收线，`python -m heyan.cli benchmark` 可随时复测，越线即失败退出。

| 指标 | 文档要求 | 实测 | 结论 |
| --- | --- | --- | --- |
| 模型体积 | ≤ 20MB | 1.74MB（INT8，实际部署的那一个文件） | 达标，余量 18.3MB |
| 端到端延迟 p95 | ≤ 3s | 7ms | 达标 |
| 运行内存增量 | ≤ 200MB | 15MB | 达标 |
| 目标设备内存 | ≤ 2GB RAM | 已在 2048MB 档位下判定 | 达标 |
| 目标设备成本 | ≤ 300 元 | 已在 300 元档位下判定 | 达标 |
| 量化掉点 | ≤ 3% | 见下方「关于精度」 | 达标 |

> 体积口径说明：包里另有 FP32（5.87MB，仅作训练对照）和便携 npz（5.45MB，无 onnxruntime 时的退路）。
> 一台设备只会装其中一个，所以 20MB 预算按实际部署的 INT8 文件判定，三者合计 13.05MB 单独报在
> `benchmark.json` 的 `all_models_mb` 里，不混为一谈。

### 关于精度

**设备上真正跑的那个 INT8 包**，混合验证集（152 张 = 136 合成 + 16 田间实拍）Top-1 **0.8816**，
量化掉点为负，`≤3%` 的精度合同达标。拆开看：

| 模型 | 田间留出集（16 张，未参与训练） | 老类别合成回放（112 张） | 混合验证集（152 张） |
| --- | --- | --- | --- |
| v1.0.0 微调前 | 0.00（完全认不出玉米） | 0.6875 | — |
| v1.1.0 冻骨干头-only FP32 | 0.75 | 0.8393 | 0.8092 |
| **v1.1.0 部署 INT8（QAT 后）** | **1.00（16/16）** | — | **0.8816** |

老类别不降反升（0.6875 → 0.8393），所以没有灾难性遗忘（容差 0.02）。

**这些数字仍然偏乐观。** 合成图占大头，田间实拍一共只有 82 张（玉米三类，每类 25~31 张），
留出集只有 16 张，一张图就是 6.25 个百分点，16/16 的 95% 置信区间下界也只有约 0.80。
另外，82 张全跑会得到 82/82，但其中 66 张在训练集内，那个数字含记忆成分，**不要拿去汇报**。
它能证明的是：
「冻骨干 + 混合回放 + QAT」这条路在极小样本下走得通、预算守得住、老知识没丢；
不能证明的是水稻/花生/蔬菜在真实田间的精度 —— 那三类至今只有合成数据。

> FP32 侧有两个数，别混：QAT 前的训练权重在混合验证集上是 0.8092，而量化阶段的对照基准
> 是 QAT 之后重导出的 FP32 ONNX（BN 统计已漂移），只有 0.7105。掉点合同按后者算，
> 即 0.7105 → 0.8816。`build_report.json` 的 `summary` 里两个数都列着。

### 用田间照片微调

真实照片按采集目录名丢进去即可，不必手工造 csv：

```powershell
# 1) 体检 + 导入：目录名经 heyan/assets/label_aliases.json 翻译成 class_id
python tools/ingest_field_samples.py --image-folder dateBase_Maize --out artifacts/data/field

# 2) 冻骨干只训分类头，合成+真实混合回放，QAT 量化，打新版包替换旧包
python tools/finetune_field.py --version 1.1.0
```

别名映射固化在 `heyan/assets/label_aliases.json`（`Maize_healthy` → 玉米健康、
`Maize_RustDisease` → 玉米锈病、`Maize_spotDisease` → 玉米大斑病），认不出的目录当场报错，
不静默跳过 —— 小样本下少一个目录就是少一整类。新增作物或新采集批次只改这个文件，不动代码。
`tools/finetune_field.py --dry-run` 可以先只看数据体检与热启动对齐报告。

## 快速开始

```powershell
pip install -r requirements.txt        # 完整环境（含 torch，用于训练）
pip install -r requirements-edge.txt   # 或：边缘设备最小环境（numpy + pillow）

python -m heyan.cli serve --port 8080  # 启动界面，浏览器打开 http://127.0.0.1:8080
```

仓库里已经带了构建好的模型包（`artifacts/bundles/heyan-mnv3s-int8-v1.1.0`），
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

## 让别人也能打开

**同一个局域网**（合作社办公室、田边同一台路由器）：

```powershell
python -m heyan.cli serve --lan --port 8080
```

`--lan` 绑到 0.0.0.0，并打印出 `http://172.20.10.2:8080` 这样真能打开的地址。
手机要用页内实时取景需要 HTTPS：先 `python tools/make_dev_cert.py` 生成自签证书，
再带 `--ssl-cert/--ssl-key`，或直接用 `tools\serve_lan.ps1 -Https -OpenFirewall`。

**公网**（人在外地、跨网络）走内网穿透：

```powershell
python -m heyan.cli tunnel --port 8080
```

它同时起服务和隧道，最后打印一条可直接发出去的分享链接，形如
`https://xxxx.trycloudflare.com/?token=口令`；窗口开着链接才有效，`Ctrl+C` 一起关。
客户端按 cloudflared、ngrok、cpolar 的顺序自动探测：cloudflared 免注册，
把 `cloudflared-windows-amd64.exe` 丢进 `artifacts/bin/` 即可；ngrok 和 cpolar
要先注册拿 authtoken，国内网络通常 cpolar 更稳。`--client` 可以点名。

公网模式下记录库对拿到链接的人完全敞开，所以 `tunnel` 强制三件事：只监听
127.0.0.1、必须有访问口令（不给就自动生成）、开限流与公网加固头。
细节和免费档的如实限制见 `docs/operations.md` 的 3.2 节。演示完就关，别长期挂着。

## 命令一览

| 命令 | 作用 |
| --- | --- |
| `serve` | 启动本地 Web 界面（PWA，可离线缓存）；`--lan` 开放局域网，`--tunnel` 进公网加固模式 |
| `tunnel` | 内网穿透：同时起服务和 cloudflared/ngrok/cpolar，打印可分享的公网链接 |
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

离线预渲染，不依赖网络和系统 TTS。模型包里现有 **404 条**（普通话 202 + 英文 202，
22050Hz WAV，合计约 45MB），由 Windows SAPI 合成。玉米三类是 v1.1.0 新增的，
语音包同步补渲染了 11 条/语种，其余沿用原有文件。

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
  classes.py         17 类作物胁迫体系
  i18n.py            五语种话术表与覆盖度自检
  advice.py          严重程度分级与处置建议
  core/              模型包、推理引擎、预处理、推理后端（onnxruntime / numpy / torch）
  train/             微调、蒸馏、剪枝、量化、QAT、ONNX 导出、流水线
  tts/               语音包、合成器、语言与话术
  data/              记录存储、交换 Schema、导出器、保险/补贴/农资适配器
  eval/              指标、预算基准、SUS+TAM 量表
  server/            Flask 服务与前端（index.html / app.js / style.css / sw.js）
  assets/            taxonomy.json、advisory.json、label_aliases.json
tools/               演示数据集、田间样本导入与微调、应用图标、模型包打包
tests/               契约测试（硬指标、接口、存储导出、语音）
artifacts/           构建产物：模型包、数据集、记录库、导出、语音包
docs/                需求追溯表与运维手册
```

## 测试

```powershell
python -m pytest tests -q
```

32 项，跑的是仓库里真实的模型包和语音包（通过 junction 借进测试沙箱，不污染 `artifacts/`）。
`tests/test_core_contract.py` 专门守文档里那几条硬指标，构建产物一旦越线就红。

## 已知边界

- **田间实拍只覆盖玉米三类共 82 张**，验证集里仅 16 张，一张图就是 6.25 个百分点；
  水稻、花生、蔬菜至今只有合成数据，真实田间精度未经验证。要收紧就得继续按
  上文「用田间照片微调」补样本重训重测。
- **INT8 在 CPU 上比 FP32 慢**（onnxruntime 12.1ms vs 3.5ms，无 int8 算子融合）。
  离 3s 预算远得很，所以照旧交付 INT8 换体积；但别把「量化=更快」当成前提。
- **粤语 / 客家话 / 潮汕话缺语音**，客家话与潮汕话的文案还缺母语者审校。
- **用户对比实验（§3.3）尚未开展**，SUS/TAM 量表工具已就绪（`heyan usability`），
  但 60 名农户的分组实测需要线下组织。
- 保险、补贴、农资三个适配器是**接口原型**：数据格式、字段裁剪、授权门控都已实现并可测，
  但没有对接任何真实业务系统。
- 便携 npz 退路（纯 NumPy 解释器）能跑通，但速度远慢于 onnxruntime，只作无运行时环境的兜底。

## 许可

按文档 §4.1「开放工程」的要求，模型、代码、数据集与文档均以开放方式发布。
