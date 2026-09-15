# 运维手册

面向县农技站信息员、试点部署人员和后续接手的开发者。
需求对照见 [requirements-traceability.md](requirements-traceability.md)。

## 1. 环境

| 场景 | 依赖 | 说明 |
| --- | --- | --- |
| 开发 / 训练 | `pip install -r requirements.txt` | 含 torch、torchvision、onnx、onnxruntime、flask、pytest、psutil |
| 边缘设备（只跑识别） | `pip install -r requirements-edge.txt` | 仅 numpy + pillow。缺 onnxruntime 时自动退回纯 NumPy 解释器，能跑但慢 |

本机开发环境实测：Python 3.14.3，Windows + PowerShell 5.1。

> PowerShell 5.1 控制台默认 GBK，打印中文/JSON 前先执行
> `[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $env:PYTHONIOENCODING="utf-8"`，
> 否则 `python -c "..."` 里带中文会抛 `UnicodeEncodeError`。

## 2. 目录约定

所有运行期产物都在一个根目录下，默认 `./artifacts`，可用环境变量 `HEYAN_HOME` 整体重定向
（这是 SD 卡 / U 盘部署的关键开关，见 `heyan/config.py::Paths`）。

```text
<HEYAN_HOME>/
  bundles/     模型包（识别所需的一切：onnx、labels、advisory、语音、基准报告）
  records/     heyan.db，识别记录库（含哈希链）
  data/        数据集（demo_dataset / field）
  exports/     导出结果
  outbox/      待外发的对接文件（保险/补贴/农资）
  voicepacks/  独立语音包（不在 bundle 里时用）
  dist/        发行 zip 与 sha256 校验单
```

仓库自带的模型包在 `artifacts/bundles/heyan-mnv3s-int8-v1.1.0/`（`serve` 自动挑最新的一个；
上一版 `v1.0.0` 仍留在原地，方便 `--bundle` 指名对比）：

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| `model_int8.onnx` | 1.74MB | **部署文件**，20MB 预算按它判定 |
| `model_fp32.onnx` | 5.87MB | 训练对照，不上设备 |
| `model_portable.hgraph.npz` | 5.45MB | 无 onnxruntime 时的退路 |
| `labels.json` | — | 模型输出下标 ↔ 类别 id（训练字母序，与展示序不同） |
| `advisory.json` | — | 每类的严重程度分级与处置建议文案 |
| `manifest.json` | — | 构建参数、压缩比、预算判定结果、完整指标 |
| `benchmark.json` / `benchmark.txt` | — | 最近一次基准报告 |
| `build_report.json` | — | 各阶段完整指标，含田间/合成两份遗忘体检 |
| `voicepack/` | ~45MB | 404 条预渲染语音（zh 202 + en 202） |

## 3. 日常运行

最朴素的启动方式，一个终端一条命令：

```powershell
python -m heyan.cli serve --port 8080
```

正常输出：

```text
[heyan] 界面地址 http://127.0.0.1:8080
[heyan] 模型包 E:\mywork\AgricultureSoftware\artifacts\bundles\heyan-mnv3s-int8-v1.1.0
 * Running on http://127.0.0.1:8080
Press CTRL+C to quit
```

`Ctrl+C` 停止。换端口就改 `--port`；要局域网内其他设备访问，加 `--host 0.0.0.0`
（注意这会暴露识别记录接口，只在可信内网用）。

> 默认端口 8765 在 Windows 上常被输入法占用（`WinError 10013`），`serve` 会自动往后找空闲端口，
> **以终端打印的地址为准**。

不起服务只识别一张图：

```powershell
python -m heyan.cli recognize path\to\leaf.jpg --brief
python -m heyan.cli recognize path\to\leaf.jpg --backend numpy --lang yue
```

自检三件套：

```powershell
python -m heyan.cli info --probe      # 模型包元信息 + 引擎能否加载
python -m heyan.cli benchmark         # 复测体积/延迟/内存，越线非零退出
```

再加一条浏览器/命令行请求：`GET /api/integrity` 看记录库哈希链是否完好。

## 4. 模型包分发（SD 卡 / U 盘部署）

打包并生成校验单：

```powershell
python tools\package_bundle.py
# -> artifacts/dist/heyan-mnv3s-int8-v1.1.0.zip（约 36.6MB）
# -> artifacts/dist/heyan-mnv3s-int8-v1.1.0.zip.sha256
```

拷到目标设备后先校验再解压：

```powershell
python tools\package_bundle.py --verify D:\heyan-mnv3s-int8-v1.1.0.zip
```

校验通过再解压到目标设备的 `<HEYAN_HOME>/bundles/` 下，然后设 `HEYAN_HOME` 指向该目录即可，
例如把整套跑在 U 盘上：

```powershell
$env:HEYAN_HOME = "E:\heyan"
python -m heyan.cli serve --port 8080
```

按设备实际情况复判预算（文档要求 RAM ≤ 2GB、成本 ≤ 300 元）：

```powershell
python -m heyan.cli benchmark --device-ram-mb 2048 --device-cost-cny 300 --images D:\field\*.jpg
```

## 5. 用真实田间照片重训

v1.1.0 已经这么做过一轮：82 张玉米田间实拍（健康 30 / 锈病 25 / 大斑病 27）进来，
类别从 14 扩到 17，田间验证 top1 从 0.0 提到 0.75，老类别不降反升。下面按样本量分两条路。

### 5.1 小样本路线（每类几张到几十张，推荐）

第一步，把照片按采集目录丢进来，体检 + 导入：

```powershell
python tools\ingest_field_samples.py --image-folder dateBase_Maize --out artifacts\data\field
```

目录名不必等于 `class_id`：`heyan/assets/label_aliases.json` 负责翻译
（`Maize_RustDisease` → `maize_rust`，中文别名「玉米锈病」同样命中）。解析前会归一化
大小写与分隔符。**认不出的目录当场报错**，不静默跳过 —— 小样本下少一个目录就是少一整类。
新作物或新采集批次只改这个 json，不动代码。

体检会报：能否解码、通道数、分辨率分布、每类样本量、标签合法性、重复文件
（同一张照片被贴两个标签是最隐蔽的错误）。

第二步，冻骨干只训分类头，合成 + 真实混合回放，QAT 量化，打新版包：

```powershell
python tools\finetune_field.py --dry-run            # 先看数据体检与热启动对齐报告
python tools\finetune_field.py --version 1.2.0      # 全流程，产出新 bundle 替换旧包
```

这条路刻意不跑完整流水线：真实照片每类只有二三十张，全量微调必然过拟合。所以
`linear_probe` 冻住 MobileNetV3 主干只训分类头，真实照片走完整增强（随机旋转/翻转/亮度/
色温/遮挡/JPEG），并把合成回放集按 `class_id` 对齐拼进同一个训练集防灾难性遗忘。
训完自动做两份遗忘体检：老类别在合成回放上、新类别在田间实拍上，训前训后各测一次，
老类别掉超过 `--forget-tolerance`（默认 0.02）就报警。

### 5.2 全量路线（样本充足时）

每类上百张以后才值得走完整的「蒸馏 → 剪枝 → QAT」。先备 `labels.csv`，两列，UTF-8：

```csv
filename,class_id
IMG_0001.jpg,rice_blast
IMG_0002.jpg,peanut_leaf_spot
```

`class_id` 必须是 `heyan/assets/taxonomy.json` 里的 17 个之一。拍糊了、不是作物的照片
统一标 `unusable`——这是兜底类，宁可让它多学一点，也不要硬塞进病害类。

```powershell
python -m heyan.cli ingest --labels-csv D:\field\labels.csv --image-root D:\field\photos --min-per-class 10
python -m heyan.cli build --data-dir artifacts\data\field --labels-csv artifacts\data\field\labels.csv --image-root artifacts\data\field\images --voice-langs zh,en --pack-zip
```

常用开关：

| 开关 | 作用 |
| --- | --- |
| `--strategy head` | 只训分类头，最快，样本极少时用 |
| `--strategy partial` | 默认，解冻后 4 个 block（`--unfreeze-blocks`） |
| `--no-qat` | 退回纯 PTQ。**掉点会远超 3%**，只用于对比实验，不要用于交付 |
| `--qat-epochs` | QAT 轮数，默认 6 |
| `--per-tensor` | 逐张量量化，更快但精度略降 |
| `--no-prune` / `--no-distill` | 关掉对应阶段做消融 |

构建产物落在 `artifacts/bundles/<model-id>-<version>/`，`serve` 会自动挑最新的一个。
流水线任一步越预算（20MB / 3s / 200MB / 掉点 3%），`build` 会在汇总里明确报出来。

## 6. 方言录音工作流

粤语、客家话、潮汕话目前没有语音：本机 SAPI 没有这三种发音人，
`voicepack/index.json` 的 `unvoiced_languages` 如实记着，运行时退回普通话播报并标 `degraded`。

补齐**不需要改代码**，投放同名 wav 即可。运行期解析顺序（`heyan/tts/voicepack.py`）：

1. `recordings/<lang>/<slug>.wav` —— 真人录音，优先级最高
2. `<lang>/<slug>.wav` —— 预渲染合成音
3. 上面两步按语言回退链重试（潮汕话 → 粤语 → 普通话）
4. 拿不到 slug 时按文本反查 `index.json` 再走 1~3

语音包根目录按顺序查找：`<bundle>/voicepack` → CLI 指定的额外根 →
`<HEYAN_HOME>/voicepacks` → `heyan/assets/voicepacks`。

组织一次录音的步骤：

1. 取 slug 清单：打开 `artifacts/bundles/heyan-mnv3s-int8-v1.1.0/voicepack/index.json`，
   每条都有 `slug`、对应文本和时长上限（`VOICE_MAX_SECONDS = 12`）。
2. 按清单录 wav，参数对齐现有语音包：单声道、22050Hz、16-bit。
3. 放到 `<语音包根>/recordings/hak/<slug>.wav`（客家话；潮汕话用 `teochew`，粤语用 `yue`）。
4. 重启 `serve`，`GET /api/voice/status` 里该语言的 `degraded` 应消失。
5. 想让录音随模型包一起分发，把整个语音包目录拷进 bundle 后重新
   `python tools\package_bundle.py`。

新渲染合成音（有对应发音人的语言）：

```powershell
python -m heyan.cli voicepack --langs zh,en --into-bundle
```

客家话与潮汕话的**文案**还需要母语者审校，i18n 里对这两种语言标了 `review_needed`；
审校改 `heyan/i18n.py`，改完跑测试确认 143 键 × 5 语种仍然全覆盖
（`tests/test_core_contract.py::test_i18n_full_coverage`）。

## 7. 导出与授权门控

四个用途 × 四个通道，字段裁剪各不相同：

| 用途 `--purpose` | 面向 | 携带 |
| --- | --- | --- |
| `statistics` | 农技站批量统计 | 聚合与逐条记录，可匿名化 |
| `insurance` | 保险定损 | 出险证据所需字段、置信度、链校验信息 |
| `subsidy` | 补贴发放 | 使用频次、周期、区域态势 |
| `agri_supply` | 农资备货 | 类别与区域需求汇总 |

| 通道 `--channel` | 行为 |
| --- | --- |
| `local` | 写 `<HEYAN_HOME>/exports/` |
| `usb` | 枚举可移动盘写入（`--usb-target` 可指定具体盘符），写完 flush |
| `bluetooth` | 探测适配器并投递 |
| `download` | 生成浏览器下载链接（`GET /api/export/file`） |

```powershell
python -m heyan.cli export --fmt both --purpose insurance --channel usb --since 2026-08-01
python -m heyan.cli export --purpose statistics --anonymize --out artifacts\exports\stats
```

**同意门控是默认打开的**：未获农户授权的记录不导出（`enforce_consent`）。
授权在界面上按条管理（`POST /api/records/<id>/consent`），也可以批量放开口子——
但那样导出的包会在元数据里标明未做授权过滤，别拿去做保险或补贴申报。

## 8. 数据完整性与隐私

每条记录是一份自描述 JSON，带三段哈希：`prev_sha256`（上一条）、`record_sha256`（自己）、
`chain_index`（链上位置）。改一条或抽一条，`GET /api/integrity` 会报 `chain broken`。

删除同样认账：

- `chain_tombstones` —— 每删一条落一枚墓碑，记下被删记录的哈希与时间
- `chain_ledger` —— 写入/删除计数与链头（`adds` / `deleted` / `head_index` / `head_sha256`）

绕过应用直接改数据库文件，账目也会对不上。老库首次使用时会按现状补一条
`baseline_from_existing_db` 基线，不会把历史记录误判成篡改。

隐私要点：

- 照片存库前会降采样（`test_image_downsampled_on_save` 守着），不保留原图分辨率
- 导出可开匿名化，手机号走 `mask_phone` 掩码
- 全程无外发网络请求；`outbox` 只把文件生成到本地，等有网或走 U 盘时再发
- 记录库就是 `records/heyan.db` 一个文件，备份即拷文件

**备份**：先停 `serve`，拷走 `records/heyan.db`（连同 `bundles/` 一起，保证哈希能对上当时的模型版本）。
恢复后跑一次 `GET /api/integrity` 确认链完好。

## 9. 对接发件箱

```powershell
python -m heyan.cli outbox describe     # 三个适配器各自要什么、给什么
python -m heyan.cli outbox schemas      # 写出对接 JSON Schema
python -m heyan.cli outbox status       # 待发条目
python -m heyan.cli outbox dispatch --names insurance,subsidy
python -m heyan.cli outbox flush --target E:\   # 落到 U 盘，人工送达
```

界面上对应 `GET /api/adapters` 与 `POST /api/adapters/dispatch`。

> 这三个适配器是**接口原型**：数据格式、字段裁剪、授权门控、发件箱都实现并可测，
> 但没有对接任何真实保险/补贴/农资系统。真接的时候，改 `heyan/data/integration.py`
> 里对应 `IntegrationAdapter` 子类的传输部分即可，Schema 与门控逻辑不用动。

## 10. 可用性施测

SUS（10 题）+ TAM（PU / PEOU / BI）+ 任务计时，对应文档 §3.3(2) 的测量指标：

```powershell
python -m heyan.cli usability --lang zh --participant P001 --out artifacts\usability
```

结果落在 `artifacts/usability/`，含 SUS 总分、Bangor 形容词分级、百分位提示、
TAM 各维度均值和任务成功率/耗时。**60 人分组对比实验尚未开展**，需要线下组织。

## 11. 故障排查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `serve` 打印的端口不是我要的 | 端口被占（Windows 上 8765 常被输入法占，`WinError 10013`） | 看终端打印的实际地址，或显式 `--port 8080` |
| `UnicodeEncodeError: 'gbk' codec` | PowerShell 控制台 GBK | 先设 `[Console]::OutputEncoding` 与 `$env:PYTHONIOENCODING` 为 UTF-8 |
| 界面能开但识别报错 | onnxruntime 缺失或版本不兼容 | `--backend numpy` 先兜底；正式部署补装 onnxruntime |
| 识别结果全是一类 | 用了合成数据训的模型 | 见第 5 节，用真实照片重训 |
| 播报是普通话但界面选了方言 | 该语言无语音，已 `degraded` 回退 | 见第 6 节补录音 |
| `/api/integrity` 报 `chain broken` | 记录被改或被抽走 | 对比备份，查 `chain_ledger` 的 adds/deleted 与实际条数 |
| `pytest` 起不来，报 TEMP 相关错误 | 系统 TEMP 目录损坏 | `tests/conftest.py` 已把临时目录重定向到 `artifacts/_pytest_home`；仍失败就手动设 `$env:TEMP` |
| 导出到 U 盘失败 | 盘符未识别或只读 | `--usb-target` 显式指定盘符，确认盘可写 |

## 12. 测试

```powershell
python -m pytest tests -q
```

31 项，跑的是仓库里**真实**的模型包和语音包（通过 junction 借进测试沙箱，不污染 `artifacts/`）：

| 文件 | 项数 | 守什么 |
| --- | --- | --- |
| `tests/test_core_contract.py` | 8 | 文档里的硬指标（体积/延迟/内存/掉点）、类别体系、i18n 覆盖、建议文案、配色可区分度 |
| `tests/test_server_api.py` | 12 | 全部 HTTP 接口契约、模型下标来源、路径穿越防护、SPA 回退 |
| `tests/test_store_export.py` | 5 | 哈希链、删除墓碑、回访与授权、导出门控与匿名化、图片降采样 |
| `tests/test_tts.py` | 6 | slug 稳定性、语音命中、方言回退标记、录音覆盖合成音、重拍不播诊断、缺文案时如实报错 |

改了 `heyan/config.py` 里的验收线，`test_core_contract.py` 会立刻红——这是有意的，
硬指标不该被悄悄挪动。
