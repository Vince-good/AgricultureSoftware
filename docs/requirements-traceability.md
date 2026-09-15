# 需求追溯表

把《新工科驱动的小农户视觉识别解决方案》里的每一条要求，对到代码位置、验证方式和当前状态。

**状态图例**

| 标记 | 含义 |
| --- | --- |
| ✅ | 已实现，且有自动化测试或可复测的产物守着 |
| 🟡 | 已实现，但验证依赖尚未拿到的真实数据 / 真人参与 |
| ⬜ | 工具或接口已就绪，实际工作尚未开展 |
| ➖ | 超出软件工程范围（属于文档里的调研/实验工作） |

> 一句提醒：v1.1.0 之前，仓库里所有精度数字都跑在合成图上。v1.1.0 起混进了 82 张
> **真实田间实拍**（玉米健康 30 / 锈病 25 / 大斑病 27），但合成图仍占大头
> （680 张 = 17 类 × 40）。所以精度数字只证明流水线通、预算守住、老知识没丢，
> **仍然不代表水稻/花生/蔬菜的田间准确率**——那三类至今没有一张实拍。
> 凡是标 🟡 的行，都卡在"缺更多真实田间照片"这一件事上。

## 一、硬指标（研究问题 2、§3.2(2)(4)）

这几条是文档写死的验收线，常量集中在 `heyan/config.py`，改验收线只改那里。

| 文档要求 | 常量 | 实现 | 验证 | 实测 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 量化后模型体积 ≤ 20MB | `MODEL_SIZE_MAX_MB` | `heyan/core/bundle.py::model_size_mb`（按**实际部署**的那个文件算：INT8 优先） | `tests/test_core_contract.py::test_model_size_budget` | 1.74MB | ✅ |
| 单张端到端延迟 ≤ 3s | `LATENCY_MAX_S` | `heyan/core/engine.py`（预处理+推理+分级+文案全链路计时） | `test_latency_budget` | p95 7ms | ✅ |
| 运行内存增量 ≤ 200MB | `RUNTIME_MEMORY_MAX_MB` | `heyan/eval/benchmark.py`（RSS 增量） | `test_memory_budget` | +14.8MB | ✅ |
| 目标设备 RAM ≤ 2GB | `DEVICE_RAM_MAX_MB` | `benchmark.py` 的设备画像判定 | `benchmark.json` 的 `device_profile` | 2048MB 档位达标 | ✅ |
| 目标设备成本 ≤ 300 元 | `DEVICE_COST_MAX_CNY` | 同上 | 同上 | 300 元档位达标 | ✅ |
| 量化掉点 ≤ 3% | `ACCURACY_DROP_MAX` | `heyan/train/qat.py`（激活定标走 `QUANT_CALIBRATION_PERCENTILE = 99.5` 百分位截断 + QAT；MinMax 定标被离群激活撑爆网格，掉 15~47 个点） | `test_quantization_accuracy_drop` | −17.11 个点（FP32 0.7105 → INT8 0.8816，反超） | ✅ |

> 量化掉点的分母是**混合验证集**（152 张 = 136 合成 + 16 田间实拍），所以这条 ✅ 只说明
> 量化本身没有毁掉精度，不等于田间准确率达标。百分位取 99.5 是扫出来的不是拍的：
> 99.5 在三档采样预算下都收敛到 0.783，99.9 在 0.69~0.75 之间随采样噪声乱跳，
> 99.95 采样越多掉点越大。详见 `heyan/config.py` 与 `heyan/train/qat.py` 的注释。

> 体积口径：包里三个模型文件（INT8 1.74MB / FP32 5.87MB / 便携 npz 5.45MB）**一台设备只装一个**，
> 20MB 预算按实际部署的 INT8 判定；三者合计 13.05MB 单独报在 `benchmark.json` 的 `all_models_mb`。

复测命令：`python -m heyan.cli benchmark`（越线即非零退出）。

## 二、系统四层架构（§3.1）

| 层级 | 文档要求 | 对应痛点 | 实现 | 状态 |
| --- | --- | --- | --- | --- |
| 感知层 | 手机/低成本摄像头采集作物图像 | 痛点一 | `heyan/server/static/app.js`（getUserMedia 全屏取景、相册选择、摄像头翻转）；`heyan/core/preprocess.py`（短边 256 → 中心裁 224 → 归一化） | ✅ |
| 推理层 | 轻量化模型边缘本地推理，不依赖网络 | 痛点二 | `heyan/core/engine.py` + `heyan/core/backends/`（`onnxruntime` 主路径，`numpy_graph.py` 纯 NumPy 退路，`torch_backend.py` 调试用）；`sw.js` 离线缓存，全程无外网请求 | ✅ |
| 交互层 | 语音播报 + 极简图形界面，无需文字阅读能力 | 痛点三 | `heyan/server/static/index.html`（拍照→识别→听结果三步）、`heyan/tts/`（预渲染语音包）、`heyan/i18n.py`（五语种话术） | ✅ |
| 推广层 | 与保险、补贴、农资系统联动 | 痛点四 | `heyan/data/integration.py`（三个适配器 + 发件箱）、`heyan/data/export.py`（四种通道）、`heyan/data/schema.py`（JSON Schema + 哈希链） | 🟡 接口原型，未对接真实业务系统 |

## 三、模型选型与压缩路径（§3.2）

| 文档条目 | 实现 | 状态 |
| --- | --- | --- |
| (1) 选型 MobileNetV3（精度—速度—体积三维均衡） | `heyan/train/model.py`，`config.DEFAULT_ARCH = mobilenet_v3_small`，教师 `mobilenet_v3_large` | ✅ |
| (2) 知识蒸馏（Hinton 2015） | `heyan/train/distill.py`，T=4.0、α=0.5，见 `manifest.json` 的 `extra.distillation` | ✅ |
| (2) 网络剪枝 | `heyan/train/prune.py`，低秩分解，能量阈 0.85，参数 1,532,206 → 1,333,318（−13.0%），剪后 recover 3 轮 | ✅ |
| (2) 8-bit 量化，精度损失 ≤ 3% | `heyan/train/quantize.py` + `heyan/train/qat.py`，权重 per-channel qint8 / 激活 per-tensor quint8，QAT 6 轮，压缩比 3.23× | ✅（数值待真实数据复核） |
| (3) 小样本：预训练模型微调 | `heyan/train/finetune.py`，`--strategy full/partial/head`，`tools/download_weights.py` 拉 ImageNet 预训练权重 | ✅ |
| (3) 小样本：参数高效微调 | 同上，`partial` 只解冻后 4 个 block（`--unfreeze-blocks`），骨干 lr 3e-4 / 分类头 lr 3e-3 分组 | ✅ |
| (3) 小样本：数据增强 | `heyan/train/augment.py`（旋转、裁剪、色彩抖动，模拟田间不均匀光照） | ✅ |
| (3) 田间背景杂乱、光照不均带来的泛化风险 | `tools/ingest_field_samples.py`（目录名经 `label_aliases.json` 翻译成 class_id，认不出的当场报错）+ `tools/finetune_field.py`（冻骨干只训分类头、合成+真实混合回放防灾难性遗忘、训前训后两份遗忘体检） | 🟡 玉米三类 82 张已入库并重训（田间 top1 0.0 → 0.75，老类别 0.6875 → 0.8393）；水稻/花生/蔬菜仍无实拍 |
| (4) 体积 ≤ 20MB、内存 ≤ 200MB | 见第一节 | ✅ |
| (4) 具体性能待真实设备实测 | `benchmark.py` 支持 `--device-ram-mb` / `--device-cost-cny` / `--images` 指定真机与真图 | ⬜ 需在目标设备上跑 |

**模型选择遵循文档结论**：MobileNetV3-Small 学生 + MobileNetV3-Large 教师，
经「微调 → 蒸馏 → 低秩剪枝 → QAT → INT8 量化 → ONNX 导出」压到 1.74MB。
v1.1.0 的田间微调走的是更保守的一档：`linear_probe` 冻住整个主干只训分类头，
从上一版**未剪枝**检查点热启动，fc 层按 `class_id` 对齐行号扩容，新增类别保留随机初始化。
文档 §3.2(1) 提到 ShuffleNet 与 EfficientNet-Lite 作为备选，本项目未实现这两条分支——
选型依据是文档自己的结论（"MobileNetV3 在三维权衡下较为均衡"）。

## 四、适老化交互（§3.3）

| 文档要求 | 实现 | 验证 | 状态 |
| --- | --- | --- | --- |
| 极简三步：打开 → 拍照 → 自动识别并语音播报 | `heyan/server/static/index.html` + `app.js`，首屏即取景器，一个大快门键 | `tests/test_server_api.py::test_static_shell`、`test_bootstrap_contract` | ✅ |
| 播报内容含病害名称、严重程度、处理建议 | `heyan/advice.py` + `heyan/assets/advisory.json`，`heyan/core/severity.py` 分级 | `test_advice_voice_texts_exist_for_every_class` | ✅ |
| 多语言语音：普通话、粤语、潮汕话、客家话 | `heyan/i18n.py`（zh/yue/hak/teochew/en，143 键 × 5 语种全覆盖）、`heyan/tts/languages.py` | `test_i18n_full_coverage` | ✅ 文案 / 🟡 语音 |
| 离线 TTS，语音资源预置在安装包内 | `heyan/tts/voicepack.py`（预渲染 wav）、`heyan/tts/synth.py`（运行时解析），404 条随模型包分发（zh 202 + en 202；v1.1.0 新增玉米三类时补渲染 11 条/语种，其余复用） | `tests/test_tts.py` 6 项 | ✅ |
| 视觉辅助：大字体、高对比度、图标为主 | `heyan/server/static/style.css`，`UI_MIN_FONT_PX=20`、`UI_TOUCH_TARGET_PX=72` | 常量在 `config.py`，构建时可查 | ✅ |
| 低置信度不硬下结论 | `MIN_CONFIDENCE_FOR_VERDICT=0.45`，低于阈值改播「看不清楚，请靠近一点再拍一次」 | `test_tts.py::test_retake_never_speaks_diagnosis` | ✅ |
| 用户对比实验（60 人、实验组/对照组、5 种胁迫图像） | `heyan/eval/usability.py`：SUS 评分 + Bangor 形容词分级 + TAM（PU/PEOU/BI）+ 任务计时 | `heyan usability` CLI | ⬜ 量表工具就绪，实验未开展（需线下组织） |

**已知降级**：粤语 / 客家话 / 潮汕话**没有语音**（本机 SAPI 无这三种发音人）。
`voicepack/index.json` 的 `unvoiced_languages` 如实记录，运行时退回普通话并标 `degraded`，界面同步提示。
补齐方式是真人录音，见 [operations.md](operations.md) 的「方言录音工作流」。
客家话与潮汕话的**文案**也还需母语者审校，i18n 里标了 `review_needed`。

## 五、系统集成与推广机制（§3.4）

| 文档要求 | 实现 | 验证 | 状态 |
| --- | --- | --- | --- |
| (1) 标准化数据接口（JSON Schema） | `heyan/data/schema.py`：记录 Schema + 内置校验器（不依赖 jsonschema 库），`heyan schema` 写出全部 Schema | `test_server_api.py::test_adapters_and_schemas` | ✅ |
| (1) 数据导出（USB 或蓝牙），农技站批量导出 | `heyan/data/export.py`：`local / usb / bluetooth / download` 四通道，`find_usb_targets()` 枚举可移动盘，`detect_bluetooth()` 探测适配器 | `test_export_and_integrity`、`test_export_traversal_guard` | ✅ |
| (1) 模块化：识别核心与交互界面解耦 | `heyan/core/` 不含任何 HTTP 代码，`heyan/server/` 只做传输；后端可插拔（`backends/registry.py`），同一引擎 CLI 与 Web 共用 | `heyan recognize` 与 `/api/recognize` 走同一 `RecognitionEngine` | ✅ |
| (2) 与农业保险联动（识别结果作灾损辅助证据） | `integration.py::InsuranceAdapter`：出险证据包、置信度与链校验字段、理赔所需最小字段集 | `test_adapters_and_schemas` | 🟡 原型，未接真实保险系统 |
| (3) 与补贴政策联动（使用记录作发放依据） | `integration.py::SubsidyAdapter`：按周期聚合使用记录、区域态势汇总（`_period`、`_haversine_km`） | 同上 | 🟡 原型 |
| (4) 与农资供应链联动（**经农户授权**后推荐） | `integration.py::AgriSupplyAdapter` + `SupplierDirectory`；授权门控在 `store.py` 与 `export.py`（`enforce_consent`），未授权记录默认不导出 | `test_store_export.py::test_export_consent_gate_and_anonymize`、`test_followup_and_consent` | 🟡 原型 |
| 数据可信（导出给第三方时能自证未被篡改） | `schema.py` 哈希链（`prev_sha256` / `record_sha256` / `chain_index`）+ `store.py` 删除墓碑（`chain_tombstones`）与写入/删除账本（`chain_ledger`） | `test_store_export.py::test_chain_and_filters`、`test_delete_breaks_chain_honestly`；线上 `GET /api/integrity` | ✅ |

## 六、调研与数据（§2.1–2.3）

| 文档条目 | 落到软件里的部分 | 状态 |
| --- | --- | --- |
| 2.1 调研区域：粤东西北，水稻 / 花生 / 蔬菜 | `heyan/classes.py` + `heyan/assets/taxonomy.json`：17 类（水稻 5 + 花生 4 + 蔬菜 4 + 玉米 3 + 1 个 `unusable` 兜底类），`manifest.json` 的 `region` 字段记「粤东西北」。玉米三类是 v1.1.0 按田间采集批次扩的，文档原表的三种作物仍在 | ✅ |
| 2.1 对象含农技员、保险理赔员、农资站经营者 | 三类角色分别对应导出用途 `statistics` / `insurance` / `agri_supply`，各有独立字段裁剪 | ✅ |
| 2.2 约 100 张田间真实图像样本 | `tools/ingest_field_samples.py`（两种输入布局：`--labels-csv` 或 `--image-folder` 目录名）+ `heyan ingest`，体检覆盖解码/通道数/分辨率分布/每类样本量/标签合法性/重复文件 | 🟡 已入库 82 张（玉米健康 30 / 锈病 25 / 大斑病 27），距文档的 100 张差 18 张，且全部集中在玉米 |
| 2.3 痛点一：识别手段原始、误判率高 | 17 类体系覆盖早期胁迫（氮缺乏、水分胁迫、叶斑、稻瘟、霜霉、玉米锈病与大斑病），`severity.py` 给出严重程度而非二值判断 | ✅ |
| 2.3 痛点二：现有方案要联网、步骤多、看不懂 | 全程离线（PWA + Service Worker），三步流程，结果以语音 + 图标 + 颜色呈现 | ✅ |
| 2.3 痛点三：老年农户数字鸿沟 | 见第四节适老化设计 | ✅ |
| 2.3 痛点四：信息获取被动 | 导出 + 发件箱让农技站/保险/农资方主动拿到结构化数据 | 🟡 |
| 2.3 痛点五：农户要"一拍就告诉我是啥病、该咋办、免费、不用网" | `advice.py` 每个类别都有可播报的处置建议文案；无网络依赖；无授权费用逻辑 | ✅ |

## 七、研究设计与成果共享（§4、§5）

| 文档条目 | 实现 | 状态 |
| --- | --- | --- |
| 4.1 学科整合：计算机科学 | `heyan/core/`、`heyan/train/` | ✅ |
| 4.1 学科整合：嵌入式系统 | `requirements-edge.txt`（仅 numpy + pillow）、便携 npz 退路、`HEYAN_HOME` 可整体重定向到 SD 卡/U 盘、设备画像判定 | ✅ |
| 4.1 学科整合：人机交互与工程心理学 | `heyan/eval/usability.py`（SUS + TAM + 任务计时）、适老化 UI 常量 | ✅ |
| 4.1 学科整合：工程管理与农业经济 | `heyan/data/integration.py` 三个联动适配器 | 🟡 |
| 4.1 「开放工程」：模型、代码、数据集、文档开放发布 | README「许可」章节、`tools/package_bundle.py`（带 sha256 校验单的发行包）、`docs/` | ✅ |
| 4.2 第一阶段 田野调研（已完成） | ➖ 属文档工作，软件侧只承接其产出（作物类别、语言清单、痛点） | ➖ |
| 4.2 第二阶段 技术可行性论证（进行中） | 本仓库把「论证」推进到「实测」：流水线跑通并给出预算实测值 | ✅ |
| 4.2 第三阶段 交互方案设计与验证 | 交互原型已实现；用户验证实验未开展 | ⬜ |
| 4.2 第四阶段 推广机制设计 | 三个适配器 + Schema + 导出通道为接口原型 | 🟡 |
| 4.2 第五阶段 成果整理与共享 | `README.md`、`docs/requirements-traceability.md`、`docs/operations.md`、`artifacts/dist/` 发行包 | ✅ |
| 5 预期贡献：技术路线图 | 本表第三节 + `manifest.json` 里完整可复现的流水线参数 | ✅ |
| 5 预期贡献：交互设计指南（中英双语） | i18n 五语种（含 en）话术表 + 适老化常量；独立成文的《设计指南》尚未撰写 | ⬜ |
| 5 预期贡献：需求画像与调研报告 | ➖ 文档本身 | ➖ |

## 八、诚实清单：没做到的事

1. **精度数字仍以合成数据为主**。680 张程序生成的图（17 类各 40）+ 82 张真实田间实拍，
   而实拍全部集中在玉米三类，验证集里只占 16 张 —— 一张图就是 6.25 个百分点，
   置信区间很宽。水稻、花生、蔬菜至今**没有一张实拍**，那 13 个老类别的田间精度
   完全未经验证。要收紧就得继续按 §5.1 补样本重训重测。
2. **§3.3(2) 的 60 人对比实验没做**。SUS / TAM / 任务计时工具已就绪（`heyan usability`），
   但招募、分组、线下施测不在软件工程范围内。
3. **保险、补贴、农资三个适配器是接口原型**。数据格式、字段裁剪、授权门控、发件箱都实现并可测，
   但没有对接任何真实业务系统，也没有真实的农资站目录数据。
4. **粤语、客家话、潮汕话没有语音**，客家话与潮汕话的文案还缺母语者审校。
5. **未在真实低端设备上实测**。延迟与内存数字跑在开发机上（p95 7ms / +14.8MB），
   目标是 RAM ≤ 2GB 的手机或 ≤ 300 元的开发板，需在真机复测。
6. **便携 npz 退路很慢**。纯 NumPy 解释器能跑通，只作无 onnxruntime 环境的兜底，不作性能路径。
7. **INT8 在 CPU 上比 FP32 慢**：onnxruntime 实测 12.09ms vs 3.52ms（0.29×），
   因为这条路径没有 int8 算子融合。离 3s 预算远得很，所以仍按体积优先交付 INT8，
   但"量化必然更快"在这个后端上不成立，别拿它当性能优化手段。
