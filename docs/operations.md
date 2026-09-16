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

代码里没有任何写死的盘符路径：所有位置都由「项目根目录 + `HEYAN_HOME`」推导。本文命令中的
`<项目根目录>`、`<照片目录>`、`<U盘盘符>` 都是占位符，按自己的机器替换即可。

这条约束由 `tests/test_no_absolute_paths.py` 守着：它会扫描仓库内所有文本文件，一旦出现
写死的盘符路径（连注释里的示例也算）就让测试变红。新增脚本时若确实需要写绝对路径，
请把该文件加进测试里的 `DRIVE_PATH_ALLOWLIST` 并写明理由。

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
[heyan] 模型包 <项目根目录>\artifacts\bundles\heyan-mnv3s-int8-v1.1.0
 * Running on http://127.0.0.1:8080
Press CTRL+C to quit
```

`Ctrl+C` 停止。换端口就改 `--port`。要让本机之外的设备也能打开，看下面 3.1（局域网）
和 3.2（公网）。别再直接 `--host 0.0.0.0` 裸奔：那会把带姓名/村/地块的识别记录
接口和删除接口一起敞开，且没有任何鉴权。

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

### 3.1 局域网访问

```powershell
python -m heyan.cli serve --lan --port 8080
python -m heyan.cli serve --lan --port 8080 --access-token 自己定一个口令
```

`--lan` 做三件事：绑 0.0.0.0、探测本机局域网 IPv4、打印 `http://172.20.10.2:8080`
这种真能点开的地址（不会打印 0.0.0.0，那个地址打不开）。手机连不上多半是防火墙没放行，
用管理员权限跑一次封装好的脚本：

```powershell
powershell -ExecutionPolicy Bypass -File tools\serve_lan.ps1 -Port 8080 -OpenFirewall
powershell -ExecutionPolicy Bypass -File tools\serve_lan.ps1 -RemoveFirewallRule
```

没有管理员权限时脚本不会失败，它会把该粘的 `netsh` 命令打印出来。

手机浏览器在 http 页面里拿不到摄像头权限（非安全上下文），页内实时取景会退化成
"从相册选图上传"，识别本身照常。要取景就补 HTTPS：

```powershell
python tools\make_dev_cert.py
python -m heyan.cli serve --lan --port 8080 --ssl-cert artifacts\certs\heyan-lan.crt --ssl-key artifacts\certs\heyan-lan.key
```

自签证书每台设备首次访问都要手动信任一次（"高级 - 继续前往"），这是自签的固有代价。
证书目录里自带 `.gitignore`，私钥不会被提交。

### 3.2 公网访问（内网穿透）

田间主机多半挂在 4G 路由或家宽 NAT 后面，没有公网 IP，端口映射无从下手。隧道是从
内网主动往外的长连接，不用动路由器、不用找运营商要公网地址：

```powershell
powershell -ExecutionPolicy Bypass -File tools\share_public.ps1        # 或直接双击 tools\share_public.bat
python -m heyan.cli tunnel --port 8080
python -m heyan.cli tunnel --port 8080 --client cpolar
python -m heyan.cli tunnel --port 8080 --access-token 我的口令 --subdomain heyan-demo
```

`share_public.ps1` 是 `tunnel` 的一键包装：从 8080 起找第一个空闲端口（本机已经开着
`serve` 也不会撞上退出码 7），把 `-Client / -Token / -Region / -Bundle` 透传下去，
`-DryRun` 只打印将执行的命令不真跑。日常演示双击 `.bat` 就够，要固定端口才手敲 `tunnel`。

输出里那条 `https://<域名>/?token=<口令>` 就是能直接发出去的分享链接。窗口必须开着，
`Ctrl+C` 会把服务和隧道一起收掉。口令也可以用环境变量 `HEYAN_ACCESS_TOKEN` 传，
免得留在命令历史里。

三家客户端，自动探测顺序就是下表顺序：

| 客户端 | 要注册吗 | 固定域名 | 备注 |
| --- | --- | --- | --- |
| cloudflared | 不用 | 不能 | 下载 exe 丢进 `artifacts\bin\` 即可，最适合临时演示 |
| ngrok | 要 authtoken | 付费档 | 免费档会给浏览器插一张警告页，前端已带跳过头 |
| cpolar | 要 authtoken | 付费档 | 国内直连质量通常最好 |

客户端装在 PATH 里或直接放进 `artifacts\bin\`（该目录已被 git 忽略）都能被找到。
authtoken 是一次性配置：

```powershell
ngrok config add-authtoken <你的TOKEN>
cpolar authtoken <你的TOKEN>
```

**安全模型**（`tunnel` 强制执行，不给口令就自动生成一个）：

- 服务只监听 127.0.0.1，由隧道客户端从本机连进来，网卡上不额外开口子；
- 没带口令时：浏览器请求返回一张口令页，`/api/*` 返回 401 `auth_required`，
  静态资源返回干净的 401（不会把 HTML 口令页塞给 `sw.js`）；
- 口令三种传法：`?token=`（自动换成 HttpOnly cookie，并把地址里的口令摘掉再 302）、
  `X-HeYan-Token` 头、cookie；cookie 有效期 12 小时；
- 限流只在公网模式生效：识别 12 次/分钟、其余 240 次/分钟，按来源计，超了返回 429
  带 `Retry-After`；
- 加固头：nosniff、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer`、
  `X-Robots-Tag: noindex`，记录与导出接口一律 `no-store`；
- 公网模式下 `/api/health` 只报模型包名不报完整路径，`/api/bootstrap` 不吐本机目录，
  报错也不带 `detail`；
- 隧道那边终止 TLS，服务靠 `X-Forwarded-Proto` 认出外层是 https，
  跳转不会把人甩回 http，cookie 也能正确带上 `Secure`。

**如实告知的限制**：

- 免费档域名每次重启都变，发出去的旧链接随之失效；
- 有带宽和并发上限，几十人同时刷会卡；
- 隧道服务商能看到全部过路流量：HTTPS 只在浏览器到服务商之间加密，服务商到本机
  这一跳是它自己解开的。涉密数据不要走公网；
- 拿到链接的人就能查看和删除本机全部识别记录。别把链接发到公开群里，更别挂着不管，
  演示完 `Ctrl+C`；
- cloudflared 的免费快速隧道没有可用性保证。实测会遇到
  `failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": context deadline exceeded`，
  这时 `tunnel` 打印日志尾部并以退出码 5 结束，而不是发一个认错的链接（分享链接里
  带着口令，认错域名等于把口令送给别人）。重跑一次通常就好。

退出码：0 正常结束，2 参数与客户端能力冲突（如给 cloudflared 指定 `--subdomain`），
3 没找到隧道客户端，4 服务没起来，5 隧道没给出公网地址，6 隧道还活着但服务先退了，
7 端口已被占用（`tunnel` 会自己起服务，撞上还在跑的旧 `serve` 就会把公网地址挂到那个
没开 `ProxyFix`、没做硬化的进程上，所以直接拒绝启动；停掉旧的或换 `--port`）。

日志落在 `artifacts\logs\` 下，一次运行两份：`tunnel-server-*.log` 是服务，
`tunnel-<客户端>-*.log` 是隧道。排查失败原因先看后者。

排查：

| 现象 | 原因与处理 |
| --- | --- |
| 没找到可用的隧道客户端 | 没装或没放进 `artifacts\bin\`，按打印出来的三步指引装一个 |
| 退出码 7，提示端口被占用 | 之前的 `serve`/`tunnel` 还在跑，去那个窗口 `Ctrl+C`，或换 `--port` |
| 退出码 5，日志有 `ERR_NGROK_105` | authtoken 没配，`ngrok config add-authtoken <TOKEN>` |
| 退出码 5，日志有 `failed to request quick Tunnel` | cloudflared 快速隧道超时/限流，重跑或换 cpolar |
| 退出码 4 | 服务本身没起来，看同目录 `tunnel-server-*.log` |
| 链接能打开但一直要口令 | cookie 过期（12 小时）或换了浏览器，用完整分享链接重开 |
| 页面开了但摄像头用不了 | 隧道给的是 HTTPS，本该可用；若走局域网 http 见 3.1 |
| 对方打开先看到一张 ngrok 警告页 | 免费档行为，点 Visit Site 即可，前端已带跳过头 |

### 3.3 云端部署（本机可以关机）

3.1 和 3.2 都有一个共同前提：**这台机器一直开着**。窗口一关、电脑一睡、
家里一断网，链接立刻失效。比赛和正式交付靠不住这个前提，所以还有第三条路：
把整套软件装进容器跑在云端。

```powershell
python tools/make_cloud_bundle.py     # 生成 artifacts/cloud 部署目录，约 59MB
```

部署目录里是代码 + 一个模型包 + `Dockerfile` + HF Spaces 认的 README 头 +
git-lfs 规则 + 一份"不忽略模型"的 `.gitignore`。按白名单拷贝，
`artifacts/records/heyan.db` 这类本机数据不会被带上公网。

容器里的启动入口是 `python -m heyan.server.cloud`，和本机 `serve` 的区别在于：
绑 `0.0.0.0`、端口认平台注入的 `$PORT`、端口被占**直接退出而不顺延**
（平台只健康检查它分配的那一个端口，悄悄换端口等于"日志说起来了、链接永远
打不开"）、一律按公网加固、启动即预热模型。

推到 Hugging Face Spaces（免费、不要信用卡、自带 HTTPS）就能拿到一条
`https://<用户名>-<空间名>.hf.space` 的长期链接。**从零到公网链接的完整步骤、
模型文件的三种上传方式、免费档的休眠与无持久存储限制、故障排查表，
见 `deploy-cloud.md`。**

## 4. 模型包分发（SD 卡 / U 盘部署）

打包并生成校验单：

```powershell
python tools\package_bundle.py
# -> artifacts/dist/heyan-mnv3s-int8-v1.1.0.zip（约 36.6MB）
# -> artifacts/dist/heyan-mnv3s-int8-v1.1.0.zip.sha256
```

拷到目标设备后先校验再解压：

```powershell
python tools\package_bundle.py --verify .\heyan-mnv3s-int8-v1.1.0.zip
```

校验通过再解压到目标设备的 `<HEYAN_HOME>/bundles/` 下，然后设 `HEYAN_HOME` 指向该目录即可，
例如把整套跑在 U 盘上：

```powershell
$env:HEYAN_HOME = "<U盘盘符>:\heyan"
python -m heyan.cli serve --port 8080
```

按设备实际情况复判预算（文档要求 RAM ≤ 2GB、成本 ≤ 300 元）：

```powershell
python -m heyan.cli benchmark --device-ram-mb 2048 --device-cost-cny 300 --images "<照片目录>\*.jpg"
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
python -m heyan.cli ingest --labels-csv "<照片目录>\labels.csv" --image-root "<照片目录>\photos" --min-per-class 10
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
python -m heyan.cli outbox flush --target "<U盘盘符>:\"   # 落到 U 盘，人工送达
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
