# 云端部署手册

把整套软件放到云上，评委点一个链接就能用，你的电脑可以关机。

和 3.2 节的内网穿透不是一回事：穿透是"别人访问的还是你这台机器"，窗口一关、
电脑一睡、家里一断网，链接立刻失效；云端是"代码和模型都跑在服务商的容器里"，
和本机再无关系。比赛这种"我人不在场也得能用"的场景，只有云端这条路。

---

## 1. 选哪个平台

| 平台 | 费用 | 要不要信用卡 | 公网 HTTPS | 适合本项目吗 |
| --- | --- | --- | --- | --- |
| **Hugging Face Spaces（Docker SDK）** | 免费档够用 | 不要 | 自带 | **推荐** |
| Render / Railway / Fly.io | 有免费或低价档 | 多数要绑卡 | 自带 | 可以，配置多几步 |
| 自己的云服务器 | 按机器算 | 要 | 要自己配证书 | 有现成机器就走这条 |

推荐 HF Spaces 的理由很实在：免费、不要信用卡、自带 HTTPS（手机页内实时取景
必须要 HTTPS，见 3.1 节）、推一个 git 仓库就自动构建，而且它本来就是给
"演示一个模型"设计的，评委也熟悉这个域名。

下面的步骤以 HF Spaces 为主线。第 11 节说明换别的平台要改哪几处。

---

## 2. 需要准备什么

1. 一个 Hugging Face 账号（邮箱注册即可）。
2. 本机能跑 `git`，并且装了 **git-lfs**。检查一下：

```powershell
git lfs version    # 没输出就是没装，去 git-lfs.com 下载安装，然后执行下一行
git lfs install
```

3. 一个 HF 的 **Access Token**（写权限）：登录 HF → 右上头像 → Settings →
   Access Tokens → New token，勾选 write。push 的时候用户名填 HF 用户名，
   密码填这个 token。
4. 国内网络访问 HF 常常需要代理。仓库已经为 GitHub 配过 7890 端口的代理，
   HF 要单独配一条（端口以你自己的代理软件为准，在它的设置界面能看到）：

```powershell
git config --global http.https://huggingface.co.proxy http://127.0.0.1:7890
```

---

## 3. 第一步：生成部署目录

仓库根的 `.gitignore` 把 `artifacts/` 整体挡在版本库外——这是对的，模型和语音包
太大，GitHub 推不动。但云端镜像**必须**带着模型包，否则容器起来只会显示
"请先构建模型包"。两边口径天生冲突，所以不要直接推仓库根，而是先生成一份
专门给云端的目录：

```powershell
python tools/make_cloud_bundle.py
```

输出在 `artifacts/cloud/`（约 59MB，509 个文件），内容是：

- 代码：`heyan/`、`tools/`、`tests/`、`docs/`
- 模型包：`artifacts/bundles/` 下最新的那一个，含 INT8 / FP32 模型和全部语音
- 构建配置：`Dockerfile`、`.dockerignore`、`requirements-cloud.txt`
- HF 认的 `README.md`（顶部 YAML 头声明 `sdk: docker`、`app_port: 7860`）
- `.gitattributes`（把 `*.onnx` / `*.wav` 交给 git-lfs）
- 一份**口径相反**的 `.gitignore`：这里模型是要提交的

按白名单拷贝，所以这些**不会**被带上云：`artifacts/records/heyan.db`（你本机
调试攒下的真实识别记录，推上去等于公开给每一个点链接的人）、`artifacts/data`
演示数据集、`dateBase_Maize/` 训练照片、`__pycache__`。

想瘦身的话：

```powershell
python tools/make_cloud_bundle.py --slim              # 去掉 fp32/npz，省约 12MB
python tools/make_cloud_bundle.py --voice-langs zh    # 只带普通话，再省 17MB
python tools/make_cloud_bundle.py --zip               # 顺手压成 artifacts/cloud.zip（约 37MB）
```

`--slim` 省掉的 FP32 和 `.npz` 是"没有 onnxruntime 时"的两条回退路径，云端镜像
一定装了 onnxruntime，所以用不上。裁语言也是安全的：语音索引会跟着裁，
`VoicePack` 本身还有 zh 回退链，不会 404。脚本最后会把包重新读一遍做自检，
不通过就返回非 0，不会让你推一个坏包上去。

---

## 4. 第二步：本地先验证一遍

**强烈建议**在推上去之前先在本地用云端入口跑一次。云端构建一次要几分钟，
拿它来发现"忘了拷文件"这种问题太贵了。

```powershell
cd artifacts/cloud
pip install -r requirements-cloud.txt      # 装上 waitress，走真实的生产路径
$env:HEYAN_PORT='8901'; python -X utf8 -m heyan.server.cloud
```

看到这几行就对了：

```text
[cloud] 监听 0.0.0.0:8901（容器外通过平台分配的公网地址访问）
[cloud] 推理后端 onnxruntime，模型 1.735MB
[cloud] 口令鉴权：未开启 —— 拿到链接的人都能用，适合评委演示
[cloud] waitress 起服务，8 个工作线程
```

再开一个窗口验收：

```powershell
Invoke-WebRequest http://127.0.0.1:8901/api/health -UseBasicParsing | Select-Object -Expand Content
```

要看到 `"ok":true` 且 `"engine":true`。然后浏览器打开 `http://127.0.0.1:8901`，
上传一张玉米照片，确认识别结果、防治建议、语音播报都正常。

装了 Docker 的话，把镜像也构建一遍最稳：

```powershell
docker build -t heyan-cloud .
docker run --rm -p 8901:7860 heyan-cloud
```

> 从部署目录构建，别从仓库根构建。仓库根的 `artifacts/bundles/` 下躺着 v1.0.0
> 和 v1.1.0 两个包，`.dockerignore` 会把它们**都**塞进镜像（多 58MB），
> 而"取最新的一个"靠的是文件修改时间——COPY 进镜像后这个顺序不值得赌。
> 部署目录里只有一个包，没有歧义。

---

## 5. 第三步：在 HF 上建一个 Space

1. 打开 huggingface.co，登录后点右上 **New** → **Space**。
2. **Space name** 填一个好记的英文名，比如 `heyan-crop-doctor`。它会成为公网
   地址的一部分，只能用字母、数字和连字符。
3. **License** 选 MIT（或你喜欢的）。
4. **SDK** 选 **Docker**，模板选 **Blank**。
5. **Space hardware** 保持免费的 CPU basic。
6. Visibility 选 **Public**（免费档只有 Public；Private 要付费）。
7. 点 **Create Space**。

建好后页面上有一个 **Clone repository** 按钮，里面就是你要用的 git 地址，
形如 `https://huggingface.co/spaces/<用户名>/<空间名>`。

---

## 6. 第四步：推上去

```powershell
cd artifacts/cloud
git init -b main
git lfs install
git add .
git lfs ls-files        # 关键一步：确认 model_int8.onnx 和 .wav 都在这个列表里
git commit -m "deploy: 禾眼云端镜像 v1.1.0"
git remote add space https://huggingface.co/spaces/<用户名>/<空间名>
git push space main --force
```

几个要点：

- `git lfs ls-files` 必须能列出 `model_int8.onnx` 和几百个 `.wav`。列不出来
  说明 `.gitattributes` 没生效，push 会以 "file exceeds 10MB" 之类的理由失败。
  补救：`git rm -r --cached .` 再 `git add .`。
- 网页建 Space 时 HF 会先塞一个初始 README，所以远端已经有一个提交了。
  `--force` 是用我们生成的 README 覆盖它。**这只在这个 Space 是刚建的、
  还没有任何东西值得保留时才安全**；如果这个 Space 上已经有你在维护的内容，
  改用 `git pull space main --allow-unrelated-histories` 合并后再推。
- 59MB 走 LFS 上传，代理不稳的话可能要几分钟。中途断了直接重跑 `git push`。

---

## 7. 第五步：看构建日志、拿公网链接

1. 回到 Space 页面，右上角会显示 **Building**，点进去能看实时构建日志。
   首次构建要拉 `python:3.12-slim` 基础镜像、装 onnxruntime，通常 3~6 分钟。
2. 日志里应该能看到我们自己的启动横幅（`[cloud] 监听 0.0.0.0:7860`）。
3. 状态变成 **Running** 后，点 Space 页面右上的 **⋮** → **Open in new window**，
   地址形如 `https://<用户名>-<空间名>.hf.space`。**这就是发给评委的链接。**

上线后的验收清单：

| 检查 | 怎么做 | 期望 |
| --- | --- | --- |
| 服务活着 | 打开 `<链接>/api/health` | `"ok":true`、`"engine":true` |
| 走的是快后端 | 看构建日志里那行 | `推理后端 onnxruntime`，不是 numpy |
| 识别能用 | 首页上传一张玉米照片 | 出类别、置信度、防治建议 |
| 语音能用 | 点建议旁边的播放 | 有普通话播报 |
| 手机相机能用 | 手机打开链接点"实时取景" | 能开摄像头（HTTPS 才有这个权限） |
| 记录不残留 | 看 `/api/health` 的 `records` | 刚部署时是 0 |

---

## 8. 模型文件到底怎么传上云

这是最容易卡住的一步，单独说清楚。三条路，按推荐程度排：

**A. git-lfs push（推荐，第 6 节走的就是这条）**
`.gitattributes` 把 `*.onnx` / `*.wav` / `*.npz` 标成 LFS 对象，git 只提交一个
指针，真身传到 HF 的 LFS 存储。好处是可复现、可回滚、队友 clone 就能拿到。

**B. 网页拖拽上传（补单个文件时方便）**
Space 页面 → **Files** 标签 → **Add file** → 拖进去 → Commit。
适合"就想补一个改过的 `advisory.json`"。不适合首次上传整套语音包：
几百个文件一个个拖会疯，而且网页上传对单文件体积有上限（以 HF 当时的说明为准），
漏了 LFS 标记还会直接被拒。

**C. `--zip` 后搬运（自建 Docker 主机 / 换一台网络更好的机器）**

```powershell
python tools/make_cloud_bundle.py --zip     # -> artifacts/cloud.zip，约 37MB
```

这个 zip 拷到 U 盘、scp 到云服务器、或者传到另一台能顺畅连 HF 的电脑上解开，
再按第 6 节推。zip 里就是完整的部署目录，解开即用。脚本会顺手打印 zip 的
sha256 前 16 位，跨机器搬完对一下就知道有没有传坏。

> 如果 zip 是拷到**云服务器**上直接跑，那就不用 git 了：
> 解开、`docker build`、`docker run -p 80:7860`，再把 80 端口开放出去即可。

---

## 9. 环境变量

在 HF 上：Space 页面 → **Settings** → **Variables and secrets** → New variable。
口令类的要用 **secret** 类型，别用普通 variable（普通变量在 Space 页面上是可见的）。

全部可选，不设也能跑：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PORT` / `HEYAN_PORT` | 7860 | 监听端口。平台注入的 `PORT` 会被自动认下 |
| `HOST` / `HEYAN_HOST` | 0.0.0.0 | 监听地址。云端**不要**改成回环 |
| `HEYAN_ACCESS_TOKEN` | 空 | 设了就要口令才能进；不设则拿到链接的人都能用 |
| `HEYAN_BUNDLE` | 自动取最新 | 部署目录里只有一个包，一般不用设 |
| `HEYAN_LANG` | zh | 界面语言 |
| `HEYAN_THREADS` | 8 | waitress 的 HTTP 工作线程数（并发能力） |
| `HEYAN_INFER_THREADS` | 1 | 单次推理用几个 CPU 线程（留给并发） |
| `HEYAN_RATE_RECOGNIZE` | 30 | 每个来源每分钟识别次数上限 |
| `HEYAN_RATE_DEFAULT` | 600 | 每个来源每分钟其他请求上限 |
| `HEYAN_HOME` | 容器内 artifacts | 记录库和导出件的落盘位置，必须可写 |

**要不要设口令？** 比赛演示建议**不设**——评委点开链接就能用，中间任何一道
口令页都是流失。代价是任何拿到链接的人都能看和删识别记录，所以：
演示完就把 Space 删掉或转 Private，别长期挂着公网。
要收紧就设 `HEYAN_ACCESS_TOKEN` 为一串随机字符，分享链接写成
`https://<域名>/?token=那串字符`，对方首次访问后会自动换成 HttpOnly cookie。

---

## 10. 免费档的如实限制

这些不是 bug，是免费档的取舍。提前知道，才不会在比赛当天措手不及。

- **会休眠。** 长时间没人访问，容器会被停掉（免费档大约 48 小时，以 HF 当时的
  说明为准）。休眠后第一个访问者要等冷启动，通常十几秒到半分钟，页面上表现为
  转圈或 502。**对策：上台演示前 20 分钟自己先把链接点开一遍**，
  或者用手机设个每半小时访问一次的提醒。
- **没有持久存储。** 容器重启后文件系统恢复原样，`artifacts/records/heyan.db`
  里的识别记录会清空。演示够用，但别把它当数据库。要留数据就用界面上的导出
  功能当场下载 JSON/CSV。HF 有付费的持久卷，真需要再开。
- **算力是共享的 CPU 档。** INT8 模型 1.7MB，单张图 CPU 推理几十毫秒，
  8 个 waitress 线程足够应付一个评委摊位。别指望它扛住几百人同时刷。
  免费档通常是 2 vCPU，`HEYAN_THREADS` 默认给到 8 是为了扛住突发点击；
  要是日志里出现明显排队，把它降到 4 反而更顺。
- **构建有超时。** 依赖装太久会被判失败。`requirements-cloud.txt` 故意不含
  torch，就是为了把构建时间压在几分钟内——别手痒往里加训练依赖。
- **限流按出口 IP 计。** 会场里所有人可能共用一个出口 IP，所以云端口径
  （识别 30 次/分钟）比本机口径（12 次/分钟）宽。真觉得挤就调
  `HEYAN_RATE_RECOGNIZE`，调到 60 也完全扛得住。

---

## 11. 换别的 Docker 平台要改什么

代码这边不用改。`heyan/server/cloud.py` 已经认平台注入的 `PORT`，
绑的是 `0.0.0.0`，端口被占会**直接退出**而不是顺延（平台只健康检查它分配的
那一个端口，悄悄换端口等于"日志说起来了、链接永远打不开"）。

要做的只有三件事：

1. 把 `artifacts/cloud/` 当成仓库推给那个平台（或者用它的 CLI / 网页上传 zip）。
2. 启动命令填 `python -m heyan.server.cloud`。
3. 健康检查路径填 `/api/health`。

Render 要在服务设置里手动指定 Dockerfile 路径；Railway 会自动识别根目录的
Dockerfile；Fly 需要 `fly launch` 生成一份配置，里面内部端口写 7860。
绑卡与免费额度各家不同，自己核一遍。

---

## 12. 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| push 报 file exceeds / LFS 相关错误 | `.gitattributes` 没生效。`git rm -r --cached .` 后重新 `git add .`，再用 `git lfs ls-files` 确认 |
| 构建日志 `ImportError: libgomp.so.1` | Dockerfile 里那行装 libgomp1 的命令被删了，加回去 |
| 起来了但识别慢得离谱，日志写 `推理后端 numpy` | onnxruntime 没装上（多半是上一条），静默退回了纯 NumPy 解释器 |
| 界面显示"请先构建模型包" | 模型没进镜像。检查 `git lfs ls-files` 和 `.gitignore`；别把仓库根那份 `.gitignore` 复制进部署目录 |
| 平台一直重启容器 | 端口不对。Dockerfile 的 `EXPOSE`、README 头的 `app_port`、`cloud.DEFAULT_PORT` 三处必须一致 |
| 链接打开是 502 / 503 | Space 正在冷启动或已休眠，等十几秒刷新；持续失败去看构建日志 |
| 评委说点几下就"请求太频繁" | 限流生效了。调高 `HEYAN_RATE_RECOGNIZE`，或给每人发一个带 `?token=` 的链接（令牌也参与配额区分） |
| 手机打不开摄像头 | 必须是 HTTPS。HF 自带，自建服务器要自己配证书 |
| 想撤回公网访问 | Space → Settings → 底部 Delete space，或把 Visibility 转成 Private |

---

## 13. 比赛当天清单

1. 提前一天部署完，自己用手机流量（不是会场 Wi-Fi）打开链接走一遍完整流程。
2. 上台前 20 分钟再点一遍链接，把容器唤醒。
3. 准备 2~3 张本地照片当兜底——万一现场网络不行，还能用笔记本上的
   `python -m heyan.cli serve` 走离线演示。
4. 链接印在 PPT 上，同时准备一个二维码（任何在线二维码生成器都行）。
5. 演示结束后去 Settings 里删掉 Space 或转 Private。公开模式下识别记录对
   所有人敞开，别让它一直挂着。

---

## 14. 这条路归谁测

`tests/test_cloud_deploy.py` 守着云端口径的契约：端口解析优先级、`build_app`
的公网行为、Dockerfile 与代码的端口一致性、云端依赖不含 torch，以及
`tools/make_cloud_bundle.py` 生成的部署目录带模型、不带本机记录、
README 头是 `sdk: docker`。改这几处任何一处，先跑：

```powershell
python -m pytest tests/test_cloud_deploy.py -q
```
