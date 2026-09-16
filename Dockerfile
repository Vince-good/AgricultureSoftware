# 云端部署镜像：HF Spaces(Docker SDK) / Render / Railway / Fly / 任何 Docker 主机通用。
#
# 与本地 `heyan serve` 的差别全在启动命令里：监听 0.0.0.0、端口取 $PORT、端口不可用
# 就退出而不顺延、一律按公网暴露加固。细节与全部环境变量见 heyan/server/cloud.py。
#
# 推荐先用 tools/make_cloud_bundle.py 生成干净的部署目录再构建（只带一个模型包，
# 不带本机调试留下的识别记录）。直接从仓库根构建也行，.dockerignore 会替你挡掉
# 训练数据与 artifacts/records。
FROM python:3.12-slim

# onnxruntime 的动态库链接了 OpenMP，slim 镜像里没有 libgomp.so.1，
# 不装的话 import onnxruntime 直接 ImportError，会静默退回慢一个数量级的 NumPy 后端。
#
# 这里只 apt-get clean，没按惯例再删掉 apt 的包索引目录：那条命令的路径是写死的
# POSIX 系统目录，会被 tests/test_no_absolute_paths.py 判红。省下的是约 30MB，
# 不值得为它给守卫开一个例外——容器内的系统目录本来就不影响"换机器能不能跑"，
# 但守卫一旦有了例外，下一个例外就守不住了。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean

# HEYAN_HOME 是运行期根目录：模型包从这里找，记录库和导出件也写在这里，必须可写。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEYAN_HOME=/app/artifacts

WORKDIR /app

# 先只拷依赖清单再安装：改代码不会打穿这一层缓存，重建能快几分钟
COPY requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY . .

# HF Spaces 以 uid 1000 运行容器；别的平台默认 root 也不受影响。
# 记录库要写 sqlite，所以整个工作目录交给 1000。
RUN chown -R 1000:1000 /app
USER 1000

# HF Spaces 的 Docker SDK 约定端口；平台注入 $PORT 时以 $PORT 为准
EXPOSE 7860

CMD ["python", "-m", "heyan.server.cloud"]
