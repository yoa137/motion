# Windows WSL2 / Docker 使用

用户已在 Ubuntu 22.04 / WSLg 和 Docker 中打开过空 MuJoCo 窗口。
本源码恢复版本尚未在用户电脑验证 Panda 场景。

## 直接在 WSL 运行

将完整源码（含 assets 和 uv.lock）放在 ~/projects/motion，然后执行：

```bash
cd ~/projects/motion
export UV_CACHE_DIR="$PWD/.cache/uv"
uv python install 3.12
uv sync --frozen
uv run --frozen motion --task slide --viewer
```

## 使用此前本地构建的 motion-sim:0.1 镜像

该镜像已安装 Python 3.12、MuJoCo 3.3.7、NumPy 2.2.6 和图形库。
它不含项目源码、uv 或 pytest。可通过 PYTHONPATH 挂载源码运行，
不挂载或复用宿主机 .venv：

```bash
cd ~/projects/motion
mkdir -p runs
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --env DISPLAY="$DISPLAY" \
  --env LIBGL_ALWAYS_SOFTWARE=1 \
  --env PYTHONPATH=/workspace/src \
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src="$PWD/runs",dst=/workspace/runs \
  --workdir /workspace \
  motion-sim:0.1 \
  python -m motion.cli --task slide --viewer
```

不显示窗口时移除 --viewer、DISPLAY 和 X11 挂载即可。
该容器的完整项目测试需要额外安装开发依赖；推荐先用 WSL 的 uv 环境运行测试。
