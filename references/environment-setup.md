# EEG Skill 环境手动安装指南

本 Skill 只检测环境，不会替你安装 Anaconda、Python、MNE、JupyterLab、
Git 或任何用户侧软件。完成下面步骤后，请把 Python解释器完整路径、Conda
环境名称和安装位置告诉 Codex。

## 1. 安装 Anaconda 或 Miniconda

- Anaconda 官方下载页：<https://www.anaconda.com/download>
- Miniconda 官方文档：<https://docs.anaconda.com/miniconda/>
- 如果 Anaconda 官网要求账号，请由你自行注册、登录和管理密码。
  Codex不会替你注册账号，也不会索取或填写登录凭据。
- 安装位置请记录下来，例如 `D:\Anaconda3`。不要仅回答“已安装”。

## 2. 创建并进入独立环境

在 Anaconda Prompt 中手动执行：

```powershell
conda create -n mouse-eeg python=3.13
conda activate mouse-eeg
```

## 3. 安装 EEG 分析依赖

仍在已激活的环境中手动执行：

```powershell
conda install -c conda-forge mne numpy scipy pandas matplotlib pytest jupyterlab
```

MNE 官方安装说明：<https://mne.tools/stable/install/index.html>

## 4. 启动 JupyterLab

```powershell
jupyter lab
```

浏览器打开后，不代表 Codex 已知道环境位置。请回报：

1. Conda 环境名称。
2. Python 解释器完整路径，可在 Notebook 中运行 `sys.executable` 查看。
3. Anaconda 或 Miniconda 安装目录。
4. JupyterLab 是否能正常新建并运行 Notebook。

这些信息确认前，Skill 不会继续读取 EEG 文件。

## 5. 准备Antila作者源码

由用户手动准备 `https://github.com/tortugar/Lab` 的本地checkout，并切换到：

`bcb8dae1594e64a511545e34f6050e2a417c1f45`

运行前需要提供本地Lab仓库路径。Skill会核对Git commit和
`PySleep/sleepy.py` SHA256，不会自动clone、下载、安装或替换作者源码。
