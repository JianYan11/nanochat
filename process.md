虚拟环境为 nano-gpt

## 2026-01-10: 改造 download_cc100_zhhans.py 下载方式

**Where**: `/root/autodl-tmp/nanochat/download_cc100_zhhans.py`

**What**: 
- 将下载方式从 HuggingFace datasets 库的 streaming 模式改为直接使用 requests 下载 parquet 文件
- 模仿 `nanochat/dataset.py` 的下载方式
- 添加了详细的下载进度显示（使用 tqdm）
- 添加了文件级别的进度条和总体进度条

**Why**:
- HuggingFace datasets 库的 streaming 模式在处理 parquet 文件时，即使设置了 streaming=True，首次迭代也需要先下载并解析整个文件的元数据，导致阻塞
- 直接使用 requests 下载可以立即开始写入文件，无需等待元数据解析
- 可以更好地监控下载进度，每个文件的下载都有独立的进度条
- 下载速度更快，没有抽象层的开销

**主要改动**:
1. 移除了 `from datasets import load_dataset`，改用 `import requests`
2. 添加了 `download_and_process_parquet()` 函数，直接下载并读取 parquet 文件
3. 使用 tqdm 显示每个文件的下载进度和总体处理进度
4. 下载后立即读取并处理文档，无需等待
5. 添加了更详细的进度信息输出，包括当前处理的文件编号

## 2026-01-10: Git 提交到远程仓库

**Where**: `/root/autodl-tmp/nanochat`

**What**: 
- 配置了 git 用户信息（user.name 和 user.email）
- 添加了所有更改的文件到暂存区
- 提交了更改（commit hash: 1f367b7）
- 尝试推送到远程仓库，但需要身份验证

**Why**:
- 需要将本地更改同步到远程仓库
- 当前使用 HTTPS URL，需要 GitHub Personal Access Token 进行身份验证

**提交的文件**:
- `nanochat/dataset.py` (修改)
- `scripts/tok_eval.py` (修改)
- `cc100_zhhans_repack.log` (新增)
- `docs/chinese_english_tokenizer_guide.md` (新增)
- `download_cc100_zhhans.py` (新增)
- `process.md` (新增)

**身份验证说明**:
当前远程仓库使用 HTTPS URL (`https://github.com/JianYan11/nanochat.git`)，推送需要以下方式之一：
1. **GitHub Personal Access Token (PAT)**: 在 GitHub 创建 token，推送时用户名输入 GitHub 用户名，密码输入 token
2. **SSH 密钥**: 生成 SSH 密钥对，添加到 GitHub，然后将远程 URL 改为 SSH 格式
3. **GitHub CLI**: 使用 `gh auth login` 进行身份验证
