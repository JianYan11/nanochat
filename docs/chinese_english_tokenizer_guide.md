# 中英文Tokenizer训练指南

## 一、数据集推荐

### 1. 中文语料库（适合小模型，2-6B字符）

#### 主要推荐（按优先级）：
1. **CLUECorpus Small版本** ⭐ 推荐
   - 规模：约14GB，适合小模型
   - 来源：HuggingFace: `CLUE/cluecorpussmall_14g`
   - 特点：高质量中文语料，涵盖新闻、百科、问答等
   - 下载：`datasets.load_dataset("CLUE/cluecorpussmall_14g", streaming=True)`
   - **适合**: 2-4B字符训练

2. **中文维基百科（过滤版）**
   - 来源：HuggingFace: `pleisto/wikipedia-cn-20230720-filtered`
   - 特点：结构化、高质量，规模适中
   - **适合**: 补充语料

3. **C4中文版（小规模）**
   - 来源：HuggingFace: `HuggingFaceTB/c4-zh`
   - 特点：多样化中文语料
   - **注意**: 可能需要过滤和采样

#### 不推荐（数据量太大）：
- ❌ **CLUECorpus2020完整版**：350亿汉字，对小模型来说太大
- ❌ **WuDaoCorpus**：200GB+，需要申请，对小模型不必要

### 2. 英文语料库（适合小模型）

#### 主要推荐：
1. **FineWeb-Edu-100B**（当前使用）⭐ 推荐
   - 来源：`karpathy/fineweb-edu-100b-shuffle`
   - 特点：高质量教育类英文语料
   - **使用方式**: 只下载需要的shards（每个shard约250M字符）
   - **$100 tier**: 下载8-16个shards（2-4B字符）
   - **$1000 tier**: 下载16-24个shards（4-6B字符）

2. **C4 (Colossal Clean Crawled Corpus)**（备选）
   - 来源：HuggingFace: `c4`
   - 特点：大规模、多样化的英文语料
   - **注意**: 需要采样，因为完整数据集太大

### 3. 中英文平行语料（可选，小规模）

1. **translation2019zh**（可选）
   - 规模：520万对中英文句子
   - 来源：HuggingFace: `translation2019zh`
   - 用途：帮助模型学习语言对应关系
   - **使用**: 可以混合少量平行语料（如10-20%）

## 二、词汇量设置

### 推荐配置（针对nanochat小模型）

| 模型规模 | 词汇量 | 说明 |
|----------|--------|------|
| **$100 tier (d20, 561M)** | 65,536 - 100,000 | 与当前配置一致或略增 |
| **$1000 tier (d32, 1.88B)** | 100,000 - 131,072 | 推荐使用100K |
| **更大模型** | 131,072 (2^17) | 如果训练更大模型 |

### 参考标准
- **当前nanochat**: 65,536（英文为主）
- **GPT-2**: 50,257（主要英文）
- **GPT-4 (cl100k_base)**: ~100,000（多语言，包括中文）
- **建议**: 对于小模型，100K词汇量足够，不需要150K

## 三、数据量和比例（针对nanochat小模型）

### Tokenizer训练数据量建议

| 模型规模 | 中文语料 | 英文语料 | 总字符数 | 说明 |
|----------|----------|----------|----------|------|
| **$100 tier** | 1-2B字符 | 1-2B字符 | 2-4B字符 | 对应speedrun.sh的2B |
| **$1000 tier** | 2-3B字符 | 2-3B字符 | 4-6B字符 | 对应run1000.sh的4B |
| **更大模型** | 5-10B字符 | 5-10B字符 | 10-20B字符 | 如果训练更大模型 |

### 中英文比例
- **推荐比例**: 1:1（平衡）
- **原因**: 小模型数据量有限，保持平衡更有效
- **注意**: 不需要像大模型那样用100-200B字符，2-6B字符足够

## 四、训练命令示例（针对nanochat小模型）

### $100 tier配置（对应speedrun.sh）
```bash
# 2B字符，100K词汇量，中英文1:1
python -m scripts.tok_train_multilingual \
    --max_chars=2_000_000_000 \
    --vocab_size=100000 \
    --chinese_ratio=0.5 \
    --use_chinese
```

### $1000 tier配置（对应run1000.sh）
```bash
# 4B字符，100K词汇量，中英文1:1
python -m scripts.tok_train_multilingual \
    --max_chars=4_000_000_000 \
    --vocab_size=100000 \
    --chinese_ratio=0.5 \
    --use_chinese
```

### 如果资源更有限（最小配置）
```bash
# 1B字符，65K词汇量（保持与当前一致）
python -m scripts.tok_train_multilingual \
    --max_chars=1_000_000_000 \
    --vocab_size=65536 \
    --chinese_ratio=0.5 \
    --use_chinese
```

## 五、监督微调（SFT）数据集

### 当前情况
- 主要使用英文数据集：SmolTalk, ARC, GSM8K, MMLU
- **问题**: 缺少中文对话数据

### 推荐添加的中英文数据集

1. **中文对话数据集**
   - **Belle**: `BelleGroup/train_3.5M_CN` - 350万中文对话
   - **Alpaca-CoT**: 中文指令跟随数据集
   - **Firefly**: 中文多轮对话数据集
   - **ChatGLM训练数据**: 高质量中文对话

2. **中英文混合对话**
   - **OASST**: `OpenAssistant/oasst1` - 多语言对话
   - **ShareGPT**: 包含中英文对话

3. **中文任务数据集**
   - **C-Eval**: 中文评估基准
   - **CMath**: 中文数学题
   - **CMMLU**: 中文多任务语言理解

### 建议的SFT数据混合
```python
train_ds = TaskMixture([
    # 英文任务（保持现有）
    ARC(subset="ARC-Easy", split="train"),
    ARC(subset="ARC-Challenge", split="train"),
    GSM8K(subset="main", split="train"),
    SmolTalk(split="train", stop=10_000),
    
    # 新增：中文对话
    BelleDataset(split="train", size=50_000),  # 中文对话
    AlpacaCoT(split="train", size=20_000),     # 中文指令
    
    # 新增：中文任务
    CMath(split="train"),                      # 中文数学
    CMMLU(subset="train", size=10_000),        # 中文理解
    
    # 其他
    SimpleSpelling(size=300, split="train"),
    SpellingBee(size=300, split="train"),
])
```

## 六、强化学习（RL）数据集

### 当前情况
- 只在GSM8K（英文数学题）上训练
- **问题**: 缺少中文数学题数据

### 推荐添加

1. **中文数学题**
   - **CMath**: 中文数学题数据集
   - **Math23K**: 中文数学应用题
   - **Ape210K**: 中文数学题

2. **中英文混合数学题**
   - 可以混合GSM8K和CMath，确保中英文平衡

### 建议的RL训练
```python
# 混合中英文数学题
train_task = TaskMixture([
    GSM8K(subset="main", split="train"),  # 英文数学
    CMath(split="train"),                  # 中文数学
])
```

## 七、完整训练流程

### 1. Tokenizer训练（中英文混合）
```bash
# 准备中英文混合数据集后
python -m scripts.tok_train \
    --max_chars=20_000_000_000 \
    --vocab_size=131072
```

### 2. 基础预训练
```bash
# 使用中英文混合的预训练数据
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

### 3. 中间训练（Mid-training）
```bash
# 使用中英文混合的任务数据
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
```

### 4. 监督微调（SFT）
```bash
# 使用中英文混合的对话数据
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
```

### 5. 强化学习（RL）
```bash
# 使用中英文混合的数学题
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

## 八、评估和验证

### 添加中文测试文本
在 `scripts/tok_eval.py` 中添加中文测试：

```python
chinese_text = """
自然语言处理是人工智能领域的一个重要分支。
它涉及计算机科学、语言学和认知科学的交叉。
现代的大语言模型如GPT、BERT等，在文本理解、
生成和翻译等任务上取得了显著进展。
"""

# 添加到 all_text 列表
all_text.append(("chinese", chinese_text))
```

### 评估指标
- **压缩比（Ratio）**: bytes/tokens，越高越好
- **与GPT-4对比**: 应该接近或超过GPT-4在中文上的表现
- **中英文平衡**: 确保两种语言的压缩比都合理

## 九、常见问题（针对小模型）

### Q1: 需要多少存储空间？
- **$100 tier** (2B字符):
  - 中文语料（1B字符）: 约2-3GB
  - 英文语料（1B字符）: 约2-3GB（8个shards，每个~100MB）
  - 总计: 约5-6GB
- **$1000 tier** (4B字符):
  - 中文语料（2B字符）: 约4-6GB
  - 英文语料（2B字符）: 约4-6GB（16个shards）
  - 总计: 约10-12GB

### Q2: 训练时间？
- **Tokenizer训练**:
  - 2B字符，100K词汇: 约30-60分钟（取决于CPU）
  - 4B字符，100K词汇: 约1-2小时
- **预训练**: 取决于模型大小和GPU数量（见speedrun.sh和run1000.sh）

### Q3: 如何验证中文效果？
- 运行 `scripts/tok_eval.py` 查看中文压缩比
- 手动测试一些中文文本的分词效果
- 对比GPT-4的tokenizer表现（应该接近但可能略低，因为是小模型）

### Q4: 为什么不需要更大的数据集？
- nanochat是小模型项目（561M-1.88B参数）
- 根据Chinchilla定律，小模型需要的数据量也较小
- 2-6B字符对于tokenizer训练已经足够
- 更大的数据集不会带来明显提升，反而增加存储和训练时间

## 十、总结（针对nanochat小模型）

### 关键配置
- **词汇量**: 100,000（推荐，对应$1000 tier）或 65,536（最小，对应$100 tier）
- **中文语料**: 1-3B字符（根据模型规模）
- **英文语料**: 1-3B字符（根据模型规模）
- **中英文比例**: 1:1（平衡）
- **总字符数**: 2-6B字符（足够小模型使用）

### 与原始配置对比
- **原始**: 2-4B字符（英文），65,536词汇量
- **中英文混合**: 2-6B字符（中英文各半），100,000词汇量
- **存储增加**: 约5-12GB（可接受）
- **训练时间增加**: 约30分钟-1小时（可接受）

### 关键原则
1. ✅ **Tokenizer训练**: 必须使用中英文混合语料
2. ✅ **SFT训练**: 必须使用中英文混合对话数据
3. ✅ **RL训练**: 建议使用中英文混合任务数据
4. ✅ **评估**: 必须同时评估中英文表现

### 下一步
1. 下载并准备中英文数据集
2. 修改数据集加载代码支持混合数据
3. 训练新的tokenizer
4. 更新SFT和RL的数据集配置
5. 评估中英文表现
