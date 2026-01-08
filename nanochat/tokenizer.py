"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference

GPT-4 风格的 BPE（Byte Pair Encoding）分词器。

提供两种实现方式：
1) HuggingFace Tokenizer：可以同时用于训练和推理，但使用起来比较混乱
2) RustBPE Tokenizer：使用 rustbpe 进行训练，使用 tiktoken 进行高效推理
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    # 每个文档都以序列开始（BOS）token 开头，用于分隔文档
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    # 下面的 token 仅在微调期间使用，用于将对话渲染为 token id
    "<|user_start|>", # user messages  # 用户消息开始
    "<|user_end|>",  # 用户消息结束
    "<|assistant_start|>", # assistant messages  # 助手消息开始
    "<|assistant_end|>",  # 助手消息结束
    "<|python_start|>", # assistant invokes python REPL tool  # 助手调用 Python REPL 工具开始
    "<|python_end|>",  # Python 工具调用结束
    "<|output_start|>", # python REPL outputs back to assistant  # Python REPL 输出返回给助手开始
    "<|output_end|>",  # Python 输出结束
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
# 注意：这个分割模式与 GPT-4 不同，我们使用 \p{N}{1,2} 而不是 \p{N}{1,3}
# 这样做是因为对于较小的词汇表大小，我不想在数字上"浪费"太多 token
# 我还没有验证这是否真的是个好主意，待办事项。
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
# 基于 HuggingFace Tokenizer 的通用 GPT-4 风格分词器
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""
    """
    围绕 HuggingFace Tokenizer 的轻量级包装器，提供一些实用功能
    
    这个类封装了 HuggingFace 的 Tokenizer，使其更容易使用，并提供了
    一些额外的工具方法来处理特殊 token 和编码/解码操作。
    """

    def __init__(self, tokenizer):
        """
        初始化 HuggingFace Tokenizer 包装器
        
        Args:
            tokenizer: HuggingFace Tokenizer 实例
        """
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        # 从 HuggingFace 预训练分词器初始化（例如 "gpt2"）
        """
        从 HuggingFace 预训练模型加载分词器
        
        Args:
            hf_path: HuggingFace 模型路径或名称（如 "gpt2"）
        
        Returns:
            HuggingFaceTokenizer 实例
        """
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        # 从本地磁盘目录初始化（例如 "out/tokenizer"）
        """
        从本地目录加载分词器
        
        Args:
            tokenizer_dir: 包含 tokenizer.json 文件的目录路径
        
        Returns:
            HuggingFaceTokenizer 实例
        """
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # 从文本迭代器训练分词器
        """
        从文本迭代器训练新的 BPE 分词器
        
        Args:
            text_iterator: 文本数据的迭代器（每次 yield 一个字符串）
            vocab_size: 词汇表大小（包含特殊 token）
        
        Returns:
            训练好的 HuggingFaceTokenizer 实例
        
        配置说明：
        - 使用 BPE（Byte Pair Encoding）算法
        - 使用 GPT-4 风格的正则表达式进行预分词
        - 使用 ByteLevel 编码，确保可以处理任何 Unicode 字符
        """
        # Configure the HuggingFace Tokenizer
        # 配置 HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!  # 必需！启用字节回退
            unk_token=None,  # 不使用未知 token
            fuse_unk=False,  # 不融合未知 token
        ))
        # Normalizer: None
        # 标准化器：无（不进行文本标准化）
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # 预分词器：GPT-4 风格
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # GPT-4 用于在 BPE 之前将文本分割成组的正则表达式模式
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        # 注意：模式从 \p{N}{1,3} 改为 \p{N}{1,2}，因为我怀疑对于非常小的模型
        # 和较小的词汇表大小，这会浪费一些 token 空间（但我还没有验证这一点！待办事项）
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        # HuggingFace 要求必须用 Regex 包装！
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        # 解码器：ByteLevel（与 ByteLevel 预分词器配对使用）
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        # 后处理器：无
        tokenizer.post_processor = None
        # Trainer: BPE
        # 训练器：BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency  # 无最小频率要求
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始字母表（256 个字节）
            special_tokens=SPECIAL_TOKENS,  # 特殊 token 列表
        )
        # Kick off the training
        # 开始训练
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        """
        获取词汇表大小
        
        Returns:
            int: 词汇表中的 token 总数
        """
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        """
        获取所有特殊 token 的列表
        
        Returns:
            list[str]: 特殊 token 字符串列表
        """
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        """
        将 token id 转换为 token 字符串
        
        Args:
            id: token id（整数）
        
        Returns:
            str: token 对应的字符串
        """
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # 编码单个字符串
        # prepend/append can be either a string of a special token or a token id directly.
        # prepend/append 可以是特殊 token 的字符串，也可以是 token id（整数）
        """
        编码单个字符串（内部方法）
        
        Args:
            text: 要编码的文本字符串
            prepend: 在文本前添加的 token（可以是特殊 token 字符串或 token id）
            append: 在文本后添加的 token（可以是特殊 token 字符串或 token id）
        
        Returns:
            list[int]: token id 列表
        """
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        # 通过精确匹配编码单个特殊 token
        """
        编码单个特殊 token
        
        Args:
            text: 特殊 token 字符串（如 "<|bos|>"）
        
        Returns:
            int: 特殊 token 的 id，如果不存在则返回 None
        """
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 不同的 HuggingFace 模型使用不同的 BOS token，一致性较差
        # 1) attempt to find a <|bos|> token
        # 1) 尝试查找 <|bos|> token
        """
        获取序列开始（BOS）token 的 id
        
        尝试多种可能的 BOS token 名称，因为不同的 HuggingFace 模型使用不同的命名。
        
        Returns:
            int: BOS token 的 id
        
        Raises:
            AssertionError: 如果找不到任何 BOS token
        """
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        # 2) 如果失败，尝试查找 <|endoftext|> token（例如 GPT-2 模型）
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        # 3) 如果这些都失败，最好崩溃而不是静默返回 None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        """
        编码文本（可以是字符串或字符串列表）
        
        Args:
            text: 要编码的文本（str 或 list[str]）
            *args, **kwargs: 传递给 _encode_one 的其他参数
        
        Returns:
            list[int] 或 list[list[int]]: token id 列表（单个文本）或 token id 列表的列表（多个文本）
        """
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        """
        使对象可调用，直接调用 encode 方法
        
        允许使用 tokenizer(text) 而不是 tokenizer.encode(text)
        """
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        将 token id 列表解码回文本字符串
        
        Args:
            ids: token id 列表（list[int]）
        
        Returns:
            str: 解码后的文本字符串（保留特殊 token）
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        # 将分词器保存到磁盘
        """
        将分词器保存到指定目录
        
        Args:
            tokenizer_dir: 保存目录路径（将在此目录下创建 tokenizer.json 文件）
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
# 基于 rustbpe + tiktoken 组合的分词器
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""
    """
    围绕 tiktoken（用于高效推理）的轻量级包装器，但使用 rustbpe 进行训练
    
    这个类结合了两个库的优势：
    - rustbpe: 用于训练 BPE 分词器（快速、高效）
    - tiktoken: 用于推理时的编码/解码（OpenAI 使用的高性能库）
    
    训练时使用 rustbpe，推理时使用 tiktoken，这样可以在保持训练速度的同时
    获得推理时的高性能。
    """

    def __init__(self, enc, bos_token):
        """
        RustBPETokenizer 的初始化函数。

        什么是 BPE？
        ------------------
        BPE（Byte Pair Encoding，字节对编码）是一种用于文本分词的子词单元（subword）分割算法。
        它通过反复合并训练语料中出现频率最高的字节对，将词汇分解为更细粒度的单元，有效覆盖未知词和减少词表大小。
        GPT 系列模型、Transformer 等广泛采用 BPE 作为 tokenizer。

        什么是 bos_token？
        ------------------
        bos_token (beginning of sequence token，序列开始标记) 是一个特殊的 token，
        用来指示生成/编码一段文本的开始。常见如 "<|bos|>" 或 OpenAI 的 "<|endoftext|>"。
        它在语言模型任务中用于指明文本起始位置，有助于模型区分文本的结构。

        这个初始化函数如何定义 tokenizer？
        -----------------------
        1. 接收已训练好的 tiktoken.Encoding 对象 enc，包含所有 BPE 合并规则和特殊 token 的定义。
        2. 保存 enc 到 self.enc，供后续编码/解码使用。
        3. 用 encode_special 方法把 bos_token 转换为其对应的 token id，并保存在 self.bos_token_id 中，以便模型推理时能用于生成起始 token。
        """

        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)


    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        # 1) 使用 rustbpe 进行训练
        """
        从文本迭代器训练新的 BPE 分词器
        
        训练流程：
        1. 使用 rustbpe 训练 BPE 模型（快速）
        2. 将训练结果转换为 tiktoken.Encoding 对象（用于高效推理）
        
        Args:
            text_iterator: 文本数据的迭代器（每次 yield 一个字符串）
            vocab_size: 词汇表大小（包含特殊 token）
        
        Returns:
            训练好的 RustBPETokenizer 实例
        """
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        # 特殊 token 稍后在 __init__ 中插入，我们在这里不训练它们
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        # 2) 构建关联的 tiktoken 编码对象用于推理
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            # 字典[字节, int]（token 字节 -> 合并优先级）
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
            # 字典[str, int]（特殊 token 名称 -> token id）
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        从本地目录加载分词器
        
        Args:
            tokenizer_dir: 包含 tokenizer.pkl 文件的目录路径
        
        Returns:
            RustBPETokenizer 实例
        """
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        # tiktoken 将特殊文档分隔符 token 称为 "<|endoftext|>"
        # 是的，这很令人困惑，因为这个 token 几乎总是被添加到文档的开头
        # 它最常用于在推理期间向 LLM 发出新序列开始的信号等
        # 所以在 nanoChat 中我们总是使用 "<|bos|>"（序列开始的缩写），
        # 但历史上它通常被称为 "<|endoftext|>"
        """
        从 tiktoken 预训练编码加载分词器
        
        可以加载 OpenAI 的预训练编码，如 "gpt-4"、"cl100k_base" 等。
        
        Args:
            tiktoken_name: tiktoken 编码名称（如 "gpt-4"、"cl100k_base"）
        
        Returns:
            RustBPETokenizer 实例（使用 "<|endoftext|>" 作为 BOS token）
        """
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        """
        获取词汇表大小
        
        Returns:
            int: 词汇表中的 token 总数
        """
        return self.enc.n_vocab

    def get_special_tokens(self):
        """
        获取所有特殊 token 的集合
        
        Returns:
            set[str]: 特殊 token 字符串集合
        """
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """
        将 token id 转换为 token 字符串
        
        Args:
            id: token id（整数）
        
        Returns:
            str: token 对应的字符串
        """
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """
        编码单个特殊 token（带缓存以提高性能）
        
        Args:
            text: 特殊 token 字符串（如 "<|bos|>"）
        
        Returns:
            int: 特殊 token 的 id
        """
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        """
        获取序列开始（BOS）token 的 id
        
        Returns:
            int: BOS token 的 id
        """
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings
        # text 可以是字符串或字符串列表
        """
        编码文本（可以是字符串或字符串列表）
        
        Args:
            text: 要编码的文本（str 或 list[str]）
            prepend: 在文本前添加的 token（可以是特殊 token 字符串或 token id）
            append: 在文本后添加的 token（可以是特殊 token 字符串或 token id）
            num_threads: 批量编码时使用的线程数（仅对列表输入有效）
        
        Returns:
            list[int] 或 list[list[int]]: token id 列表（单个文本）或 token id 列表的列表（多个文本）
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
                # 待办事项：这里可能稍微低效？嗯
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
                    # 待办事项：同样的问题
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        """
        使对象可调用，直接调用 encode 方法
        
        允许使用 tokenizer(text) 而不是 tokenizer.encode(text)
        """
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        将 token id 列表解码回文本字符串
        
        Args:
            ids: token id 列表（list[int]）
        
        Returns:
            str: 解码后的文本字符串
        """
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        # 将编码对象保存到磁盘
        """
        将分词器保存到指定目录
        
        Args:
            tokenizer_dir: 保存目录路径（将在此目录下创建 tokenizer.pkl 文件）
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        
        将单个聊天对话转换为 token id 列表（我们在这里称之为"文档"）。
        
        这个方法将对话格式（包含 user/assistant 消息）转换为模型训练所需的 token 序列。
        同时返回一个 mask，指示哪些 token 需要被监督学习（只有 assistant 的回复需要学习）。
        
        Args:
            conversation: 对话字典，格式为 {"messages": [...]}，每个消息包含 "role" 和 "content"
            max_tokens: 最大 token 数，超过会被截断（防止内存溢出）
        
        Returns:
            tuple: (ids, mask)
                - ids: list[int]，对话的 token id 列表
                - mask: list[int]，与 ids 等长，mask[i] = 1 表示该 token 需要被监督学习（assistant 回复），
                        mask[i] = 0 表示不需要学习（user 消息、特殊 token 等）
        """
        # ids, masks that we will return and a helper function to help build them up.
        # 我们将返回的 ids 和 mask，以及一个辅助函数来帮助构建它们
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            """
            辅助函数：添加 token 到 ids 和 mask 列表
            
            Args:
                token_ids: token id（int）或 token id 列表（list[int]）
                mask_val: mask 值（0 或 1），表示该 token 是否需要被监督学习
            """
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        # 有时第一条消息是系统消息...
        # => 只需将其与第二条（用户）消息合并
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            # 这里需要进行一些对话"手术"...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            # 避免修改原始对话
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        # 获取我们需要的所有特殊 token
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        # 现在我们可以对对话进行分词
        add_tokens(bos, 0)  # BOS token 不需要学习
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            # 这里进行一些假设的合理性检查，以防止错误使用
            must_be_from = "user" if i % 2 == 0 else "assistant"
            # 消息必须交替出现：偶数索引是 user，奇数索引是 assistant
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            # content 可以是简单字符串或部分列表（例如包含工具调用）
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)  # 特殊 token 不需要学习
                add_tokens(value_ids, 0)   # 用户消息不需要学习
                add_tokens(user_end, 0)    # 特殊 token 不需要学习
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)  # 特殊 token 不需要学习
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    # 简单字符串 => 直接添加 token
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)  # 助手回复需要学习
                elif isinstance(content, list):
                    # 内容是一个列表，可能包含文本和工具调用
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            # 文本部分 => 直接添加 token
                            add_tokens(value_ids, 1)  # 文本需要学习
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            # Python 工具调用 => 在 <|python_start|> 和 <|python_end|> 之间添加 token
                            add_tokens(python_start, 1)  # 工具调用标记需要学习
                            add_tokens(value_ids, 1)      # Python 代码需要学习
                            add_tokens(python_end, 1)     # 工具调用标记需要学习
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # Python 输出 => 在 <|output_start|> 和 <|output_end|> 之间添加 token
                            # none of these tokens are supervised because the tokens come from Python at test time
                            # 这些 token 都不需要监督学习，因为 token 来自测试时的 Python 输出
                            add_tokens(output_start, 0)  # 输出标记不需要学习
                            add_tokens(value_ids, 0)      # Python 输出不需要学习
                            add_tokens(output_end, 0)    # 输出标记不需要学习
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)  # 助手结束标记需要学习

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        # 截断到最大 max_tokens 个 token（有助于防止内存溢出）
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        """
        用于调试的小辅助函数：可视化 render_conversation 的分词结果
        
        这个函数将 token id 和 mask 转换为彩色字符串，方便查看哪些 token 需要学习。
        - 绿色：mask = 1，需要学习的 token（assistant 回复）
        - 红色：mask = 0，不需要学习的 token（user 消息、特殊 token 等）
        
        Args:
            ids: token id 列表
            mask: mask 列表（与 ids 等长）
            with_token_id: 是否在输出中包含 token id（灰色显示）
        
        Returns:
            str: 格式化的字符串，用 '|' 分隔各个 token，带颜色标记
        """
        RED = '\033[91m'    # 红色（不需要学习）
        GREEN = '\033[92m'  # 绿色（需要学习）
        RESET = '\033[0m'   # 重置颜色
        GRAY = '\033[90m'   # 灰色（用于显示 token id）
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        
        用于强化学习场景。在这种设置下，我们希望渲染对话以引导助手完成回复。
        与聊天 SFT 情况不同，我们不需要返回 mask。
        
        这个方法与 render_conversation 的区别：
        - 移除最后一条 assistant 消息（因为我们要让模型生成它）
        - 在末尾添加 <|assistant_start|> token，提示模型开始生成
        - 只返回 token id 列表，不返回 mask（因为不需要监督学习）
        
        Args:
            conversation: 对话字典，最后一条消息必须是 assistant 的回复
        
        Returns:
            list[int]: token id 列表，用于引导模型生成回复
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        # 我们需要做一些"手术"：需要弹出最后一条消息（助手的消息）
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        # 避免修改原始对话
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace
        # 原地删除最后一条消息（助手的消息）

        # Now tokenize the conversation
        # 现在对对话进行分词
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        # 最后，为了引导助手完成回复，在末尾添加助手开始 token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions
# nanochat 特定的便利函数

def get_tokenizer():
    """
    从默认位置加载分词器（nanochat 项目的便利函数）
    
    默认从项目根目录下的 "tokenizer" 目录加载 RustBPETokenizer。
    这是 nanochat 项目中最常用的获取分词器的方式。
    
    Returns:
        RustBPETokenizer 实例
    """
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    """
    加载 token 字节表示（用于某些特殊用途，如可视化）
    
    这个函数加载每个 token 的字节表示，通常用于调试或可视化。
    文件由 tok_train.py 在训练分词器时生成。
    
    Args:
        device: PyTorch 设备（"cpu" 或 "cuda"），用于加载 tensor
    
    Returns:
        torch.Tensor: token 字节表示，形状为 [vocab_size, ...]
    
    Raises:
        AssertionError: 如果 token_bytes.pt 文件不存在
    """
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
