#!/usr/bin/env python3
"""
使用直接下载方式获取 cc100-documents zh-Hans 数据集并打包成 shard parquet
模仿 dataset.py 的下载方式，使用 requests 直接下载
"""
import os
import sys
import time
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from nanochat.common import get_base_dir

# 配置
TARGET_CHARS = 2_000_000_000  # 2B characters
CHARS_PER_SHARD = 250_000_000  # 每个 shard 约 250M chars
ROW_GROUP_SIZE = 1024

# cc100-documents 的配置
BASE_URL = "https://huggingface.co/datasets/singletongue/cc100-documents/resolve/main/zh-Hans"
NUM_PARQUET_FILES = 103  # train-00000-of-00103.parquet 到 train-00102-of-00103.parquet

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 80)
print("下载 cc100-documents (zh-Hans) 数据集")
print(f"目标字符数: {TARGET_CHARS:,}")
print(f"每个 shard: {CHARS_PER_SHARD:,} chars")
print(f"输出目录: {DATA_DIR}")
print(f"源文件数: {NUM_PARQUET_FILES} 个 parquet 文件")
print("=" * 80)
print()

# 检查现有文件
existing = [f for f in os.listdir(DATA_DIR) if f.startswith("shard_") and f.endswith(".parquet")]
if existing:
    response = input(f"发现 {len(existing)} 个现有 shard 文件。是否删除？(y/N): ")
    if response.lower() == 'y':
        for f in existing:
            os.remove(os.path.join(DATA_DIR, f))
        print(f"已删除 {len(existing)} 个文件")
    else:
        print("保留现有文件，将在现有基础上追加")
        # 找到最大的 shard index
        max_idx = -1
        for f in existing:
            try:
                idx = int(f.replace("shard_", "").replace(".parquet", ""))
                max_idx = max(max_idx, idx)
            except:
                pass
        shard_index = max_idx + 1
else:
    shard_index = 0

print()
print("开始下载并处理...")
print()

shard_docs = []
shard_chars = 0
total_chars = 0
total_docs = 0

t_start = time.time()
t_last_log = t_start
LOG_INTERVAL = 5.0  # 每 5 秒打印一次进度

def download_and_process_parquet(file_index):
    """下载单个 parquet 文件并返回文档列表"""
    filename = f"train-{file_index:05d}-of-{NUM_PARQUET_FILES:05d}.parquet"
    url = f"{BASE_URL}/{filename}"
    
    # 下载到临时文件
    temp_path = os.path.join(DATA_DIR, f"temp_{filename}")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                         desc=f"下载 {filename}", leave=False, ncols=100) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # No content-length header, download without progress bar
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        
        # 读取 parquet 文件
        pf = pq.ParquetFile(temp_path)
        texts = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts.extend(rg.column('text').to_pylist())
        
        # 删除临时文件
        os.remove(temp_path)
        
        return texts
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e

try:
    # 使用 tqdm 显示总体进度
    with tqdm(total=NUM_PARQUET_FILES, desc="处理文件", unit="file", ncols=100) as file_pbar:
        # 逐个下载和处理 parquet 文件
        for file_idx in range(NUM_PARQUET_FILES):
            filename = f"train-{file_idx:05d}-of-{NUM_PARQUET_FILES:05d}.parquet"
            
            try:
                texts = download_and_process_parquet(file_idx)
                file_pbar.set_description(f"处理 {filename} ({len(texts)} 文档)")
            except Exception as e:
                print(f"\n❌ 下载文件 {file_idx} ({filename}) 失败: {e}", flush=True)
                file_pbar.update(1)
                continue
            
            # 处理每个文档
            for text in texts:
                if not isinstance(text, str):
                    continue
                
                shard_docs.append(text)
                n = len(text)  # 使用 len() 统计字符数（Unicode code points）
                shard_chars += n
                total_chars += n
                total_docs += 1

                # 定期打印进度
                now = time.time()
                if now - t_last_log >= LOG_INTERVAL:
                    elapsed = max(now - t_start, 1e-6)
                    cps = total_chars / elapsed  # chars per second
                    remaining = max(TARGET_CHARS - total_chars, 0)
                    eta = remaining / max(cps, 1e-6)
                    
                    print(
                        f"\n[进度] 文件: {file_idx+1}/{NUM_PARQUET_FILES} | "
                        f"文档数={total_docs:,} | "
                        f"总字符={total_chars:,}/{TARGET_CHARS:,} ({100*total_chars/TARGET_CHARS:.1f}%) | "
                        f"当前 shard={shard_index} ({shard_chars:,}/{CHARS_PER_SHARD:,}) | "
                        f"速度={cps:,.0f} chars/s | "
                        f"预计剩余={eta/60:.1f} 分钟",
                        flush=True
                    )
                    t_last_log = now

                # 检查是否需要写入 shard
                should_write = shard_chars >= CHARS_PER_SHARD
                should_stop = total_chars >= TARGET_CHARS

                if should_write or should_stop:
                    shard_path = os.path.join(DATA_DIR, f"shard_{shard_index:05d}.parquet")
                    table = pa.Table.from_pydict({"text": shard_docs})
                    pq.write_table(
                        table,
                        shard_path,
                        row_group_size=ROW_GROUP_SIZE,
                        use_dictionary=False,
                        compression="zstd",
                        compression_level=3,
                        write_statistics=False,
                    )
                    dt = time.time() - t_start
                    print(
                        f"\n✓ 已写入 {shard_path}\n"
                        f"  文档数: {len(shard_docs):,} | "
                        f"shard 字符数: {shard_chars:,} | "
                        f"累计字符数: {total_chars:,}/{TARGET_CHARS:,} | "
                        f"耗时: {dt:.2f}s\n",
                        flush=True
                    )
                    
                    shard_docs = []
                    shard_chars = 0
                    shard_index += 1

                if should_stop:
                    break
            
            file_pbar.update(1)
            
            if total_chars >= TARGET_CHARS:
                print(f"\n已达到目标字符数 {TARGET_CHARS:,}，停止处理", flush=True)
                break

    # 如果还有剩余数据，写入最后一个 shard
    if shard_docs:
        shard_path = os.path.join(DATA_DIR, f"shard_{shard_index:05d}.parquet")
        table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            table,
            shard_path,
            row_group_size=ROW_GROUP_SIZE,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        print(
            f"\n✓ 已写入最后一个 shard: {shard_path}\n"
            f"  文档数: {len(shard_docs):,} | "
            f"shard 字符数: {shard_chars:,} | "
            f"累计字符数: {total_chars:,}/{TARGET_CHARS:,}\n",
            flush=True
        )

except KeyboardInterrupt:
    print("\n\n用户中断下载")
    if shard_docs:
        print(f"正在保存当前进度（{len(shard_docs)} 个文档，{shard_chars:,} 字符）...")
        shard_path = os.path.join(DATA_DIR, f"shard_{shard_index:05d}.parquet")
        table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            table,
            shard_path,
            row_group_size=ROW_GROUP_SIZE,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        print(f"已保存到 {shard_path}")
    sys.exit(1)

total_time = time.time() - t_start
print("=" * 80)
print("下载完成！")
print(f"总文档数: {total_docs:,}")
print(f"总字符数: {total_chars:,}")
print(f"总 shard 数: {shard_index}")
print(f"总耗时: {total_time/60:.1f} 分钟")
if total_time > 0:
    print(f"平均速度: {total_chars/total_time:,.0f} chars/s")
print(f"输出目录: {DATA_DIR}")
print("=" * 80)
