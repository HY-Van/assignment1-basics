# Part A Byte-Pair Encoding (BPE) Tokenizer 整体思路

[`tokenizer.py`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L1) 其实在做两件事：

1. 训练 BPE：从原始文本里统计“最常一起出现的相邻字节对”，不断合并，得到 `vocab + merges`。
2. 使用 BPE：拿着已经训练好的 `vocab + merges`，把新文本编码成 token id，再把 id 解码回字符串。

你可以把它理解成一条固定流水线：

`原始字符串 -> special token 切分 -> regex 预分词 -> UTF-8 bytes -> BPE merge -> token id`

---

**第一部分：文件开头的基础工具**

[`GPT2_SPLIT_PATTERN`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L11) 和 [`GPT2_SPLIT_RE`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L12) 是 GPT-2 风格的预分词规则。它的作用不是“直接得到最终 token”，而是先把文本粗略切成比较合理的小块，比如单词、数字、标点、空白，这样 BPE 训练和编码都只在这些小块内部发生。

[`Token` / `Pair` / `TokenSeq`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L14-L16) 是类型别名，帮助我们读代码：
- `Token` 是一个 `bytes`
- `Pair` 是两个 token 组成的相邻对
- `TokenSeq` 是一个 token 序列

[`bytes_to_unicode()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L19-L29) 和 [`unicode_to_bytes()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L32-L34) 是为了兼容 GPT-2 的序列化格式。因为有些 byte 不可打印，GPT-2 会把它们映射到“可打印字符”再存进 `vocab.json` / `merges.txt`。这两个函数就是做双向转换。

[`_special_token_regex()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L37-L41) 会把 special tokens 编成正则，而且先按长度降序排，这样像 `A` 和 `AA` 这种重叠 token，会优先匹配更长的 `AA`。

[`split_on_special_tokens()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L44-L59) 很关键。它不会直接编码，只会把文本切成一段段：
- 普通文本段：`(False, segment)`
- special token 段：`(True, segment)`

这样后面我们就能保证 special token 永远不会被拆开。

[`pretokenize_text()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L62-L64) 把普通文本段按 GPT-2 regex 切成 pre-token，并立刻转成 UTF-8 bytes。这里已经体现了 Lec4 的核心思想：BPE 不是直接在 Unicode 字符上做，而是在 bytes 上做。

---

**第二部分：BPE 训练是怎么写的**

[`merge_pair_in_word()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L67-L77) 是“单词内部执行一次 merge”的最小原子操作。比如当前词是：

```python
(b't', b'h', b'e')
```

如果要 merge `(b't', b'h')`，它就会变成：

```python
(b'th', b'e')
```

[`_word_from_bytes()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L80-L81) 会把一个 bytes 串拆成“单字节 token 序列”。例如 `b"cat"` 变成：

```python
(b'c', b'a', b't')
```

[`_iter_pairs()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L84-L86) 用来枚举一个 token 序列里的所有相邻 pair。比如 `(b'c', b'a', b't')` 会产生：
- `(b'c', b'a')`
- `(b'a', b't')`

[`_add_word_stats()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L89-L98) 和 [`_remove_word_stats()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L100-L115) 是训练提速的核心。
它们维护了两份全局信息：
- `pair_counts`：每个 pair 在整个语料里出现了多少次
- `pair_to_words`：某个 pair 出现在哪些“词”里

这样每次 merge 后，我们只更新受影响的词，而不是重新扫描整个语料。

[`_collect_word_freqs()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L118-L127) 做了训练前的准备：
- 先按 special token 切开
- special token 直接跳过，不参加 merge 统计
- 普通文本做 pre-tokenization
- 每个 pre-token 变成 token 序列后放进 `Counter`

这就得到“每个 pre-token 出现了多少次”。

真正的训练主函数是 [`train_bpe()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L130-L177)。

它的流程是：
- [L135-L143] 初始化词表：先放 256 个单字节，再追加 special tokens。
- [L149-L150] 读取训练文本，收集每个 pre-token 的频次。
- [L152-L155] 根据这些 pre-token，初始化全局 pair 统计。
- [L157-L176] 反复做 merge，直到词表达到 `vocab_size`。

最重要的一行是 [`best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L159)：
- 先按频次最大选
- 频次相同时，按 pair 本身的字典序选更大的那个
这正好对应题目要求的 tie-break。

然后对所有受影响的词：
- 先把旧词从统计里删掉
- 做一次 merge
- 再把新词加回统计里

这就是“增量更新”，也是这版训练能过速度测试的原因。

---

**第三部分：Tokenizer 类怎么把文本编码/解码**

[`Tokenizer.__init__()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L180-L207) 做的是“建索引”。
它把输入的 `vocab + merges` 转成后面编码时好查的数据结构：

- [`id_to_token`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L187)：id -> bytes
- [`token_to_id`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L188)：bytes -> id
- [`pair_rank`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L190)：某个 merge pair 是第几轮学到的

`pair_rank` 很重要，因为编码时不是重新训练，而是“按训练时学到的先后顺序应用 merges”。rank 越小，说明越早学到，优先级越高。

[`from_files()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L209-L240) 负责从磁盘读取 GPT-2 风格的 `vocab.json` 和 `merges.txt`。它做的事很机械：
- 读文件
- 把可打印字符映回原始 bytes
- 再调用 `Tokenizer(...)`

[`_merge_pretoken()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L242-L266) 是编码阶段最核心的函数。
它的输入是一个 pre-token 的 bytes，比如 `b" hello"`，先拆成单字节 token 列表，然后不断循环：
- 扫当前所有相邻 pair
- 找到 rank 最小的那个 pair
- 把这个 pair 在当前序列里全部合并一次
- 继续，直到再也没有可用 merge

这正是“把训练阶段学到的 merges 按顺序应用回来”。

[`_encode_pretoken()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L268-L270) 是一个小封装：先 merge，再把 bytes token 查表变成 id。

[`_encode_ordinary_text()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L272-L274) 则是“对一整段普通文本重复这个过程”：先 regex 预分词，再逐个 pre-token 编码。

[`encode()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L276-L283) 把这些步骤真正串起来：
- 先按 special token 切段
- special token 直接映成单个 id
- 普通文本走 `_encode_ordinary_text`

所以它保证了 special token 永远不会被拆。

[`decode()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L344-L346) 则很直观：
- 先把每个 id 映回 bytes
- 把 bytes 拼起来
- 再 `decode("utf-8", errors="replace")`

这里的 `errors="replace"` 很重要，因为题目要求：如果 id 序列拼出来不是合法 UTF-8，要用替代字符 `U+FFFD`，而不是报错。

---

**第四部分：为什么 `encode_iterable()` 这么写**

[`encode_iterable()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L294-L342) 是整份代码里最“工程化”的部分。

难点是：如果你把大文件一块块读进来，不能直接对每块分别 `encode`，因为 chunk 边界可能正好切开：
- 一个 special token
- 一个单词
- 一个 `\n\n` 这样的空白 pre-token

这样结果就会和“整串一次性编码”不一致。

所以这里用了一个 `buffer`：
- [L297] 新 chunk 先接到 buffer 后面
- [L298-L304] 用 [`_longest_partial_special_suffix()`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py#L285-L292) 检查 buffer 末尾是不是某个 special token 的“前缀残片”
- 如果是，就先别动这部分，留到下个 chunk 再说

接着 [L310] 会把当前可处理部分再按 special token 切开。

如果当前末尾没有 special token 残片，那么最后一段普通文本 `final_text` 仍然可能是不完整的，所以 [L333-L339] 只会把它“除最后一个 pre-token 外”的部分立刻输出，把最后一个 pre-token 留在 buffer 里，等下一块拼上来再决定。

最后 [L341-L342]，等 iterable 真结束了，再把 buffer 里剩下的内容一次性 `encode` 掉。

你可以把它记成一句话：

`encode_iterable()` 的目标不是“尽快编码”，而是“流式地编码，但结果必须和整串 encode 完全一样”。

---

**你现在最应该抓住的 3 个核心点**

- BPE 训练的本质，是在 pre-token 内部统计“最常见的相邻 token 对”，不断合并。
- BPE 编码的本质，是把训练时学到的 merge 顺序重新按 rank 应用回去。
- `special token` 和 `encode_iterable` 这两部分，主要是在解决“边界不能乱跨”的问题。

如果你愿意，下一步我可以继续做第二个任务：拿 [`tokenizer.py`](/2022533145/CS190/assignment1-basics/cs336_basics/tokenizer.py) 里的一个具体输入字符串，手工带你走一遍 `encode()` 和 `train_bpe()` 的执行过程。