# Part B Transformer Language Model Architecture 整体结构

[`model.py`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L1) 可以分成 4 层来看：

1. 最基础的数学/初始化工具：`softmax`、`silu`、初始化标准差
2. 基础模块：`Linear`、`Embedding`、`RMSNorm`、`SwiGLU`
3. 注意力相关模块：`RotaryPositionalEmbedding`、`scaled_dot_product_attention`、`MultiHeadSelfAttention`
4. 大模块组装：`TransformerBlock`、`TransformerLM`

你可以把它理解成一条自下而上的搭积木过程：

`Linear/Embedding/RMSNorm -> FFN/Attention -> Block -> 整个语言模型`

---

**第一部分：最基础的工具函数**

[`_trunc_normal_std()`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L10-L11) 是一个很小的辅助函数。

它做的事是按照题目给的初始化规则，计算线性层权重初始化的标准差：

```python
std = sqrt(2 / (in_features + out_features))
```

为什么这样写？
因为题目要求线性层权重用：

```text
N(0, 2 / (din + dout))
```

正态分布的“方差”是 `2 / (din + dout)`，所以标准差就是它开根号。

---

[`softmax()`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L14-L17) 是数值稳定版 softmax。

代码逻辑是：

```python
shifted = x - x.amax(dim=dim, keepdim=True)
exp_shifted = torch.exp(shifted)
return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)
```

这里最关键的是第一步减去最大值。  
原因是如果某个值太大，比如 `1000`，直接 `exp(1000)` 会溢出成 `inf`。  
但 softmax 有一个性质：

```text
softmax(x) == softmax(x - c)
```

所以减去最大值以后，最大的那个元素会变成 0，指数不会爆炸。

这是作业里 `softmax` 题的核心思想。

---

[`silu()`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L20-L21) 实现的是 SiLU 激活函数：

```python
x * sigmoid(x)
```

它比 ReLU 更平滑，也是 Lec5 和题目里要求在 SwiGLU 中使用的激活函数。

---

**第二部分：基础模块**

### 1. Linear

[`Linear`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L24-L40) 是最基本的线性层。

初始化部分在 [`__init__`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L25-L37)：

- [L35] 建立参数 `weight`
- 它的形状是 `(out_features, in_features)`
- [L37] 用 `torch.nn.init.trunc_normal_` 做截断正态初始化

这里一个非常重要的点是：  
我们把权重存成 `(out, in)`，这和 PyTorch `nn.Linear.weight` 的存储方式一致，也和测试里的权重 shape 一致。

前向在 [`forward()`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L39-L40)：

```python
torch.einsum("... i, o i -> ... o", x, self.weight)
```

这个式子可以读成：

- 输入 `x` 的最后一维是 `i`
- 权重的维度是 `o i`
- 输出最后一维变成 `o`

所以它本质上就是对输入最后一维做线性变换，而且前面的 `...` 任意保留。

这意味着它既可以处理：
- `(batch, seq, d_model)`
- 也可以处理 `(batch, head, seq, d_head)`
- 甚至更多 batch-like 维度

这是 Transformer 里很重要的设计习惯。

---

### 2. Embedding

[`Embedding`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L43-L58) 比 `Linear` 更简单。

初始化时：
- [L54] 创建 `weight`
- shape 是 `(num_embeddings, embedding_dim)`

你可以把它想成一个“词表表格”：
- 每一行对应一个 token id
- 每一行是一条 `d_model` 维向量

前向在 [L57-L58]：

```python
return self.weight[token_ids]
```

也就是直接用整数下标索引。

如果 `token_ids` 的形状是 `(batch, seq)`，输出自然就会变成：

```python
(batch, seq, d_model)
```

这正是 Transformer 输入嵌入需要的形状。

---

### 3. RMSNorm

[`RMSNorm`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L61-L78) 是 pre-norm Transformer 的关键组件。

初始化时：
- [L71] `weight` 初始化成全 1
- 它对应公式中的可学习 gain 参数 `g`

前向逻辑：

1. [L74-L75] 保存原 dtype，并把输入转成 `float32`
2. [L76] 计算 RMS：
   ```python
   sqrt(mean(x^2) + eps)
   ```
3. [L77] 做归一化并乘以可学习 `weight`
4. [L78] 再转回原 dtype

为什么一定要先转 `float32`？
因为如果输入是 `float16` / `bfloat16`，平方以后更容易溢出或精度不够。题目明确要求这里要 upcast。

还有一点要注意：  
RMSNorm 只对最后一维归一化，也就是对每个 token 的 hidden vector 单独归一化，而不是跨 batch 或跨 sequence。

---

### 4. SwiGLU

[`SwiGLU`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L81-L95) 是前馈网络 FFN 的核心。

初始化时建了三个线性层：
- [L90] `w1`: `d_model -> d_ff`
- [L91] `w2`: `d_ff -> d_model`
- [L92] `w3`: `d_model -> d_ff`

前向只有一行：

```python
return self.w2(silu(self.w1(x)) * self.w3(x))
```

对应作业公式：

```text
W2( SiLU(W1x) ⊙ W3x )
```

你可以这么理解：
- `w1(x)` 走“激活分支”
- `w3(x)` 走“门控分支”
- 两者做逐元素相乘
- 再通过 `w2` 投回 `d_model`

这比传统 `Linear -> ReLU -> Linear` 更强一些，也是现代 LLM 常见做法。

---

**第三部分：RoPE 和注意力**

### 5. RotaryPositionalEmbedding

[`RotaryPositionalEmbedding`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L98-L132) 实现了 RoPE。

#### 初始化阶段

[L107-L108] 先检查 `d_k` 必须是偶数。  
因为 RoPE 是把最后一维两两分组旋转的，比如：

```text
(x0, x1), (x2, x3), (x4, x5), ...
```

接着：

- [L113] 生成位置 `0, 1, 2, ..., max_seq_len-1`
- [L114] 取偶数维的索引 `0, 2, 4, ...`
- [L115] 计算每对维度对应的频率因子 `inv_freq`
- [L116] 用外积算出所有位置、所有频率下的角度矩阵 `angles`
- [L117-L118] 预计算并缓存 `cos` 和 `sin`

这里缓存成 buffer，而不是 Parameter，因为：
- 它们不是可学习参数
- 只是固定的查找表

这和题目要求一致。

#### 前向阶段

在 [`forward()`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L120-L132)：

- [L122-L123] 根据 `token_positions` 取出当前位置的 `cos` / `sin`
- [L124-L126] 用 `unsqueeze` 把维度补齐，让它们能和任意 batch-like 维度广播
- [L128-L129] 把输入拆成偶数位和奇数位
- [L130-L131] 套用二维旋转公式：
  - `x' = x cos - y sin`
  - `y' = x sin + y cos`
- [L132] 再把它们拼回原来的最后一维

所以这段代码本质上是在“模拟很多个 2x2 旋转矩阵”，但没有真的构造大矩阵，这样更高效。

---

### 6. Scaled Dot-Product Attention

[`scaled_dot_product_attention()`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L135-L147) 对应作业里的 Attention 公式。

步骤非常标准：

1. [L141-L142] 计算打分矩阵：
   ```python
   scores = QK^T / sqrt(d_k)
   ```
2. [L143-L145] 如果有 mask，把不允许注意的位置变成 `-inf`
3. [L146] 对最后一维做 softmax，得到注意力概率
4. [L147] 用概率加权求和 `V`

为什么 mask 用 `-inf`？
因为 softmax 后：

```text
exp(-inf) = 0
```

所以这些位置的注意力概率就会变成 0。

注意这里的 `mask=True` 表示“允许注意”，`False` 表示“屏蔽”，和题面保持一致。

---

### 7. MultiHeadSelfAttention

[`MultiHeadSelfAttention`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L150-L196) 是整个模型最核心的模块之一。

#### 初始化

- [L162-L163] 检查 `d_model % num_heads == 0`
- [L166] 计算每个 head 的维度 `d_head`

然后建立 4 个线性层：
- [L169] `q_proj`
- [L170] `k_proj`
- [L171] `v_proj`
- [L172] `output_proj`

这里每个投影都是 `d_model -> d_model`。  
原因是多头注意力其实是：
- 先把 `d_model` 一次性投影成 `num_heads * d_head`
- 再 reshape 成多个 head

RoPE 部分：
- [L174-L178] 如果 `use_rope=True`，就为每个 head 的维度创建一个 `RotaryPositionalEmbedding`

也就是说 RoPE 不是作用在整个 `d_model` 上，而是作用在每个 head 的 `d_head` 上，这也符合题目要求。

#### 前向

在 [`forward()`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L180-L196)：

1. [L181-L183] 确定序列长度和默认 token positions
2. [L185-L187] 分别计算 `q/k/v`，然后 reshape 成：
   ```python
   (..., head, seq, d_head)
   ```
   这一步是多头注意力的关键
3. [L189-L191] 如果启用 RoPE，只对 `q` 和 `k` 做 RoPE，`v` 不动
4. [L193] 构造 causal mask，下三角为 True
5. [L194] 调用前面写好的 `scaled_dot_product_attention`
6. [L195] 把多个 head 拼回：
   ```python
   (..., seq, head * d_head)
   ```
7. [L196] 再通过输出投影 `output_proj`

这里你要特别记住一句话：

**多头注意力不是写了很多套 attention，而是先 reshape 出 head 维，再把 head 当成 batch-like 维统一并行计算。**

这就是为什么这一版实现既简洁又和 Lec5 的 batch-like 维思路一致。

---

**第四部分：Block 和整个 Transformer**

### 8. TransformerBlock

[`TransformerBlock`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L199-L227) 是标准 pre-norm block。

初始化时有 4 个子模块：
- [L211] `ln1`
- [L212-L220] `attn`
- [L221] `ln2`
- [L222] `ffn`

前向在 [L224-L227]：

```python
x = x + self.attn(self.ln1(x), token_positions=token_positions)
x = x + self.ffn(self.ln2(x))
return x
```

这就是 pre-norm 的标准结构：

1. 先归一化，再做 attention，再 residual add
2. 再归一化，再做 FFN，再 residual add

你可以把它和 Lec5 的 block 图直接对应起来：
- 第一半：RMSNorm -> MHA -> Add
- 第二半：RMSNorm -> FFN -> Add

这里 residual 连接的重要意义是：  
让原始信息有一条“直通路径”，更容易训练深层 Transformer。

---

### 9. TransformerLM

[`TransformerLM`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L230-L269) 是整模型的组装。

初始化时：

- [L245] `token_embeddings`
- [L246-L259] `layers = ModuleList([...])`
- [L260] `ln_final`
- [L261] `lm_head`

这和作业 Figure 1 是完全一致的。

前向在 [L263-L269]：

1. [L264] 为当前输入长度生成 `token_positions`
2. [L265] token ids -> embeddings
3. [L266-L267] 依次通过所有 Transformer blocks
4. [L268] 最后再做一次 `ln_final`
5. [L269] 用 `lm_head` 投到词表维度，输出 logits

这里最后返回的是 **logits**，不是 softmax 概率。  
这是正确的，因为：
- 训练时通常直接把 logits 喂给 cross-entropy
- 推理时也通常先拿 logits，再自己决定 greedy / sampling

---

**把整份代码串起来看**

如果输入是：

```python
in_indices.shape == (batch, seq)
```

那么整条前向路径是：

1. `Embedding`  
   `(batch, seq) -> (batch, seq, d_model)`

2. 每个 `TransformerBlock`  
   `(batch, seq, d_model) -> (batch, seq, d_model)`

3. `ln_final`  
   形状不变

4. `lm_head`  
   `(batch, seq, d_model) -> (batch, seq, vocab_size)`

所以最后每个位置都会得到一个“下一个 token 的词表打分”。

---

**这份实现里最值得你重点理解的 5 个点**

- [`Linear.forward`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L39-L40)：为什么只对最后一维做线性变换，就能自动支持 batch、seq、head 等前置维度。
- [`RMSNorm.forward`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L73-L78)：为什么要先转 `float32`。
- [`RotaryPositionalEmbedding.forward`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L120-L132)：RoPE 本质上是在最后一维上“两两旋转”。
- [`MultiHeadSelfAttention.forward`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L180-L196)：多头注意力的本质是 reshape + 并行 attention。
- [`TransformerBlock.forward`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L224-L227)：pre-norm block 的残差结构。

如果你愿意，下一步我可以继续用一个非常具体的小例子，手工带你走一遍 [`MultiHeadSelfAttention.forward`](/2022533145/CS190/assignment1-basics/cs336_basics/model.py#L180-L196) 的张量形状变化。