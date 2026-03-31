# Training a Transformer LM, Training loop, Generating text 整体思路

这三个文件其实对应 Transformer 训练和推理的三条主线：

- [`training.py`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py)：训练时需要的“公共工具”
- [`optim.py`](/2022533145/CS190/assignment1-basics/cs336_basics/optim.py)：参数怎么更新，也就是 AdamW
- [`generation.py`](/2022533145/CS190/assignment1-basics/cs336_basics/generation.py)：模型训练好以后，怎么一步一步生成文本

你可以把它们分别理解成：

- `training.py` 解决“怎么算 loss、怎么取 batch、怎么存档”
- `optim.py` 解决“loss.backward() 之后，参数怎么改”
- `generation.py` 解决“给一个 prompt，怎么往后续写”

---

## `training.py`

### 1. `cross_entropy`

看 [`cross_entropy()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L13)。

它实现的是分类任务里最常见的交叉熵损失。  
在语言模型里，每个位置都会输出一个长度为 `vocab_size` 的 logits 向量，表示“下一个 token 是每个词表项的分数”。

代码是：

```python
log_norm = torch.logsumexp(logits, dim=-1)
target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
return (log_norm - target_logits).mean()
```

这个公式其实就是：

```text
CrossEntropy = -log p(correct class)
             = log(sum(exp(logits))) - logits[target]
```

为什么不用先 `softmax` 再 `log`？
因为那样数值不稳定。  
如果 logits 很大，比如 1000，`exp(1000)` 会直接溢出。  
`torch.logsumexp` 是 PyTorch 提供的稳定写法，本质上等价于先减最大值再算。

这一段你要抓住的重点是：

- logits 是“未归一化分数”
- cross entropy 本质是在拿“正确类别的 log probability”
- `logsumexp` 是数值稳定的关键

---

### 2. `get_batch`

看 [`get_batch()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L19)。

这个函数负责从一整条 token 序列里，随机抽出训练样本。

假设数据是：

```text
[10, 11, 12, 13, 14, 15, 16, ...]
```

如果 `context_length = 4`，某次采样起点是 `i = 2`，那就会得到：

```text
x = [12, 13, 14, 15]
y = [13, 14, 15, 16]
```

也就是：
- `x` 是输入
- `y` 是向右平移一位后的目标

这正是自回归语言模型训练的基本形式。

代码分成几步：

- `torch.as_tensor(dataset, dtype=torch.long)`
  - 把 numpy 数组或 tensor 统一变成 `torch.long`
- `max_start = data.shape[0] - context_length`
  - 计算合法起点数量
- `starts = torch.randint(0, max_start, (batch_size,))`
  - 随机选 `batch_size` 个起点
- `offsets = torch.arange(context_length)`
  - 构造 `[0, 1, 2, ..., context_length-1]`
- 用广播一次性拿出整批 `x` 和 `y`

这里最值得学的是：  
我们没有用 Python `for` 循环逐个样本切片，而是用张量广播一次性取整批数据，这样更快也更符合 PyTorch 风格。

---

### 3. `clip_gradients`

看 [`clip_gradients()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L34)。

梯度裁剪的作用是防止梯度过大，导致训练突然爆炸。  
Lec6 里也提到过，尤其在深网络或训练初期，这一步很常见。

逻辑是：

1. 先收集所有有梯度的参数：
   ```python
   grads = [param.grad for param in parameters if param.grad is not None]
   ```
2. 计算全局梯度 L2 norm：
   ```python
   total_norm = sqrt(sum(sum(grad^2)))
   ```
3. 算出缩放比例：
   ```python
   clip_coef = max_l2_norm / (total_norm + 1e-6)
   ```
4. 如果已经没超，就不动；如果超了，就把所有梯度整体乘一个比例

这里的思想非常重要：

- 不是把每个参数单独裁到某个大小
- 而是把“所有梯度当成一个大向量”，控制这个大向量的总长度

这和 `torch.nn.utils.clip_grad_norm_` 的思路一致。

`1e-6` 是为了防止分母为 0。

---

### 4. `get_lr_cosine_schedule`

看 [`get_lr_cosine_schedule()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L48)。

它实现的是：

- 前面一段：linear warmup
- 后面一段：cosine decay
- 最后：保持在最小学习率

分三段看：

#### 第一段：warmup

```python
if it < warmup_iters:
    return max_learning_rate * it / warmup_iters
```

也就是从 0 线性升到 `max_learning_rate`。

为什么要 warmup？
因为一开始参数还是随机的，如果学习率太大，训练很容易不稳定。  
warmup 就像“先慢慢起步”。

#### 第二段：cosine decay

```python
decay_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
```

当 `decay_progress` 从 0 走到 1 时，`cos(pi * progress)` 会从 1 走到 -1，  
所以整个系数会从 1 平滑降到 0。

#### 第三段：收尾固定

```python
if it >= cosine_cycle_iters:
    return min_learning_rate
```

也就是训练后期不再继续降低，稳定在一个小学习率。

---

### 5. `save_checkpoint` 和 `load_checkpoint`

看 [`save_checkpoint()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L64) 和 [`load_checkpoint()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L78)。

`save_checkpoint` 存了三样东西：

- `model_state_dict`
- `optimizer_state_dict`
- `iteration`

为什么 optimizer 也要存？
因为 AdamW 不只是参数值重要，它内部还有动量状态 `m`、`v`。  
如果只恢复模型参数，不恢复优化器状态，继续训练时轨迹就变了。

`load_checkpoint` 做的就是反过来恢复：
- `model.load_state_dict(...)`
- `optimizer.load_state_dict(...)`
- 返回之前训练到了第几步

这是训练可中断恢复的标准做法。

---

## `optim.py`

### 1. `AdamW` 类的整体作用

看 [`AdamW`](/2022533145/CS190/assignment1-basics/cs336_basics/optim.py#L9)。

它继承自 `torch.optim.Optimizer`，说明我们是在自己实现一个 PyTorch 优化器。

AdamW 和普通 SGD 的区别在于：

- SGD：直接沿梯度方向走
- AdamW：会维护一阶矩和二阶矩，自动调节不同参数的更新步长

直觉上可以理解成：

- `m`：梯度的“平均方向”
- `v`：梯度大小的“平均强度”

这样更新会更平滑，也更自适应。

---

### 2. `__init__`

看 [`__init__()`](/2022533145/CS190/assignment1-basics/cs336_basics/optim.py#L10-L29)。

前半部分是在检查超参数是否合法：

- `lr >= 0`
- `eps >= 0`
- `0 <= beta1 < 1`
- `0 <= beta2 < 1`
- `weight_decay >= 0`

后半部分：

```python
super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
```

这是 Optimizer 基类的标准写法。  
它会把这些默认超参数放进 `param_groups` 里，后面 `step()` 时就能统一访问。

---

### 3. `step`

核心在 [`step()`](/2022533145/CS190/assignment1-basics/cs336_basics/optim.py#L31-L70)。

这是优化器最重要的函数：每次训练循环调用一次 `optimizer.step()`，就会真正更新参数。

#### 第一步：处理 closure

```python
loss = None
if closure is not None:
    with torch.enable_grad():
        loss = closure()
```

这是 PyTorch Optimizer 的通用接口。  
这次作业里基本不会用到 `closure`，但保留这个接口更规范。

#### 第二步：遍历 parameter groups

```python
for group in self.param_groups:
```

这是因为 PyTorch 允许不同参数组用不同学习率、weight decay 等超参数。

#### 第三步：遍历每个参数

```python
for param in group["params"]:
    grad = param.grad
    if grad is None:
        continue
```

如果某个参数这次没有梯度，就跳过。

然后检查稀疏梯度：
```python
if grad.is_sparse:
    raise RuntimeError(...)
```

因为这版 AdamW 没有实现 sparse 版本。

---

### 4. 初始化状态 `m`、`v`、`step`

这部分在 [`step()`](/2022533145/CS190/assignment1-basics/cs336_basics/optim.py#L49-L54)。

第一次见到某个参数时，它在 `self.state[param]` 里还没有历史状态，所以要初始化：

- `step = 0`
- `m = zeros_like(param)`
- `v = zeros_like(param)`

之后每次更新都会重复使用这些状态。

---

### 5. 更新一阶矩、二阶矩

核心公式：

```python
m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
```

对应数学上：

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

含义是：

- `m` 记录梯度的指数滑动平均
- `v` 记录梯度平方的指数滑动平均

这样能过滤噪声，不会被单次梯度剧烈波动带偏。

---

### 6. bias correction

代码：

```python
m_hat = m / (1.0 - beta1**step)
v_hat = v / (1.0 - beta2**step)
```

为什么要做这个？
因为刚开始时 `m` 和 `v` 都是从 0 启动的，前几步会偏小。  
bias correction 就是在补偿这种“冷启动偏差”。

这是 Adam 系列算法里非常关键的一步。

---

### 7. 自适应更新

代码：

```python
param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
```

可以理解为：

```text
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

直觉上：
- 分子 `m_hat` 是更新方向
- 分母 `sqrt(v_hat)` 会根据梯度历史大小自适应缩放步长

如果某个参数过去梯度一直很大，它的更新会被压小；
如果某个参数梯度一直较小，它还能保持一定步长。

---

### 8. decoupled weight decay

最后这句：

```python
param.mul_(1.0 - lr * weight_decay)
```

这就是 AdamW 和“Adam + L2 regularization”最核心的区别。

它不是把 `weight_decay * param` 混进梯度里，而是单独做一次参数衰减。  
所以叫 decoupled weight decay。

直观理解：

- Adam 的自适应更新负责“朝哪里走”
- weight decay 负责“把参数整体往 0 收一点”

这两件事分开做，行为更稳定，也更符合 AdamW 的设计。

---

## `generation.py`

### 1. `sample_next_token`

看 [`sample_next_token()`](/2022533145/CS190/assignment1-basics/cs336_basics/generation.py#L9)。

它的任务是：

- 模型已经输出整段序列的 logits
- 我们只关心最后一个位置
- 决定“下一个 token 采样谁”

第一句：

```python
next_logits = logits[..., -1, :]
```

就是只取最后一个时间步。

因为自回归生成时，每次只预测“当前序列的下一个 token”。

---

### 2. `temperature == 0` 时 greedy decoding

```python
if temperature == 0:
    return torch.argmax(next_logits, dim=-1)
```

这表示完全不随机，直接选分数最大的 token。

这就是 greedy decoding。

优点是确定性强；  
缺点是容易重复、保守、模式单一。

---

### 3. 温度缩放

```python
probs = softmax(next_logits / temperature, dim=-1)
```

如果 `temperature > 1`：
- logits 被压平
- 概率更平均
- 生成更随机

如果 `0 < temperature < 1`：
- logits 差距被放大
- 概率更尖锐
- 生成更保守

所以 temperature 本质上是在调“随机程度”。

---

### 4. `top_p` nucleus sampling

如果 `top_p < 1.0`，就会进入 nucleus sampling。

代码流程是：

1. 按概率降序排序
2. 做累计概率 `cumulative_probs`
3. 保留累计概率刚达到 `top_p` 的最小前缀
4. 其余位置概率清零
5. 重新归一化
6. 再随机采样

为什么要这样做？
因为有些时候长尾 token 概率极低，保留它们只会增加噪声。  
top-p 会让采样只在“当前最有希望的一小团 token”里进行。

它比 top-k 更自适应，因为保留的 token 数量不是固定的，而是由概率分布决定的。

---

### 5. `generate`

看 [`generate()`](/2022533145/CS190/assignment1-basics/cs336_basics/generation.py#L33)。

这是完整的自回归生成循环。

#### 开头：统一形状

```python
generated = prompt_ids.clone()
if generated.ndim == 1:
    generated = generated.unsqueeze(0)
```

这一步是为了让输入统一成 batch 形式。  
如果你只给了一条 prompt，比如 shape 是 `(seq,)`，就把它变成 `(1, seq)`。

---

#### 每轮生成一个 token

循环里最重要的流程是：

1. 准备当前上下文 `model_input`
2. 送进模型得到 logits
3. 用 `sample_next_token` 采样一个新 token
4. 把它拼到当前序列后面

对应代码：

```python
logits = model(model_input)
next_token = sample_next_token(...).unsqueeze(-1)
generated = torch.cat((generated, next_token), dim=-1)
```

这就是标准 autoregressive decoding。

---

#### 为什么要截断到 `context_length`

```python
if context_length is not None and model_input.shape[-1] > context_length:
    model_input = model_input[..., -context_length:]
```

因为模型只能处理固定窗口大小。  
如果已经生成得比窗口更长，就只能保留最后 `context_length` 个 token。

这和真实 LLM 推理时的“滑动窗口”思路一致。

---

#### 为什么可以遇到 EOS 提前停

```python
if eos_token_id is not None and torch.all(next_token == eos_token_id):
    break
```

如果模型已经生成了结束标记 `<eos>`，通常就不需要继续往后采样了。

这样生成会更自然，也避免无意义地继续输出。

---

## 把三份代码串起来看

训练时的主流程是：

1. `get_batch()` 从数据里抽 `x, y`
2. 模型前向得到 `logits`
3. `cross_entropy()` 算 loss
4. `loss.backward()` 算梯度
5. `clip_gradients()` 防止梯度爆炸
6. `AdamW.step()` 更新参数
7. `save_checkpoint()` 周期性保存状态

推理时的主流程是：

1. tokenizer 把 prompt 变成 token ids
2. `generate()` 把 prompt 喂给模型
3. 每轮调用 `sample_next_token()` 采样下一个 token
4. 最后 tokenizer 再把 token ids 解码回文本

---

## 这三份代码里你最应该重点理解的 6 个点

- [`cross_entropy()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L13)：为什么用 `logsumexp`，而不是手写 `softmax + log`
- [`get_batch()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L19)：为什么语言模型标签是输入右移一位
- [`clip_gradients()`](/2022533145/CS190/assignment1-basics/cs336_basics/training.py#L34)：为什么裁的是“全局梯度范数”
- [`AdamW.step()`](/2022533145/CS190/assignment1-basics/cs336_basics/optim.py#L31)：`m`、`v`、bias correction、adaptive step 分别在做什么
- [`sample_next_token()`](/2022533145/CS190/assignment1-basics/cs336_basics/generation.py#L9)：temperature 和 top-p 分别控制什么
- [`generate()`](/2022533145/CS190/assignment1-basics/cs336_basics/generation.py#L33)：为什么生成时每次只追加 1 个 token

如果你愿意，下一步我可以继续用一个非常具体的小例子，手工带你走一遍：
- 一次 `cross_entropy` 是怎么从 logits 算出 loss 的
- 或者一轮 `generate()` 是怎么一步一步把 prompt 扩展成一句完整文本的