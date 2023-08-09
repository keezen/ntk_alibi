# NTK-ALiBi：通过插值实现ALiBi编码的长文本外推

[English version](readme_en.md)

## 简介
- 针对ALiBi位置编码在长文本中注意力视野有限的问题，本文提出了基于ALiBi编码的两种插值方法：内插值和NTK-ALiBi插值。
- 实验表明：无需微调情况下，插值方法能够有效扩大ALiBi编码的注意力视野，提高长文本上的任务效果。


## ALiBi位置编码
- 主要思路：在注意力分数中加上相对位置惩罚偏置项，实现局部注意力的效果，缓解长文本下注意力发散的问题。
    - 注意力分数：a_ij = q_i k_j - m_h |i - j|
    - 偏置系数：m_h = 1 / 2^(8h/H)，其中H为注意力头数，h为第h个头
- 不足：局部注意力受限于训练长度，注意力视野有限，不能很好处理长文本的推理。


## 内插值
- 内插值：假设推理长度为训练长度的a倍，简单对位置缩减a倍，实现位置内插值。等价于将偏置系数m缩小a倍，即将注意力视野扩大了a倍。
    - a_ij = q_i k_j - (m_h/a) |i - j|
- 不足：内插值方法会将不同视野大小下的m值统一缩减，会损失视野分辨率。


## NTK-ALiBi插值
- 频域：RoPE编码或ALiBi编码其共同点，都是将位置空间编码为频域空间，其中三角函数（RoPE）或偏置项（ALiBi）的系数，即为频域值。
- NTK-RoPE插值：NTK-RoPE位置编码的改进，在于保持分辨率的情况下（高频），实现了频域空间缩放（低频），从而实现位置空间的插值。
- NTK-ALiBi插值：受NTK编码的启发，我们也可以对ALiBi的频域空间进行缩放，实现NTK-ALiBi的位置插值。改进后的偏置项系数为：
    - m_h = 1 / {2^(8h/H) * a^((h-1)/(H-1))}
    - 令b = a^(1/(H-1)), 则有：m_h = b / {2^(8/H) * b}^h
    - NTK改进后可以实现高频分辨率不变，低频视野放大的效果

- 解释：NTK-ALiBi的公式看起来可能有些难懂，但核心思想与苏建林大佬所说的[“高频外推，低频内插”](https://kexue.fm/archives/9675)相同。下面从两种情况考虑：
    - h=1时，视野较小，为高频情况。m_h = 1 / 2^(8/H)，与原始偏置系数相同，相当于直接外推，因此是高频外推。高频视野分辨率不变。
    - h=H时，视野较大，为低频情况。m_h = 1 / {2^8 * a}，在原始偏置基础上缩减了a倍，等价于对位置进行了内插值，因此是低频内插。低频视野变大a倍。

- 代码：修改自https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L86
```python
def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """Psuedo code for NTK-ALiBi."""
    batch_size, seq_length = attention_mask.shape
    a = 2.0   # ntk step 1: scale ratio a = inference_length / pretraining_length
    scale = a ** (1.0 / (num_heads-1))  # ntk step 2: coefficient b, for computation convenience
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    base /= scale  # ntk step 3: divide b to alibi base
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    slopes *= scale  # ntk step 4: fix alibi bias m_h by multiplying b

    if closest_power_of_2 != num_heads:  # todo: fix ntk-alibi when num_heads is not power of 2
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
```


## 实验记录
### LongEval
- 数据集：LongEval，包含topics和lines两个任务，针对不同输入长度各有50条长文本
- 基线模型：bigscience/bloom-1b7，预训练长度2048
- 实验结果：topics 5，推理长度约3K

| 方法 |	准确率/% |
| ----- | ----- |
| 原始ALiBi编码 |	2 |
| 内插值, a=2 |	74 |
| NTK-ALiBi插值, a=2 |	96 |

- 实验结果：lines 200，推理长度约5K

| 方法 |	准确率/% |
| ----- | ----- |
|原始ALiBi编码|	0|
|内插值, a=2|	30|
|NTK-ALiBi插值, a=2|	40|

- 结果分析：ALiBi编码进行插值后，无须进行任何微调，在大约两倍训练长度的外推长度（3～5K）上，均取得了显著的效果提升。加上NTK-ALiBi插值后，效果进一步提升。
- 不足：受限于资源和时间，本文并未在更多任务和缩放系数上进行实验，欢迎讨论和补充。

### LongBench
- 数据集：LongBench
    - TREC：小样本文本分类任务，推理长度约5K
- 基线模型：bigscience/1b7，预训练长度2048
- 实验结果：TREC

| 方法 |	准确率/% |
| ----- | ----- |
| Bloom-1B7, 原始ALiBi编码	| 13.0 |
| Bloom-1B7, NTK-ALiBi插值, a=4 |	61.5 |
| \*GPT-3.5-Turbo-16k |	68.0 |
| \*Llama2-7B-chat-4k |	60.5 |
| \*ChatGLM2-6B |	44.0 |
| \*ChatGLM2-6B-32k |	62.0 |

注：*结果摘自https://github.com/THUDM/LongBench

- 结果分析：
    - ALiBi编码进行NTK插值后，无须进行任何微调，在TREC文本分类任务上取得显著提升13.0%->61.5%。
    - NTK-ALiBi编码后的Bloom-1B7模型，TREC文本分类准确率明显好于ChatGLM2-6B，与Llama2-7B-chat-4k和ChatGLM2-6B-32k效果接近。


## 参考文献
- NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
- Transformer升级之路：10、RoPE是一种β进制编码: https://kexue.fm/archives/9675
- How Long Can Open-Source LLMs Truly Promise on Context Length?: https://lmsys.org/blog/2023-06-29-longchat/
- LongEval: https://github.com/DachengLi1/LongChat
- Bloom-1B7: https://huggingface.co/bigscience/bloom-1b7
- Press. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR. 2022. (ALiBi.)
- Chen. Extending Context Window of Large Language Models via Positional Interpolation. 2023. (Meta.)


## 引用
欢迎转载和引用，但请指明出处和链接：
```
@misc{NtkAlibi2023,
    title = {NTK-ALiBi：通过插值实现大模型ALiBi位置编码的长文本外推},
    url = {https://github.com/keezen/ntk_alibi},
    author = { },
    month = {August},
    year = {2023}
}
```
