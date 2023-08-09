# NTK-ALiBi: Long Text Extrapolation of ALiBi Position Encoding through Interpolation

## Introduction

- To address the limited attention span of ALiBi position encoding in long texts, this paper proposes two interpolation methods based on ALiBi encoding: internal interpolation and NTK-ALiBi interpolation.
- Experiments show that without fine-tuning, the interpolation methods can effectively expand the attention span of ALiBi encoding and improve the task performance on long texts.

## ALiBi Position Encoding

- Main idea: Add a relative position penalty bias term to the attention score to achieve the effect of local attention and alleviate the problem of attention divergence in long texts.
  - Attention score: a_ij = q_i k_j - m_h |i - j|
  - Bias coefficient: m_h = 1 / 2^(8h/H), where H is the number of attention heads, and h is the h-th head
- Limitation: Local attention is limited by training length, and the attention span is limited, which cannot handle long-text inference well.

## Internal Interpolation

- Internal interpolation: Assume that the inference length is a times the training length, and simply reduce the position by a times to achieve position internal interpolation. This is equivalent to reducing the bias coefficient m by a times, i.e., expanding the attention span by a times.
  - a_ij = q_i k_j - (m_h/a) |i - j|
- Limitation: The internal interpolation method will uniformly reduce the m values under different field of view sizes, resulting in a loss of field of view resolution.

## NTK-ALiBi Interpolation

- Frequency domain: RoPE encoding or ALiBi encoding share the commonality of encoding position space into frequency domain, where the coefficients of trigonometric functions (RoPE) or bias terms (ALiBi) are the frequency domain values.
- NTK-RoPE interpolation: The improvement of NTK-RoPE position encoding lies in achieving frequency domain scaling (low frequency) while maintaining resolution (high frequency), thus realizing position space interpolation.
- NTK-ALiBi interpolation: Inspired by NTK encoding, we can also scale the frequency domain space of ALiBi to achieve NTK-ALiBi position interpolation. The improved bias term coefficient is:

  - m_h = 1 / {2^(8h/H) * a^((h-1)/(H-1))}
  - Let b = a^(1/(H-1)), then: m_h = b / {2^(8/H) * b}^h
  - After NTK improvement, the effect of unchanged high-frequency resolution and enlarged low-frequency field of view can be achieved
- Explanation: The formula of NTK-ALiBi may seem a bit difficult to understand, but the core idea is the same as Su Jianlin's [“High-frequency extrapolation, low-frequency interpolation”](https://kexue.fm/archives/9675). Consider the following two cases:

  - When h=1, the field of view is small, which is a high-frequency situation. m_h = 1 / 2^(8/H), which is the same as the original bias coefficient, equivalent to direct extrapolation, so it is high-frequency extrapolation. The high-frequency field of view resolution remains unchanged.
  - When h=H, the field of view is large, which is a low-frequency situation. m_h = 1 / {2^8 * a}, which reduces the original bias by a times, equivalent to position interpolation, so it is low-frequency interpolation. The low-frequency field of view increases by a times.
- Code: Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L86

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

## Experiment
### LongEval
- Dataset: LongEval, including topics and lines tasks, with 50 long texts for each task at different input lengths
- Baseline model: bigscience/bloom-1b7, pre-training length 2048
- Experiment result: topics 5, inference length about 3K

| Method                       | Accuracy/% |
| ---------------------------- | ---------- |
| Original ALiBi Encoding      | 2          |
| Internal Interpolation, a=2  | 74         |
| NTK-ALiBi Interpolation, a=2 | 96         |

- Experiment result: lines 200, inference length about 5K

| Method                       | Accuracy/% |
| ---------------------------- | ---------- |
| Original ALiBi Encoding      | 0          |
| Internal Interpolation, a=2  | 30         |
| NTK-ALiBi Interpolation, a=2 | 40         |

- Result analysis: After interpolation of ALiBi encoding, without any fine-tuning, significant performance improvements were achieved on extrapolation lengths of about twice the training length (3~5K). Adding NTK-ALiBi interpolation further improved the performance.
- Limitations: Due to resource and time constraints, this paper did not conduct experiments on more tasks and scaling factors. Discussions and supplements are welcome.

### LongBench
- Dataset: LongBench
    - TREC: Few-shot text classification task, inference length about 5K
- Baseline model: bigscience/1b7, pre-training length 2048
- Experimental results: TREC

|  Method  |	  Accuracy/%  |
| ------ | ----- |
| Bloom-1B7, original ALiBi encoding	| 13.0 |
| Bloom-1B7, NTK-ALiBi interpolation, a=4 |	61.5 |
| \*GPT-3.5-Turbo-16k |	68.0 |
| \*Llama2-7B-chat-4k |	60.5 |
| \*ChatGLM2-6B |	44.0 |
| \*ChatGLM2-6B-32k |	62.0 |

Note: *Results are taken from https://github.com/THUDM/LongBench

- Result analysis:
    - After NTK interpolation of ALiBi encoding, without any fine-tuning, a significant improvement is achieved in the TREC text classification task from 13.0% to 61.5%.
    - The Bloom-1B7 model with NTK-ALiBi encoding has a significantly better TREC text classification accuracy than ChatGLM2-6B, and is close to the performance of Llama2-7B-chat-4k and ChatGLM2-6B-32k.


## References

- NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
- Transformer Upgrade Path: 10, RoPE is a β-based encoding: https://kexue.fm/archives/9675
- How Long Can Open-Source LLMs Truly Promise on Context Length?: https://lmsys.org/blog/2023-06-29-longchat/
- LongEval: https://github.com/DachengLi1/LongChat
- Bloom-1B7: https://huggingface.co/bigscience/bloom-1b7
- Press. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR. 2022. (ALiBi.)
- Chen. Extending Context Window of Large Language Models via Positional Interpolation. 2023. (Meta.)


## Citation
If you find this repo to be useful, plese cite:
```
@misc{NtkAlibi2023,
    title = {NTK-ALiBi: Long Text Extrapolation of ALiBi Position Encoding through Interpolation},
    url = {https://github.com/keezen/ntk_alibi},
    author = { },
    month = {August},
    year = {2023}
}
```
