# FastForwardAttention
FastForwardAttention (FFA) is a pytorch extention for efficient exact iterative attention for natural language generation. It can easily replace iterative attention in huggingface models and improve throughput by 1.96x (see results below) and is especially useful for long sequence generation. It's built by performing gpu utilization optimizations on top of [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)'s implementation for faster inference.

## Install
Run 

```
python setup.py install
```

## Use
FFA handels iterative attention for transformer decoder. When generating tokens, the decoder only calculates attention between the current token and the rest of the sequence, which is repeated for each token. Before performing iterative attention, initialize FFA with K and V from the prompt


```
from ff_attention import FastForwardAttn as FFA
ffa = FFA(K, V, prompt_len+output_len, use_default_kernel_launch)
```

Where set `use_default_kernel_launch=True` to use FasterTransformer's default kernel launch or `use_default_kernel_launch=False` to use optimized kernel launch. Then, to calculate attention between a new token at timestep `t` and the rest of the sequence  (with pruning threshold `p`, set `p=0` for exact attention) call 

```
ffa_attn = ffa.attention(new_token_q, new_token_k, new_token_v, softmax_scale, t, p)
```

`ffa_attn` is the attention output. That is, we have 

```
def torch_attn_func(Q, K, V, softmax_temp):
    return torch.bmm(torch.nn.functional.softmax(torch.bmm(Q, K.transpose(1, 2))*softmax_temp, dim=-1), V)
    
K = torch.cat([K, new_token_k], axis=-2)
V = torch.cat([V, new_token_v], axis=-2)
torch_attn =torch_attn_func(new_token_q.view(head_num, 1, head_dim), K.view(head_num, t+1, head_dim), V.view(head_num, t+1, head_dim), softmax_temp)

assert torch.abs(torch_attn-ffa_attn).max() < 1e-7
```

A complete example is in example.py

## Results
We perform [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) summarization task with [opt-1.3b](https://huggingface.co/facebook/opt-1.3b) and compare FFA with default huggingface model. Results below are total model forward pass time for performing summarization for 100 test samples from cnn_dailymail dataset. Results are on RTX 3090. We use 1,000 tokens from the atricles (to avoid OOM across batch sizes) and generate a 256 and 128 token summary. 

The modified code for using FFA with opt can be found in modeling_opt.py. The code modifies [this file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) in the transformers library, so that it uses FFA for attention calculation. Run 

```
python summarize.py --max_ite 100  --hf_model_name facebook/opt-1.3b --data_type fp16 --output_len 256
```
to obtain the following results

|Batch size | 10 | 5 | 2 | 1 |
| -------- |--------| --------|  --------| --------| 
|FFA + HuggingFace runtime (s) | 34.88 | 48.99 |  88.65 | 160.64 |
|Default HuggingFace runtime (s) | 68.46 | 82.57 | 119.34 | 194.98 |

and 
```
python summarize.py --max_ite 100  --hf_model_name facebook/opt-1.3b --data_type fp16 --output_len 128
```
to obtain

|Batch size | 10 | 5 | 2 | 1 |
| -------- |--------| --------|  --------| --------| 
|FFA + HuggingFace runtime (s)| 23.36 | 30.44 | 50.35 | 90.95 |
|Default HuggingFace runtime (s) | 38.88 | 45.83 | 64.59 | 105.60 |

Next, we compare the attention inference time with the orignal FasterTransformer kernel. For a max token value, `x`, the models are given an input prompt with `x-128` tokens, and `128` tokens are generated iteratively, so that the final sequence length is `x`. The reported values below are the ratio between total time for ffa kernel compared with [FasterTransformer's kernel](https://github.com/NVIDIA/FasterTransformer/tree/main/src/fastertransformer/kernels). 

Speedup on A100:
![alt text](https://github.com/szeighami/FastForwardAttention/blob/main/results/A100.png)

Speedup on RTX 2080 Ti:
![alt text](https://github.com/szeighami/FastForwardAttention/blob/main/results/2080.png)

