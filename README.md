# FastForwardAttention
FastForwardAttention (FFA) is a pytorch extention for efficient exact iterative attention implementation for natural language generation. It can easily replace iterative attention in huggingface models and improve throughput by 2x (see results below). It's built by performing gpu utilization optimization on top of [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)'s implementation for faster inference (see results below)

## FFA with pytorch
### Install
Run 

```
python setup.py install
```

### Use
FFA handels iterative attention for transformer decoder. When generating tokens, the decoder only calculates attention between the current token and the rest of the sequence, which is repeated for each token. Before performing iterative attention, initialize FFA with K and V from the prompt


```
from ff_attention import FastForwardAttn as FFA
ffa = FFA(K, V, prompt_len+output_len)
```

Then, for to calculate attention between a new token at timestep t and the rest of the sequence call

```
ffa_attn = ffa.attention(new_token_q, new_token_k, new_token_v, softmax_scale, t, 0)
```

A complete example is in examply.py

### Results
We perform [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) summarization task with [opt-1.3b](https://huggingface.co/facebook/opt-1.3b) and compare FFA with default huggingface model. Results below are total model forward pass time for performing summarization for 100 test samples from cnn_dailymail dataset. 

|Batch size | 10 | 5 | 2 | 1 |
| -------- |--------| --------|  --------| --------| 
|FFA runtime (s) | 23.36 | 30.44 | 50.35 | 90.95 |
|HuggingFace runtime (s) | 38.88 | 45.83 | 64.59 | 105.60 |

Next, we compare the attention inference time with the orignal FasterTransformer kernel. 

