import torch
from math import sqrt
from ff_attention import FastForwardAttn as FFA


def torch_attn_func(Q, K, V, softmax_temp):
    return torch.bmm(torch.nn.functional.softmax(torch.bmm(Q, K.transpose(1, 2))*softmax_temp, dim=-1), V)

batch_size = 10
head_num =32 
head_dim = 64
prompt_len = 1024
output_len = 128
softmax_temp = 1/(sqrt(head_dim))

dtype = torch.float16 #supports torch.float16 and torch.float32

K = torch.rand(batch_size, head_num, prompt_len, head_dim, dtype=dtype).cuda()
V = torch.rand(batch_size, head_num, prompt_len, head_dim, dtype=dtype).cuda()
Q = torch.rand(batch_size, head_num, prompt_len, head_dim, dtype=dtype).cuda()

#attention for the prompt is not calculated by ffa
attn = torch_attn_func(Q.view(batch_size*head_num, prompt_len, head_dim), K.view(batch_size*head_num, prompt_len, head_dim), V.view(batch_size*head_num, prompt_len, head_dim), softmax_temp)

#init ffa
total_sequence_length = prompt_len+output_len
ffa = FFA(K, V, total_sequence_length)

for i in range(output_len):
    new_token_k = torch.rand(batch_size, head_num, 1, head_dim, dtype=dtype).cuda()
    new_token_v = torch.rand(batch_size, head_num, 1, head_dim, dtype=dtype).cuda()
    new_token_q = torch.rand(batch_size, head_num, 1, head_dim, dtype=dtype).cuda()
    curr_timestep = prompt_len+i

    #using ffa
    ffa_attn = ffa.attention(new_token_q, new_token_k, new_token_v, softmax_temp, curr_timestep, 0)

    #using pytorch
    K = torch.cat([K, new_token_k], axis=-2)
    V = torch.cat([V, new_token_v], axis=-2)
    torch_attn =torch_attn_func(new_token_q.view(batch_size*head_num, 1, head_dim), K.view(batch_size*head_num, curr_timestep+1, head_dim), V.view(batch_size*head_num, curr_timestep+1, head_dim), softmax_temp)


    #Difference is at most 1 percent
    assert (torch.abs(torch_attn.view(batch_size, head_num, 1, head_dim)-ffa_attn.view(batch_size, head_num, 1, head_dim))/torch_attn.view(batch_size, head_num, 1, head_dim)).max() <  1e-2




    

