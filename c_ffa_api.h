#include"kernel_launch_utils.h"



template <typename T>
void call_attention(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb, bool right_or_no_padding){
    printf("unsupported attention type");
}

template <>
void call_attention<float>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, float* q, float* k, float* v, float* k_cache, float* v_cache, float* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb, bool right_or_no_padding);

template <>
void call_attention<uint16_t>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, uint16_t* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb, bool right_or_no_padding);


template <typename T>
void do_attention(int batch_size,int head_num, int head_dim, int max_seq_length, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out, float softmax_scale, int curr_timestep, int* seq_lengths, float prob_thresh, int tpb, int tpv, int tpk){
    call_attention<T>(batch_size, head_num, head_dim, max_seq_length, softmax_scale, curr_timestep, q, k, v, k_cache, v_cache, out, seq_lengths, prob_thresh, tpv, tpk, tpb, true);
}
