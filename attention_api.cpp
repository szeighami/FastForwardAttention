#include <torch/extension.h>

template<typename T>
void call_attention(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out);

template<>
void call_attention<float>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, float* q, float* k, float* v, float* k_cache, float* v_cache, float* out);


void attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out, float softmax_scale)
{
    int batch_size = k_cache.size(0);
    int head_num = k_cache.size(1);
    int seq_length = k_cache.size(3);
    int head_dim = k_cache.size(2)*k_cache.size(4);

    call_attention<float>(batch_size, head_num, head_dim, seq_length, softmax_scale, q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), k_cache.data_ptr<float>(), v_cache.data_ptr<float>(), out.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("attention", &attention, "Attention forward (CUDA)");
}
