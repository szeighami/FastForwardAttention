#include <torch/extension.h>

template<typename T>
void call_attention(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out);

template<>
void call_attention<float>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, float* q, float* k, float* v, float* k_cache, float* v_cache, float* out);

template<>
void call_attention<uint16_t>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, uint16_t* out);



torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_cache, torch::Tensor v_cache, float softmax_scale, int curr_timestep)
{
    int batch_size = k_cache.size(0);
    int head_num = k_cache.size(1);
    int seq_length = k_cache.size(3);
    int head_dim = k_cache.size(2)*k_cache.size(4);
    //printf("batch_size %d\n", batch_size);
    //printf("head_num %d\n", head_num );
    //printf("seq_length %d\n", seq_length);
    //printf("head_dim  %d\n", head_dim );
    auto options = q.options();
    torch::Tensor out = torch::empty({batch_size, head_num, head_dim}, options);


    if (q.scalar_type() == torch::ScalarType::Float){
        using T = float;
        call_attention<T>(batch_size, head_num, head_dim, seq_length, softmax_scale, curr_timestep, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), k_cache.data_ptr<T>(), v_cache.data_ptr<T>(), out.data_ptr<T>());
    }
    else if (q.scalar_type() == torch::ScalarType::Half){
        using T = uint16_t;
        //T* a = static_cast<T*>(q.data_ptr());
        //auto a = q.data_ptr();
        call_attention<T>(batch_size, head_num, head_dim, seq_length, softmax_scale, curr_timestep, reinterpret_cast<T*>(q.data_ptr<at::Half>()), reinterpret_cast<T*>(k.data_ptr<at::Half>()), reinterpret_cast<T*>(v.data_ptr<at::Half>()), reinterpret_cast<T*>(k_cache.data_ptr<at::Half>()), reinterpret_cast<T*>(v_cache.data_ptr<at::Half>()), reinterpret_cast<T*>(out.data_ptr<at::Half>()));
    }
    else
        printf("unsupported tensort type\n");





    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("attention", &attention, "Attention forward (CUDA)");
}
