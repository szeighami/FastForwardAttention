#include <torch/extension.h>
#include "kernel_launch_utils.h"

template<typename T>
void call_attention(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb);

template<>
void call_attention<float>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, float* q, float* k, float* v, float* k_cache, float* v_cache, float* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb);

template<>
void call_attention<uint16_t>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, uint16_t* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb);


template<typename T>
void invokeTranspose4dBatchMajor(T*           k_dst,
                                 T*           v_dst,
                                 const T*     k_src,
                                 const T*     v_src,
                                 const int    local_batch_size,
                                 const int    seq_len,
                                 const int    max_seq_len,
                                 const int    size_per_head,
                                 const int    local_head_num
                                 );
template<>
void invokeTranspose4dBatchMajor<float>(float*           k_dst,
                                         float*           v_dst,
                                         const float*     k_src,
                                         const float*     v_src,
                                         const int    local_batch_size,
                                         const int    seq_len,
                                         const int    max_seq_len,
                                         const int    size_per_head,
                                         const int    local_head_num
                                         );

template<>
void invokeTranspose4dBatchMajor<uint16_t>(uint16_t*           k_dst,
                                         uint16_t*           v_dst,
                                         const uint16_t*     k_src,
                                         const uint16_t*     v_src,
                                         const int    local_batch_size,
                                         const int    seq_len,
                                         const int    max_seq_len,
                                         const int    size_per_head,
                                         const int    local_head_num
                                         );

std::pair<torch::Tensor, torch::Tensor> init_attention_cache(torch::Tensor K, torch::Tensor V, int max_timesteps)
{

    int batch_size = K.size(0);
    int head_num = K.size(1);
    int seq_length = K.size(2);
    int head_dim = K.size(3);

    torch::Tensor k_cache;
    torch::Tensor v_cache;

    if (K.scalar_type() == torch::ScalarType::Float){
        using T = float;
        int elem_per_16_b = 4;

        auto options = K.options();
        k_cache = torch::empty({batch_size, head_num, head_dim/elem_per_16_b, max_timesteps, elem_per_16_b}, options);
       v_cache = torch::empty({batch_size, head_num, max_timesteps, head_dim}, options);

        invokeTranspose4dBatchMajor<T>( k_cache.data_ptr<float>(),
                                     v_cache.data_ptr<float>(),
                                     reinterpret_cast<const T*>(K.data_ptr<float>()),
                                     reinterpret_cast<const T*>(V.data_ptr<float>()),
                                     batch_size,
                                     seq_length,
                                     max_timesteps,
                                     head_dim,
                                     head_num
                                     );

    }
    else if (K.scalar_type() == torch::ScalarType::Half){
        using T = uint16_t;
        int elem_per_16_b = 8;

        auto options = K.options();
        k_cache = torch::empty({batch_size, head_num, head_dim/elem_per_16_b, max_timesteps, elem_per_16_b}, options);
        v_cache = torch::empty({batch_size, head_num, max_timesteps, head_dim}, options);

        invokeTranspose4dBatchMajor<T>( reinterpret_cast<T*>(k_cache.data_ptr<at::Half>()),
                                     reinterpret_cast<T*>(v_cache.data_ptr<at::Half>()),
                                     reinterpret_cast<const T*>(K.data_ptr<at::Half>()),
                                     reinterpret_cast<const T*>(V.data_ptr<at::Half>()),
                                     batch_size,
                                     seq_length,
                                     max_timesteps,
                                     head_dim,
                                     head_num
                                     );

    }
    else
        printf("unsupported tensort type\n");



    return std::make_pair(k_cache, v_cache);


}

torch::Tensor do_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_cache, torch::Tensor v_cache, float softmax_scale, int curr_timestep, torch::Tensor seq_lengths, float prob_thresh, int tpv, int tpk, int tpb)
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
        call_attention<T>(batch_size, head_num, head_dim, seq_length, softmax_scale, curr_timestep, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), k_cache.data_ptr<T>(), v_cache.data_ptr<T>(), out.data_ptr<T>(), seq_lengths.data_ptr<int>(), prob_thresh, tpv, tpk, tpb);
    }
    else if (q.scalar_type() == torch::ScalarType::Half){
        using T = uint16_t;
        //T* a = static_cast<T*>(q.data_ptr());
        //auto a = q.data_ptr();
        call_attention<T>(batch_size, head_num, head_dim, seq_length, softmax_scale, curr_timestep, reinterpret_cast<T*>(q.data_ptr<at::Half>()), reinterpret_cast<T*>(k.data_ptr<at::Half>()), reinterpret_cast<T*>(v.data_ptr<at::Half>()), reinterpret_cast<T*>(k_cache.data_ptr<at::Half>()), reinterpret_cast<T*>(v_cache.data_ptr<at::Half>()), reinterpret_cast<T*>(out.data_ptr<at::Half>()), seq_lengths.data_ptr<int>(), prob_thresh, tpv, tpk, tpb);
    }
    else
        printf("unsupported tensort type\n");


    return out;
}

torch::Tensor do_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_cache, torch::Tensor v_cache, float softmax_scale, int curr_timestep, float prob_thresh, int tpv, int tpk, int tpb)
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
        call_attention<T>(batch_size, head_num, head_dim, seq_length, softmax_scale, curr_timestep, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), k_cache.data_ptr<T>(), v_cache.data_ptr<T>(), out.data_ptr<T>(), nullptr, prob_thresh, tpv, tpk, tpb);
    }
    else if (q.scalar_type() == torch::ScalarType::Half){
        using T = uint16_t;
        //T* a = static_cast<T*>(q.data_ptr());
        //auto a = q.data_ptr();
        call_attention<T>(batch_size, head_num, head_dim, seq_length, softmax_scale, curr_timestep, reinterpret_cast<T*>(q.data_ptr<at::Half>()), reinterpret_cast<T*>(k.data_ptr<at::Half>()), reinterpret_cast<T*>(v.data_ptr<at::Half>()), reinterpret_cast<T*>(k_cache.data_ptr<at::Half>()), reinterpret_cast<T*>(v_cache.data_ptr<at::Half>()), reinterpret_cast<T*>(out.data_ptr<at::Half>()), nullptr, prob_thresh, tpv, tpk, tpb);
    }
    else
        printf("unsupported tensort type\n");


    return out;
}


struct FastForwardAttn {
    FastForwardAttn(torch::Tensor K, torch::Tensor V, int max_timesteps) 
    {  
        auto kv_cache = init_attention_cache(K, V, max_timesteps);
        k_cache = kv_cache.first;
        v_cache = kv_cache.second;

        int seq_length = K.size(2);
        int batch_size = K.size(0);
        int num_heads = K.size(1);
        int head_dim = K.size(3);
        int precision = K.scalar_type() == torch::ScalarType::Float ? 32 : 16;

        int no_sms = 82;  float C = 3;   float gpu_clock_rate = 1395e6; int mySharedMemAllocationSize=128; int max_sharedmemory_per_block=102400; int run_time_smem=1024;
        float gpu_transfer_rate =936*std::pow(2,30) ;

        get_launch_params(precision,batch_size, num_heads, head_dim, seq_length, no_sms, C, gpu_transfer_rate, gpu_clock_rate, mySharedMemAllocationSize, max_sharedmemory_per_block, run_time_smem, tpb, tpv, tpk);
    }

    torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_cache, torch::Tensor v_cache, float softmax_scale, int curr_timestep, float prob_thresh=0)
    {
        return do_attention(q, k, v, k_cache, v_cache, softmax_scale, curr_timestep, prob_thresh, tpv, tpk, tpb);
    }

    torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_cache, torch::Tensor v_cache, float softmax_scale, int curr_timestep, torch::Tensor seq_lengths, float prob_thresh=0)
    {
        return do_attention(q, k, v, k_cache, v_cache, softmax_scale, curr_timestep, seq_lengths, prob_thresh, tpv, tpk, tpb);
    }

    std::pair<torch::Tensor, torch::Tensor> get_cache() { return std::make_pair(k_cache, v_cache); }



    //void setName(const std::string &name_) { name = name_; }
    //const std::string &getName() const { return name; }

	torch::Tensor k_cache;
	torch::Tensor v_cache;
    int tpv;
    int tpk;
    int tpb;
	
};



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<FastForwardAttn>(m, "FastForwardAttn")
      .def(py::init<torch::Tensor, torch::Tensor, int>())
      .def("get_cache", &FastForwardAttn::get_cache, "Get KV_cache")
      .def("attention", py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, int, float>(&FastForwardAttn::attention), "Attention forward (CUDA) fixed timesteps")
      .def("attention", py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, int, torch::Tensor, float>(&FastForwardAttn::attention), "Attention forward (CUDA) varied timesteps");
}
