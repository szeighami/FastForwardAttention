#c++ -MMD -MF /workspace/FastForwardAttention/c_build/c_ffa_api.o.d -pthread  -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC  -I/usr/local/cuda/include -c -c /workspace/FastForwardAttention/c_ffa_api.cpp -o /workspace/FastForwardAttention/c_build/c_ffa_api.o  -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14

/usr/local/cuda/bin/nvcc -c -c /workspace/FastForwardAttention/attention.cu -o /workspace/FastForwardAttention/c_build/attention.o  --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ '-gencode=arch=compute_86,code="sm_86,compute_86"'  -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -lcudart -lcurand

c++ -shared  -Wl,-soname,libffa.so -o libffa.so attention.o -L /usr/local/cuda/lib64/ -lcudart -lcurand

c++ -o run_attention main.cpp -L/workspace/FastForwardAttention/c_build -lffa

