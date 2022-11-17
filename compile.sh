export ffa_path="${ffa_path:-$PWD}"

echo "Compiling from and to $ffa_path"

echo "Compiling attention kernel"
/usr/local/cuda/bin/nvcc -c -c $ffa_path/attention.cu -o $ffa_path/attention.o  --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ '-gencode=arch=compute_75,code="sm_75,compute_75"' '-gencode=arch=compute_80,code="sm_80,compute_80"' '-gencode=arch=compute_86,code="sm_86,compute_86"'  -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -lcudart -lcurand

echo "Building the library"
c++ -shared  -Wl,-soname,libffa.so -o $ffa_path/libffa.so $ffa_path/attention.o -L /usr/local/cuda/lib64/ -lcudart -lcurand

echo "adding it to the path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ffa_path
