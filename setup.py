from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ff_attention',
    ext_modules=[
        CUDAExtension('ff_attention', [
            'torch_ffa_api.cpp',
            'attention.cu',
        ],
        extra_compile_args={'nvcc':"-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -gencode=arch=compute_86,code=\"sm_86,compute_86\" -gencode=arch=compute_80,code=\"sm_80,compute_80\"".split(" ")}#--gencode=arch=compute_70,code=\"sm_70,compute_70\"  --gencode=arch=compute_75,code=\"sm_75,compute_75\" -gencode=arch=compute_80,code=\"sm_80,compute_80\" -gencode=arch=compute_86,code=\"sm_86,compute_86\"".split(" ") }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

