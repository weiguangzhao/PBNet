from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
include_dirs = ['/usr/local/cuda/include/']
setup(
    name='PB_lib',
    ext_modules=[
        CUDAExtension('PB_lib', [
            'src/PB_lib_api.cpp',
            'src/PB_lib.cpp',
            'src/cuda.cu'
        ],  include_dirs=include_dirs, extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)

