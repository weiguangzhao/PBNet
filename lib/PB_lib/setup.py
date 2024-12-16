from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='PB_lib',
    ext_modules=[
        CUDAExtension('PB_lib', [
            'src/PB_lib_api.cpp',
            'src/PB_lib.cpp',
            'src/cuda.cu'
        ],  extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)

