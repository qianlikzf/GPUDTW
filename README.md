# Project description

Dynamic Time Warping (DTW) is a mathematical technique used to compare two temporal sequences that do not align perfectly. Detailed introductions and explanations of the algorithm can be found [here](https://builtin.com/data-science/dynamic-time-warping). For long sequences, DTW computations can be computationally intensive, thus requiring significant processing time. However, by utilizing GPU acceleration, the computation speed of DTW can be increased by hundreds of times. Using this software package, the DTW distances of millions of one-dimensional sequence data can be calculated in just a few seconds.

The software in this package is a library enables the utilization of GPUs for accelerated dynamic time warping (DTW), supporting both CUDA and OpenCL. This means you can reap the benefits of GPU acceleration from common NVIDIA graphics cards, as well as utilize AMD or Intel hardware for compute acceleration.

This software has been optimized for GPU memory. If the capacity of your array exceeds the GPU memory, the software will automatically split the data into smaller chunks and perform the calculations separately. This means that as long as your host machine's memory can accommodate your data, you don't need to worry about the GPU's memory capacity, even if your GPU only has 512M of memory.

The software has been tested on various GPUs, including the older GeForce 9800 GT and the modern RTX 4090, and both can utilize this software effectively. AMD's range of high-end and low-end graphics cards, as well as Intel's IGP, can also utilize the OpenCL functionality of this software. 

# Hardware specifications

This software is compatible with various modern PC hardware devices, including portable laptops, and may take advantage of acceleration provided by the CPU's integrated graphics.

# Software requirements

To use this software package, you need to install the CORRECT graphics card drivers first, because some drivers shipped by Windows do not support OpenCL. You can use the software GPU-z to check whether your graphics card supports OpenCL. It is recommended to install the Adrenalin driver on AMD graphics cards, rather than the PRO driver.

This software requires the support of the pycuda or pyopencl modules in Python. You can choose to install the one that suits your acceleration needs, and it's not necessary to install both modules. Here are the installation commands:
~~~bash
pip install pycuda
pip install pyopencl
~~~
Note: You may need to install the cuda tool kit before installing pycuda. Although pycuda and pyopencl are the basic dependencies for this software package, in order to reduce the installation difficulties and provide users with more options, these dependencies are not forcibly configured in the setup of this software package.

Then, you can choose from the following two options:

## CUDA
When utilizing the CUDA acceleration feature of this software, you need to install the CUDA Toolkit. You can download and use any version of the CUDA Toolkit that is compatible with your graphics card, even the older 6.5 version.

## OpenCL
Using OpenCL is simpler and only necessitates the installation of correct graphics card drivers. 

The OpenCL.dll version 2.0.4.0 bundled with Windows 10 is not compatible with the latest pyopencl module. To resolve this issue, you need to locate another OpenCL.dll file (version 3.0.3.0) in the installation directory of the module, which is typically located in "lib/site-packages/GPUDTW". Then, manually copy this file to the system32 subdirectory within your Windows directory, overwriting the existing 2.0.4.0 version.

In cases where there are multiple graphics cards or your CPU with integrated graphics, you may be prompted to select which hardware to use for accelerated computing. At this point, you should choose the stronger graphics card based on your knowledge of your hardware's performance. When using a laptop with an 11th generation Intel processor or newer processors, selecting the Intel IGP may yield unexpected benefits.

# How to use

Prepare your data, which should be a 2D numpy array. The first dimension represents the size of the data, and the second dimension represents the length of the vectors to be calculated. Your data consists of two arrays, one as the source and the other as the target. The vector length of the source data must be equal to that of the target data. If they are not equal, you need to align them, and there are many methods for alignment, such as upsampling and downsampling. 

Then, simply call the **cuda_dtw** or **opencl_dtw** function to compute the DTW Euclidean distance between the two datasets. 

The returned data is a 2D array, where the first dimension represents the number of source data vectors, and the second dimension represents the number of target data vectors. For instance, the data at position (2,3) corresponds to the DTW Euclidean distance between the second source data vector and the third target data vector.

Additionally, a CPU-based computation function is provided to validate the results obtained from the GPU. This CPU function leverages the numba module for parallel acceleration, so you'll need to install the numba module by running "pip install numba".

Examples are available in the unit test script **test.py**

~~~python
from __future__ import absolute_import
from __future__ import print_function

import numpy
import time

try:
    from GPUDTW import cuda_dtw
except:
    pass

try:
    from GPUDTW import opencl_dtw
except:
    pass

try:
    from GPUDTW import cpu_dtw, dtw_1D_jit2
except:
    pass

if __name__ == '__main__':
    S = numpy.random.random ((3,1212))
    S = S.astype(numpy.float32)
    T = numpy.random.random ((1312,1212))
    T = T.astype(numpy.float32)

    t0 = time.time()
    ret_cpu =cpu_dtw (S, T, dtw_1D_jit2)
    print ("cpu time",time.time()-t0)

    if 'cuda_dtw' in locals():
        t0 = time.time()
        ret_cuda = cuda_dtw (S, T)
        print ("cuda time:",time.time()-t0)
        cuda_verify = numpy.sqrt((ret_cuda - ret_cpu)**2)
        print ("Maximum Deviation in cuda with CPU ", cuda_verify.max())

    #os.environ['PYOPENCL_CTX'] = '0'
    if 'opencl_dtw' in locals():
        t0 = time.time()
        ret_opencl = opencl_dtw (S, T)
        print ("OpenCL time:",time.time()-t0)
        opencl_verify = numpy.sqrt((ret_opencl - ret_cpu)**2)
        print ("Maximum Deviation in OpenCL with CPU", opencl_verify.max())
~~~

# Trouble tip

1. When you encounter issues with the program not working properly or crashing, you should check if your graphics card driver version is correct, or if it matches the version of the CUDA Toolkit. For related knowledge on NVIDIA, you can refer to this [link](https://developer.nvidia.com/cuda-gpus).

2. The vector length cannot be excessively large, typically not exceeding around 2500, due to the limitations of the GPU's high-speed local memory (on-chip memory). If the vector length is too large, the computation buffer cannot be accommodated within the GPU's local memory, and the program will directly report an error and exit. Alternatively, one can consider placing the computation buffer in the graphics memory (global memory), but the access speed of graphics memory is significantly slower than local memory, and the advantage of parallel computing will not be as apparent. Additionally, the DTW algorithm is often used for comparing time-series data, and in practical work, it is rare to encounter time-series with lengths exceeding 1000. Therefore, this issue can be temporarily shelved. If you do encounter such a special case, please leave a detailed message on GitHub, explaining your specific work background. If you can convince me, perhaps I will have time to update a new version that can perform calculations on graphics memory.

# Copyright

 Copyright (C) 2024 Wuhan University of Technology

 Authors: Wang Zihao <qianlkzf@outlook.com> 
  
 This program is free software: you can redistribute it and/or modify  
 it under the terms of the GNU General Public License as published by  
 the Free Software Foundation, either version 3 of the License, or  
 (at your option) any later version.  
  
 This program is distributed in the hope that it will be useful,  
 but WITHOUT ANY WARRANTY; without even the implied warranty of  
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the  
 GNU General Public License for more details.  
  
 You should have received a copy of the GNU General Public License  
 along with this program.  If not, see <https://www.gnu.org/licenses/>.  
 
## License  
  
This project is licensed under the [GNU General Public License v3.0](LICENSE).  
  
## GitHub Repository  
  
You can find the source code and more information about this project on GitHub at:  
  
[<img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github">](https://github.com/qianlikzf/GPUDTW)  
  
Or visit directly:  
  
[https://github.com/qianlikzf/GPUDTW](https://github.com/qianlikzf/GPUDTW)

