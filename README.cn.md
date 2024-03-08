# 项目描述

动态时间规整(DTW)是一种用于比较两个不完全对齐的时间序列的数学技术。该算法的详细介绍和解释可以在访问[算法说明](https://www.cnblogs.com/Li-JT/p/16966748.html)。DTW在计算长序列数据时可能非常耗时，通过利用GPU加速，DTW的计算速度可以提高数百倍。利用本软件，可以在几秒钟内计算出数百万个一维序列数据的DTW距离。

本软件是一个python库，可以利用gpu加速DTW计算，支持CUDA和OpenCL两种加速。这样既可以用常见的NVIDIA显卡，也可以使用AMD或英特尔显卡进行计算加速。

本软件对GPU显存进行了优化。如果数据量超出GPU显存容量，数据会被分割成小块执行计算。只要你的主机内存可以容纳数据，就无需考虑GPU的显存容量，只有512M显存的GPU也能运行本软件。

本软件已在各种显卡上进行了测试，包括古老的GeForce 9800 GT和现代的RTX 4090。本软件的OpenCL功能可以运行在AMD的高、低端显卡以及英特尔的IGP核显上。

# 硬件规格

本软件与各种现代PC硬件设备兼容，包括便携式笔记本电脑，并且可以利用CPU集成图形提供的加速。

# 软件需求

由于Windows自带的部分显卡驱动程序不支持OpenCL，因此需要安装正确的显卡驱动程序才能使用本软件。可以使用GPU-z软件检查显卡是否支持OpenCL。建议在AMD显卡上安装Adrenalin驱动，不要安装PRO驱动。

本软件需要Python中的pycuda或pyopencl模块的支持。您可以选择安装适合您的模块，并且没有必要同时安装两个模块。下面是安装命令:
~~~bash
pip install pycuda
pip install pyopencl
~~~
注意:在安装pycuda之前，您可能需要安装cuda工具包。虽然pycuda和pyopencl是本软件包的基本依赖项，但是为了降低安装难度，为用户提供更多的选择，在本软件包的安装过程中没有强制配置这些依赖项。

然后，您可以选择以下两个选项：

## CUDA
使用本软件的CUDA加速特性时，需要安装CUDA Toolkit。应下载并使用与您的显卡兼容的任何版本的CUDA工具包，甚至是旧的6.5版本。

## OpenCL
使用OpenCL简单一些，只需要安装正确的显卡驱动程序。

Windows 10捆绑的OpenCL.dll版本是2.0.4.0，与最新的pyopencl不兼容。要解决此问题，您需要在本软件的安装目录中找到另一个OpenCL.dll文件(版本3.0.3.0)，该文件通常位于“lib/site-packages/GPUDTW”。然后，手动将该文件复制到Windows目录下的system32子目录，覆盖现有的2.0.4.0版本。

在有多个显卡或集成图形的CPU的情况下，软件运行时可能会提示您选择使用哪个硬件来加速计算。此时，您应该选择性能更强的显卡。当使用带有第11代或更新的英特尔处理器笔记本电脑时，选择英特尔IGP可能会产生意想不到结果。

# 使用说明

输入数据集是二维的numpy数组。第一个维度表示数据集的大小，第二个维度表示要计算的向量长度。输入数据集由两个数组组成，一个作为源，另一个作为目标。源数据的向量长度必须等于目标数据的向量长度。如果它们不相等，则需要对齐它们，比如上采样和下采样的对齐方法。

然后，只需调用**cuda_dtw**或**opencl_dtw**函数来计算两个数据集之间的DTW欧氏距离。

返回的数据是一个2D数组，其中第一维表示源数据向量的位置，第二维表示目标数据向量的位置。例如，位置(2,3)处的数据对应于第2个源数据向量与第3个目标数据向量之间的DTW欧氏距离。

此外，本软件还提供了一个基于cpu的计算函数来验证从GPU获得的结果。这个CPU函数使用了numba模块进行并行加速，因此您需要通过运行“pip install numba”来安装numba模块。

在本软件包的安装目录中可以找到**test.py**，也就是下面的示例代码
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

# 问题处理

1. 遇到程序不能正常工作或崩溃的问题时，应该检查您的显卡驱动程序版本是否正确，或者是否与CUDA工具包的版本匹配。有关NVIDIA的相关知识，可以参考此[链接](https://developer.nvidia.com/cuda-gpus)。

2. 由于GPU的高速local memory(也就是显卡的片上内存)一般都比较小，典型的是32k~64k，所以计算的向量长度不能过大，通常不超过2500左右。如果向量长度太大，GPU的local memory无法容纳计算缓冲区，程序将直接报告错误并退出。当然也可以考虑将计算缓冲区放在显存(global memory)中，但是显存的访问速度明显慢于片上内存，导致并行计算的优势将不那么明显。此外，DTW算法经常用于时间序列数据的比较，在实际工作中很少遇到长度超过1000的时间序列。所以，向量不能过长的问题被我暂时搁置。如果您有特殊需求，请在GitHub上留下详细的信息，详细解释您的具体工作背景，也许能说服我，有时间的话，我可以更新一个在显存上执行计算的新版本。（稍微多解释一下：在GPU加速程序中已经优化了计算缓冲区，只需要三倍的一个向量数组容量，而不是CPU版本中的向量长度的平方倍。一个2048长度的float32向量，就要占掉8k内存，翻三倍，也就是24k内存，这样基本上就把local memory塞满了。）

# 版权说明

版权所有：武汉理工大学(2024)

作者: 王子浩 <qianlkzf@outlook.com> 
 
这个程序是自由软件：您可以按照自由软件基金会发布的GNU通用公共许可证条款重新分发它或对其进行修改，可以是许可证的第三版，或者（根据您的选择）任何后续版本。

分发此程序是希望它有用，但不做任何担保；甚至没有暗示对适销性或特定用途的适合性的担保。有关更多详细信息，请参阅GNU通用公共许可证。

您应该已经收到了随附此程序的GNU通用公共许可证副本。如果没有，请访问https://www.gnu.org/licenses/。

## 许可证

本项目采用《GNU通用公共许可证v3.0》授权。

## GitHub仓库
您可以在GitHub上找到此项目的源代码和更多信息，地址如下：

[<img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github">](https://github.com/qianlikzf/GPUDTW)  

或者直接访问：

[https://github.com/qianlikzf/GPUDTW](https://github.com/qianlikzf/GPUDTW)
