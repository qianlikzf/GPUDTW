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
