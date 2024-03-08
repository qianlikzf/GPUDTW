"""
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
"""

from __future__ import absolute_import
from __future__ import print_function

import pyopencl as cl
import pyopencl.array as cl_array

import os
import codecs
import numpy,time

def opencl_dtw (SrcS, TrgS):
    """Caculate DTW Euclidean distance between two dataset by OpenCL acceleration"""
    ctx, queue, prg, dev_Param = OpenCL_Init ()
    ret = opencl_dtw_run (SrcS, TrgS, ctx, queue, prg, dev_Param)
    return (ret)

def OpenCL_Init():
    fp = codecs.open(os.path.dirname(os.path.abspath(__file__))
                     +"./opencldtw.cl","r","utf-8")
    opencl_Source_Str = fp.read()
    fp.close()

    ctx = cl.create_some_context()
    prg = cl.Program(ctx, opencl_Source_Str).build()

    dev_Param = {}
    dev_Param["MAX_MEM_ALLOC_SIZE"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "MAX_MEM_ALLOC_SIZE"))
    dev_Param["LOCAL_MEM_SIZE"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "LOCAL_MEM_SIZE"))
    dev_Param["MAX_WORK_ITEM_SIZES"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "MAX_WORK_ITEM_SIZES"))
    dev_Param["MAX_WORK_GROUP_SIZE"] = \
        ctx.devices[0].get_info(getattr(cl.device_info, "MAX_WORK_GROUP_SIZE"))
    #print (dev_Param)

    queue = cl.CommandQueue(ctx)
    return ctx, queue, prg, dev_Param

# 获取本地内存类型对象
float_type = cl.tools.get_or_register_dtype("local_mem", "float")

def opencl_dtw_run (SrcS, TrgS, ctx, queue, prg, dev_Param):
    # 返回值
    allret = numpy.empty ((SrcS.shape[0],TrgS.shape[0]), dtype=numpy.float32)
    # cot1 根据 LOCAL_MEM_SIZE 和 kernel 需要分配的临时内存决定
    # kernel 计算 需要 3 倍的Trg缓冲区
    cot1 = int(dev_Param["LOCAL_MEM_SIZE"] 
               / (TrgS.shape[1] *3 *float_type.itemsize))
    cot2 = int(dev_Param["MAX_WORK_ITEM_SIZES"][0] / TrgS.shape[1])
    cot3 = int(dev_Param["MAX_WORK_GROUP_SIZE"] / TrgS.shape[1])
    """
    if cot3 < 1:
        raise ("Your OpenCL device's paralle MAX_WORK_GROUP_SIZE is lower than your data sequence length")
    if cot2 < 1:
        raise ("Your OpenCL device's paralle MAX_WORK_ITEM_SIZES is lower than your data sequence length")
    """
    if cot1 < 1:
        # Memory Problem can Not be optimized
        raise ("The basic memory requirements of the DTW algorithm exceed the capacity of your OpenCL device's LOCAL_MEM_SIZE, rendering it inadequate for parallel computing.")
    # one_Unit_calcCount 一个kernel函数内可以处理的 Trg 序列数量
    if cot2 < 1 or cot3 < 1:
        # MAX_WORK_GROUP_SIZE or MAX_WORK_ITEM_SIZES lower than data sequence length
        # so it only one work in a work group
        # we should maximize the utilization of local memory.
        one_Unit_calcCount = 1
        WORK_GROUP_SIZE = min (dev_Param["MAX_WORK_ITEM_SIZES"][0], dev_Param["MAX_WORK_GROUP_SIZE"])
        #allret = opencl_dtw_run_low_resource(SrcS, TrgS, queue, prg, dev_Param)
        #return (allret)
    else:
        one_Unit_calcCount = min(cot1, cot2, cot3)

    one_Unit_Mem = (TrgS.shape[1] +1)*one_Unit_calcCount *float_type.itemsize
    src_Mem = (SrcS.size + TrgS.shape[1])*float_type.itemsize
    Grp_Cot = int((dev_Param["MAX_MEM_ALLOC_SIZE"] -src_Mem)/one_Unit_Mem)
    # Grp_Cot 作为总的计算数，必须是一个单元内计算数的整数倍
    Grp_Cot = Grp_Cot - Grp_Cot %one_Unit_calcCount
    #print ("Grp_Cot ",Grp_Cot)

    if (Grp_Cot /one_Unit_calcCount) > dev_Param["MAX_WORK_GROUP_SIZE"]:
        Grp_Cot = dev_Param["MAX_WORK_GROUP_SIZE"] *one_Unit_calcCount

    # 对齐 TrgS 到 one_Unit_calcCount
    TrgS_Alignment =  one_Unit_calcCount -TrgS.shape[0] % one_Unit_calcCount
    #print ("TrgS.shape[0]", TrgS.shape[0], TrgS_Alignment)
    if TrgS_Alignment != one_Unit_calcCount:
        tmP = numpy.empty((TrgS.shape[0]+TrgS_Alignment, TrgS.shape[1]),dtype=numpy.float32)
        tmP[:TrgS.shape[0],:] = TrgS
        TrgS = tmP
        #Add_tmp = numpy.ones((TrgS_Alignment,TrgS.shape[1]),dtype=numpy.float32)
        #TrgS = numpy.concatenate ((TrgS, Add_tmp))
    T0 = TrgS.shape[0]
    #print ("TrgS_Alignment,TRG_COT",TrgS_Alignment,TRG_COT)

    #print ("Grp_Cot, one_Unit_calcCount", Grp_Cot, one_Unit_calcCount, T0)
    Splits = list(range(0, T0, Grp_Cot *one_Unit_calcCount))
    Splits.append (T0)

    local_size = one_Unit_calcCount
    #print ("local_size ",local_size)

    for j in range(len(Splits)-1):
        TrgS_sub = TrgS[Splits[j]:Splits[j+1],:]
        Ts0 = TrgS_sub.shape[0]
        Ts1 = TrgS_sub.shape[1]
        #global_size = Ts0 *Ts1
        #local_size  = TRG_COT *Ts1
        global_size = Ts0
        #print ("global_size ",global_size)

        t = numpy.reshape(TrgS_sub,(Ts0 *Ts1))
        t_dev = cl_array.to_device(queue, t)
        #print ("local_size, global_size, t_dev nbytes(MB) \n",local_size,global_size,t.nbytes/1024/1024)
 
        SRC_LEN = SrcS.shape[1]
        TRG_LEN = TrgS.shape[1]

        for i in range(SrcS.shape[0]):
            s = SrcS[i,:]
            s_dev = cl_array.to_device(queue, s)
            r_dev = cl_array.empty (queue, (Ts0,), dtype=numpy.float32)


            shared_mem_size = TRG_LEN *one_Unit_calcCount *float_type.itemsize            
            path_h1 = cl.LocalMemory (shared_mem_size)
            path_h2 = cl.LocalMemory (shared_mem_size)
            dist    = cl.LocalMemory (shared_mem_size)
            
            print ("global_size, local_size", global_size, local_size)
            if "WORK_GROUP_SIZE" not in locals():
                prg.opencl_dtw(queue, 
                    (global_size ,TRG_LEN), 
                    (local_size  ,TRG_LEN),
                    numpy.uint32(SRC_LEN),
                    #numpy.uint32(TRG_LEN),
                    #numpy.uint32(one_Unit_calcCount),
                    s_dev.data, t_dev.data, r_dev.data,\
                    path_h1,path_h2,dist)
            else:
                #print ("opencl_dtw_low_resource WORK_GROUP_SIZE",WORK_GROUP_SIZE)
                prg.opencl_dtw_low_resource(queue,\
                    (global_size ,WORK_GROUP_SIZE), 
                    (1           ,WORK_GROUP_SIZE),
                    numpy.uint32(SRC_LEN),
                    numpy.uint32(TRG_LEN),
                    s_dev.data, t_dev.data, r_dev.data,
                    #localMem)
                    path_h1,path_h2,dist)
            r = r_dev.get()
            allret[i,Splits[j]:Splits[j+1]] = r
            #print(la.norm((dest_dev - (a_dev+b_dev)).get()))

    if TrgS_Alignment != one_Unit_calcCount:
        allret = allret[:, 0:-TrgS_Alignment]
    return (allret)

