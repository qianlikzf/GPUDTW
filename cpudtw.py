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

import numpy
import numba

"""
Calculate the DTW Euclidean distance between two 1D vectors, 
using numba for parallel acceleration
"""
@numba.njit(fastmath=True,parallel=True,nogil=True)
def dtw_1D_jit2(s1,s2):
    """Caculate DTW Euclidean distance between two vectors by CPU"""
    l1 = len(s1)
    l2 = len(s2)
    cum_sum = numpy.empty((l1 + 1, l2 + 1))
    cum_sum[0,  0] = 0.0
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = (s1[i]-s2[j])*(s1[i]-s2[j])

    for i in range(l1):
        for j in range(l2):
            #cum_sum[i + 1, j + 1] = (s1[i]-s2[j])*(s1[i]-s2[j])
            if numpy.isfinite(cum_sum[i + 1, j + 1]):
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j])
    ret = numpy.sqrt(cum_sum[l1, l2])
    return (ret)

def cpu_dtw (SrcS, TrgS, funC):
    """Caculate DTW Euclidean distance between two dataset by CPU"""
    ret = numpy.empty ((SrcS.shape[0],TrgS.shape[0]))
    for i in range(SrcS.shape[0]):
        for j in range(TrgS.shape[0]):
            a = SrcS[i]
            b = TrgS[j]
            ret[i,j] = funC(a,b)            
    return ret