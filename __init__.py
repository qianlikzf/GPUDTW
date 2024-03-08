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
try:
    import pyopencl
    from .opencldtw import opencl_dtw
except:
    print ("Warning: pyopencl is not installed.")
    pass

try:
    import pycuda
    from .cudadtw import cuda_dtw
except:
    print ("Warning: pycuda is not installed.")
    pass

try:
    import numba
    from .cpudtw import cpu_dtw, dtw_1D_jit2    
except:
    print ("Warning: numba is not installed.")
    pass