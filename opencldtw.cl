/*  
 * Copyright (C) 2024 Wuhan University of Technology
 * Authors: Wang Zihao <qianlkzf@outlook.com> 
 *  
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, either version 3 of the License, or  
 * (at your option) any later version.  
 *  
 * This program is distributed in the hope that it will be useful,  
 * but WITHOUT ANY WARRANTY; without even the implied warranty of  
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the  
 * GNU General Public License for more details.  
 *  
 * You should have received a copy of the GNU General Public License  
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.  
 */
 
 __kernel void opencl_dtw (
unsigned SRC_LEN,
//unsigned TRG_LEN,
__global const float *S,
__global const float *TT,
__global float       *RR,
__local float path_h1[],
__local float path_h2[],
__local float dist[]
)
{
    int i, j;    
    __local float *ex;
    
    unsigned int row_id = get_local_id(0);    
    unsigned int row_x  = get_local_id(1);
    unsigned int Len_x  = get_local_size(1); // TRG_LEN
    int row_start_id = row_id *Len_x;
    //int row_x  = row_id % TRG_LEN;

    int t_gid = get_group_id(0) *get_local_size(0) *get_local_size(1)
               + row_start_id + row_x;
    int buf_id = row_start_id + row_x;
    
    // 1.   first line speical, do first
    // 1.1. paralle, first line's every element's dist
    dist[buf_id] = (S[0] -TT[t_gid])
                  *(S[0] -TT[t_gid]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // 1.2. serial, first line's every element's serial's dist
    if (row_x == 0){
        // use row_x == 0 to limit do one element only
        path_h1[buf_id] = dist[buf_id];
        for (i=1; i <Len_x; i++) {
            path_h1[i +row_start_id] = path_h1[i-1 +row_start_id]
                                     +    dist[i   +row_start_id];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 2. do the rest lines by circle
    for (i=1; i <SRC_LEN; i++){ 
        // 2.1. paralle, calc itself's DISTANCE, for speed follow progress
        //    use memeory to rise speed
        dist[buf_id] = (S[i] -TT[t_gid])
                      *(S[i] -TT[t_gid]);
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2.2 paralle, get from upper line's "up","left-up"- the min dist
        if (row_x == 0) 
            // FIRST element speical, 
            // just use his down element add DISTANCE only
            path_h2[buf_id] =  path_h1[buf_id] + dist[buf_id];
        else
            // Different from the traditional method of taking 
            // the minimum value of the three elements, 
            // in order to achieve better parallel efficiency, 
            // we first compare the minimum value of the elements 
            // on the left bottom and the bottom, and then cache it in path_h2.
            path_h2[buf_id] = min (path_h1[buf_id   ],
                                   path_h1[buf_id -1]);
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2.3 serial, compare to left(front) element with myself, 
        //     get the less value, so done the compartion of three elements.
        if (row_x == 0)
        // the first element had stored in dist, so can use here
            for (j=1; j<Len_x; j++) 
                path_h2[j +row_start_id] = min(path_h2[j   +row_start_id], 
                                               path_h2[j-1 +row_start_id])
                                         + dist[j +row_start_id];
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2.4 swap the two path array to prepare the next line caculation
        ex      = path_h2;
        path_h2 = path_h1;
        path_h1 = ex;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // when all done, can return the result
    if (row_x == 0){
        RR[get_global_id(0)] = 
            sqrt(ex[row_start_id +Len_x -1]);
    }
}

__kernel void opencl_dtw_low_resource (
unsigned SRC_LEN,
unsigned TRG_LEN,
__global const float *S,
__global const float *TT,
__global float       *RR,

__local float path_h1[],
__local float path_h2[],
__local float dist[]
)
{
    /* Due to limited work items number, the work can only be carried out 
       in parallel blocks as much as possible, and cannot be completed 
       in one parallel operation. */
    int i, j, k;
    __local float *ex;
    
    unsigned int row_id  = get_local_id(0);    
    // row_id is just only one, threads shall be in 2nd dimension
    unsigned int row_x   = get_local_id(1);
    unsigned int row_Len = get_local_size(0);  // it must be 1
    unsigned int x_step  = get_local_size(1);
    unsigned int cCircle = TRG_LEN / x_step;
    unsigned int rCircle = TRG_LEN % x_step;
    unsigned int buf_id;
    unsigned int t_gid;
    unsigned int t_gid_Base = get_group_id(0) *get_local_size(0) *TRG_LEN;
    // There's a bit of a stretch here, perhaps in the future, 
    // there could be more than one non-contiguous row_id, 
    // so we still perform calculations on "row_start_id" that would otherwise be zero.
    unsigned int row_start_id = row_id *TRG_LEN;
    
    // 1.   first line speical, do first
    // 1.1. paralle, first line's every element's dist    
    for (k=0; k < cCircle; k++){
        buf_id = row_id*TRG_LEN + k*x_step + row_x;
        t_gid  = t_gid_Base + buf_id;
        dist[buf_id] = (S[0] -TT[t_gid])
                     * (S[0] -TT[t_gid]);
    }
    if ((rCircle > 0) && (row_x < rCircle)){
        buf_id = row_id*TRG_LEN + cCircle*x_step + row_x;
        t_gid  = t_gid_Base + buf_id;
        dist[buf_id] = (S[0] -TT[t_gid])
                     * (S[0] -TT[t_gid]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 1.2. serial, first line's every element's serial's dist
    if (row_x == 0){
        // use row_x == 0 to limit do one element only
        buf_id = row_id*TRG_LEN + row_x;
        path_h1[buf_id] = dist[buf_id];
        for (i=1; i <TRG_LEN; i++) {
            path_h1[i +row_start_id] = path_h1[i-1 +row_start_id]
                                     +    dist[i   +row_start_id];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /*
    if (get_local_id(0) == 0)
        printf ("group_0 %d ,local_size %d\n", get_group_id(0) ,get_local_size(0));
    */
    // 2. do the rest lines by circle
    for (i=1; i <SRC_LEN; i++){ 
        // 2.1. paralle, calc itself's DISTANCE, for speed follow progress
        //    use memeory to rise speed
        for (k=0; k < cCircle; k++){
            buf_id = row_id*TRG_LEN + k*x_step + row_x;
            t_gid  = t_gid_Base + buf_id;
            dist[buf_id] = (S[i] -TT[t_gid])
                         * (S[i] -TT[t_gid]);
        }
        if ((rCircle > 0) && (row_x < rCircle)){
            buf_id = row_id*TRG_LEN + cCircle*x_step + row_x;
            t_gid  = t_gid_Base + buf_id;
            dist[buf_id] = (S[i] -TT[t_gid])
                         * (S[i] -TT[t_gid]);
        }
        /*
        dist[buf_id] = (S[i] -TT[t_gid])
                      *(S[i] -TT[t_gid]);
        */
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2.2 paralle, get from upper line's "up","left-up"- the min dist
        /*
        if (row_x == 0) 
            path_h2[buf_id] =  path_h1[buf_id] + dist[buf_id];
        else
            path_h2[buf_id] = min (path_h1[buf_id   ],
                                   path_h1[buf_id -1]);
        */
        for (k=0; k < cCircle; k++){
            if ((k == 0)&&(row_x == 0)){
                // FIRST element speical, 
                // just use his down element add DISTANCE only
                path_h2[row_start_id] =  path_h1[row_start_id] + dist[row_start_id];
            } else {
                buf_id = row_id*TRG_LEN + k*x_step + row_x;
                path_h2[buf_id] = min (path_h1[buf_id   ],
                                       path_h1[buf_id -1]);
            }
        }
        if ((rCircle > 0) && (row_x < rCircle)){
            buf_id = row_id*TRG_LEN + cCircle*x_step + row_x;
            path_h2[buf_id] = min (path_h1[buf_id   ],
                                   path_h1[buf_id -1]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2.3 serial, compare to left(front) element with myself, 
        //     get the less value, so done the compartion of three elements.
        if (row_x == 0)
        // the first element had stored in dist, so can use here
            for (j=1; j<TRG_LEN; j++) 
                path_h2[j +row_start_id] = min(path_h2[j   +row_start_id], 
                                               path_h2[j-1 +row_start_id])
                                         + dist[j +row_start_id];
        barrier(CLK_LOCAL_MEM_FENCE);
        // 2.4 swap the two path array to prepare the next line caculation
        ex      = path_h2;
        path_h2 = path_h1;
        path_h1 = ex;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // when all done, can return the result
    if (row_x == 0){
        RR[get_global_id(0)] = 
            sqrt(ex[row_start_id +TRG_LEN -1]);
    }
}