// Min reduction kernel in OpenCL. Assumes 1D workgroups in a 1D range.

#define BLOCK_SIZE 512

__kernel void reduceMin(__global const float * const llike, const unsigned int size, __global float * const min)
{
  __local like[BLOCK_SIZE];
  __local state[BLOCK_SIZE];
  const unsigned int gid = get_global_id(0);
  const unsigned int tid = get_local_id(0);
  const unsigned int bid = get_group_id(0);
  const unsigned int bsize = get_local_size(0);

  // Load into shared memory
  like[tid] = FLOAT_MAX;
  if (gid < size) {
    like[tid] = llike[gid];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Reduce!
  for (unsigned int i = bsize/2; i > 0; i >>= 1) {
    if (tid < i && gid < size) {
      if (like[tid] < like[tid + i]) {
        like[tid] = like[tid];
	state[tid] = tid;
      } else {
        like[tid] = like[tid + i];
	state[tid] = tid + i;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // One thread per workgroup writes back and the CPU then does the rest
  if (tid == 0) {
    min[bid] = state[0];
  }
}
