//BACKTRACK TO RECOVER PATH

__kernel
void path(__global const float* llike_trow, __global const float* transition, __global const int path_tplus1, const int N_STATES, __global int p, __global int s_out){
	int s = get_global_id(0);

	if (s < N_STATES){
		p = llike_trow[s] + transition[s * N_STATES + path_tplus1];
		s_out = s;
	}

//Each thread will calculate p, host needs to compare p of different threads and select the minimum (and its corresponding state)

};

