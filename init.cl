//INITIALIZE WITH FIRST OBSERVATION AND INITIAL PROBABILITIES

__kernel
void init(__global const float* init, __global const float* emission, __global  float* llike, const int N_TOKENS, const int obs_0){
	int s = get_global_id(0);
	
	// Each thread initializes 1 element
	if (s < N_STATES)
		llike[0 + s] = init[s] + emission[s*N_TOKENS + obs_0];

//	// Each thread initializes 16 elements
//	while (s < N_STATES){
//		llike[0 + s] = init[s] + emission[s*N_TOKENS + obs_0];
//		s += 16;
//	}
};

