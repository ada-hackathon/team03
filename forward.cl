__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))

void forward(__global float **llike, __global float *transition, __global float *emission,
    uint n_states, uint n_tokens, int t, uint obs)
{
    uint tid = get_global_id(0);

    float transition_local;
    uint8 curr = tid;
    float emission_local = emission[curr*n_tokens+obs];
    uint8 prev;
    float p;
    float min_p;

    prev = 0;
    min_p = llike[t-1][prev] +
            transition[prev*n_states+curr] +
            emission_local;
    for (prev = 1; prev < n_states; prev++) {
        p = llike[t-1][prev] +
            transition[prev*n_states+curr] +
            emission_local;
        if (p < min_p) {
            min_p = p;
        }
    }
    llike[t][curr] = min_p;
}