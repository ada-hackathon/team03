/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
//OpenCL utility layer include
#include "xcl2.hpp"
#include "viterbi.h"
#include <iostream>
#include <vector>
#include <iostream> 
#include <math.h>

#define DETERMINISTIC_TRANSITION 1
using namespace std;

void generate_inputs(struct bench_args_t data) {
  int fd;
  state_t s0, s1, s[N_OBS];
  tok_t o;
  step_t t;
  double P[N_STATES];
  double Q[N_TOKENS];
  struct prng_rand_t state;
#if !defined(DETERMINISTIC_TRANSITION) || !defined(DETERMINISTIC_EMISSION)
  prob_t r;
#endif
#if defined(DETERMINISTIC_TRANSITION) || defined(DETERMINISTIC_TRANSITION)
  prob_t min;
  tok_t omin;
#endif

  // Fill data structure
  prng_srand(1,&state);

  // Generate a random transition matrix P(S1|S0)
  // Invariant: SUM_S1 P(S1|S0) = 1
  for(s0=0; s0<N_STATES; s0++) {
    // Generate random weights
    double sum = 0;
    for(s1=0; s1<N_STATES; s1++) {
      P[s1] = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
#ifndef DETERMINISTIC_TRANSITION
      if(s1==s0) P[s1] = N_STATES; // self-transitions are much more likely
#else
      if(s1==s0) P[s1] = 0.5/N_STATES; // self-transitions are less likely (otherwise we'd never change states)
#endif
      sum += P[s1];
    }
    // Normalize and convert to -log domain
    for(s1=0; s1<N_STATES; s1++) {
      data.transition[s0*N_STATES+s1] = -1*logf(P[s1]/sum);
    }
  }

  // Generate a random emission matrix P(O|S)
  // Invariant: SUM_O P(O|S) = 1
  for(s0=0; s0<N_STATES; s0++) {
    // Generate random weights
    double sum = 0;
    for(o=0; o<N_TOKENS; o++) {
      Q[o] = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
      if( o==s0 ) Q[o] = N_TOKENS; // one token is much more likely
      sum += Q[o];
    }
    // Normalize and convert to -log domain
    for(o=0; o<N_TOKENS; o++) {
      data.emission[s0*N_TOKENS+o] = -1*logf(Q[o]/sum);
    }
  }

  // Generate a random starting distribution P(S_0)
  // Invariant: SUM P(S_0) = 1
  {
    // Generate random weights
    double sum = 0;
    for(s0=0; s0<N_STATES; s0++) {
      P[s0] = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
      sum += P[s0];
    }
    // Normalize and convert to -log domain
    for(s0=0; s0<N_STATES; s0++) {
      data.init[s0] = -1*logf(P[s0]/sum);
    }
  }


  // To get observations, just run the HMM forwards N_OBS steps;
  // Nondeterministic sampling uses the inverse transform method

  // Sample s_0 from init
#ifndef DETERMINISTIC_TRANSITION
  r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
  s[0]=0; do{r-=expf(-data.init[s[0]]);} while(r>0&&(++s[0]));
#else
  s[0]=0;
  min=data.init[0];
  for(s0=1;s0<N_STATES;s0++) {
    if(data.init[s0]<min) {
      min=data.init[s0];
      s[0]=s0;
    }
  }
#endif
  // Sample o_0 from emission
#ifndef DETERMINISTIC_EMISSION
  r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
  for( o=0; r>0; ++o ) {r-=expf(-data.emission[s[0]*N_TOKENS+o]);}
  o=0; do{ r-=expf(-data.emission[s[0]*N_TOKENS+o]); }while(r>0 && ++o);
  data.obs[0] = o;
#else
  omin=0;
  min=data.emission[s[0]*N_TOKENS+0];
  for(o=1;o<N_TOKENS;o++) {
    if(data.emission[s[0]*N_TOKENS+o]<min) {
      min=data.emission[s[0]*N_TOKENS+o];
      omin=o;
    }
  }
  data.obs[0] = omin;
#endif

  for(t=1; t<N_OBS; t++) {
    // Sample s_t from transition
#ifndef DETERMINISTIC_TRANSITION
    r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
    s[t]=0; do{r-=expf(-data.transition[s[t-1]*N_STATES+s[t]]);} while(r>0&&++s[t]);
#else
    s[t]=0;
    min=data.transition[s[t-1]*N_STATES];
    for(s0=1;s0<N_STATES;s0++) {
      if(data.transition[s[t-1]*N_STATES+s0]<min) {
        min=data.transition[s[t-1]*N_STATES+s0];
        s[t]=s0;
      }
    }
#endif
    // Sample o_t from emission
#ifndef DETERMINISTIC_EMISSION
    r = ((double)prng_rand(&state)/(double)PRNG_RAND_MAX);
    o=0; do{ r-=expf(-data.emission[s[t]*N_TOKENS+o]); }while(r>0 && ++o);
    data.obs[t] = o;
#else
    omin=0;
    min=data.emission[s[t]*N_TOKENS+0];
    for(o=1;o<N_TOKENS;o++) {
      if(data.emission[s[t]*N_TOKENS+o]<min) {
        min=data.emission[s[t]*N_TOKENS+o];
        omin=o;
      }
    }
    data.obs[t] = omin;
#endif
  }

  #ifdef DEBUG
  dump_path(s);
  dump_obs(data.obs);
  #endif

  
}


int main(int argc, char** argv)
{
    //Allocate Memory in Host Memory
    size_t obs_size_bytes = sizeof(tok_t) * N_OBS;
    size_t init_size_bytes = sizeof(prob_t) * N_STATES;
    size_t trans_size_bytes = sizeof(prob_t) * N_STATES*N_STATES;
    size_t emission_size_bytes = sizeof(prob_t) * N_STATES*N_TOKENS;
    size_t path_size_bytes = sizeof(state_t) * N_OBS;

    struct bench_args_t data;
    generate_inputs(data);

    //Initialize inputs
    std::vector<tok_t,aligned_allocator<tok_t>> obs     (N_OBS);
    std::vector<prob_t,aligned_allocator<prob_t>> init     (N_STATES);
    std::vector<prob_t,aligned_allocator<prob_t>> transition (N_STATES*N_STATES);
    std::vector<prob_t,aligned_allocator<prob_t>> emission (N_STATES*N_TOKENS);
    std::vector<state_t,aligned_allocator<state_t>> path(N_OBS);

    //Reference output
    std::vector<state_t,aligned_allocator<state_t>> path_ref(N_OBS);

    //Populate reference from check.data
    std::ifstream check("check.data");
    std::string str;
    for(int i = 0 ; i < N_OBS ; i++){
	    std::getline(check, str);
	    path_ref[i] = std::stoi(str, null, 10);
    }

    // Copy "data"
    obs.assign(data.obs, data.obs + N_OBS);
    init.assign(data.init, data.init + N_STATES);
    transition.assign(data.transition, data.transition + N_STATES*N_STATES);
    emission.assign(data.emission, data.emission + N_STATES*N_TOKENS);
 

  //  for(int i= 0;i< DATA_SIZE;i++)
   // 	cout<<source_sw_results[i]<<endl;
//OPENCL HOST CODE AREA START
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 

    //TODO: Modify here
    //Create Program and Kernel
    std::string binaryFile = xcl::find_binary_file(device_name,"init");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl_mult(program,"init");

    //Allocate Buffer in Global Memory
    cl::Buffer buffer_obs (context, CL_MEM_READ_ONLY,
                        obs_size_bytes);
    cl::Buffer buffer_init (context, CL_MEM_READ_ONLY,
                        init_size_bytes);
    cl::Buffer buffer_trans (context, CL_MEM_READ_ONLY,
                        trans_size_bytes);
    cl::Buffer buffer_emission (context, CL_MEM_READ_ONLY,
                        emission_size_bytes);
    cl::Buffer buffer_path (context, CL_MEM_WRITE_ONLY,
                           path_size_bytes);
    

    //Copy input data to device global memory
    q.enqueueWriteBuffer(buffer_obs, CL_TRUE, 0, obs_size_bytes, obs.data());
    q.enqueueWriteBuffer(buffer_init, CL_TRUE, 0, init_size_bytes, init.data());
    q.enqueueWriteBuffer(buffer_trans, CL_TRUE, 0, trans_size_bytes, transition.data());
    q.enqueueWriteBuffer(buffer_emission, CL_TRUE, 0, emission_size_bytes, emission.data());


    // TODO:Modify here
   // int inc = INCR_VALUE;
    int size = N_STATES;
    int obs0 = 1;
    //Set the Kernel Arguments
    int narg=0;
    krnl_mult.setArg(narg++,buffer_init);
    krnl_mult.setArg(narg++,buffer_emission);
    krnl_mult.setArg(narg++,buffer_path);
    krnl_mult.setArg(narg++,size);
    krnl_mult.setArg(narg++,obs0);

    //Launch the Kernel
    q.enqueueNDRangeKernel(krnl_mult,cl::NullRange,cl::NDRange(1,size),cl::NullRange);

    //Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(buffer_path, CL_TRUE, 0, path_size_bytes, path.data());

    q.finish();
    //TODO:Finish

//OPENCL HOST CODE AREA END
    
    // Compare the results of the Device to the simulation
    bool match = true;
    for (int i = 0 ; i < N_OBS ; i++){
        if (path[i] != path_ref[i]){
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << path_ref[i]
                << " Device result = " << path[i] << std::endl;
            match = false;
          break;
       }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
}
