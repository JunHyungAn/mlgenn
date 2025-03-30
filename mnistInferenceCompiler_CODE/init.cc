#include "definitions.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    float* Input;
    float* V;
    uint32_t* spkCntSynSpike0;
    uint32_t* spkSynSpike0;
    
}
;
struct MergedNeuronInitGroup1
 {
    float* V;
    float* outPostInSyn0;
    uint32_t* spkCntSynSpike0;
    uint32_t* spkSynSpike0;
    
}
;
struct MergedNeuronInitGroup2
 {
    uint32_t* Scount;
    float* V;
    float* outPostInSyn0;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, float* Input, float* V, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0) {
    MergedNeuronInitGroup0 group = {Input, V, spkCntSynSpike0, spkSynSpike0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, float* V, float* outPostInSyn0, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0) {
    MergedNeuronInitGroup1 group = {V, outPostInSyn0, spkCntSynSpike0, spkSynSpike0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedNeuronInitGroup2 d_mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, uint32_t* Scount, float* V, float* outPostInSyn0) {
    MergedNeuronInitGroup2 group = {Scount, V, outPostInSyn0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup2, &group, sizeof(MergedNeuronInitGroup2), idx * sizeof(MergedNeuronInitGroup2), cudaMemcpyHostToDevice, 0));
}
void initializeHost() {
}
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {800, };
__device__ unsigned int d_mergedNeuronInitGroupStartID2[] = {928, };

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 800) {
        const unsigned int lid = id - 0;
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        // only do this for existing neurons
        if(lid < 784u) {
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 128; d++) {
                    group->V[(d * 784u) + lid] = initVal;
                }
            }
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 128; d++) {
                    group->Input[(d * 784u) + lid] = initVal;
                }
            }
            for(unsigned int d = 0; d < 128; d++) {
                group->spkSynSpike0[(d * 784u) + lid] = 0;
            }
            if(lid == 0) {
                for(unsigned int d = 0; d < 128; d++) {
                    group->spkCntSynSpike0[d] = 0;
                }
            }
        }
    }
    // merged1
    if(id >= 800 && id < 928) {
        const unsigned int lid = id - 800;
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        // only do this for existing neurons
        if(lid < 128u) {
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 128; d++) {
                    group->V[(d * 128u) + lid] = initVal;
                }
            }
            for(unsigned int d = 0; d < 128; d++) {
                group->spkSynSpike0[(d * 128u) + lid] = 0;
            }
            if(lid == 0) {
                for(unsigned int d = 0; d < 128; d++) {
                    group->spkCntSynSpike0[d] = 0;
                }
            }
            for(unsigned int d = 0; d < 128; d++) {
                group->outPostInSyn0[(d * 128u) + lid] = 0.000000000e+00f;
            }
        }
    }
    // merged2
    if(id >= 928 && id < 960) {
        const unsigned int lid = id - 928;
        struct MergedNeuronInitGroup2 *group = &d_mergedNeuronInitGroup2[0]; 
        // only do this for existing neurons
        if(lid < 10u) {
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 128; d++) {
                    group->V[(d * 10u) + lid] = initVal;
                }
            }
             {
                uint32_t initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 128; d++) {
                    group->Scount[(d * 10u) + lid] = initVal;
                }
            }
            for(unsigned int d = 0; d < 128; d++) {
                group->outPostInSyn0[(d * 10u) + lid] = 0.000000000e+00f;
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Synapse groups
    
    // ------------------------------------------------------------------------
    // Custom update groups
    
    // ------------------------------------------------------------------------
    // Custom WU update groups
    
    // ------------------------------------------------------------------------
    // Custom connectivity presynaptic update groups
    
    // ------------------------------------------------------------------------
    // Custom connectivity postsynaptic update groups
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    
}
void initialize() {
    unsigned long long deviceRNGSeed = 0;
     {
        const dim3 threads(32, 1);
        const dim3 grid(30, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
}
