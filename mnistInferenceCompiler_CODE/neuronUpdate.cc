#include "definitions.h"

struct MergedNeuronUpdateGroup0
 {
    float* Input;
    float* V;
    uint32_t* spkCntSynSpike0;
    uint32_t* spkSynSpike0;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    float* V;
    float* outPostInSyn0;
    uint32_t* spkCntSynSpike0;
    uint32_t* spkSynSpike0;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    uint32_t* Scount;
    float* V;
    float* outPostInSyn0;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    uint32_t* spkCntSynSpike0;
    
}
;
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup0 d_mergedNeuronSpikeQueueUpdateGroup0[2];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, uint32_t* spkCntSynSpike0) {
    MergedNeuronSpikeQueueUpdateGroup0 group = {spkCntSynSpike0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup0, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup0), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedNeuronUpdateGroup0 d_mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, float* Input, float* V, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0) {
    MergedNeuronUpdateGroup0 group = {Input, V, spkCntSynSpike0, spkSynSpike0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, float* V, float* outPostInSyn0, uint32_t* spkCntSynSpike0, uint32_t* spkSynSpike0) {
    MergedNeuronUpdateGroup1 group = {V, outPostInSyn0, spkCntSynSpike0, spkSynSpike0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedNeuronUpdateGroup2 d_mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, uint32_t* Scount, float* V, float* outPostInSyn0) {
    MergedNeuronUpdateGroup2 group = {Scount, V, outPostInSyn0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &group, sizeof(MergedNeuronUpdateGroup2), idx * sizeof(MergedNeuronUpdateGroup2), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {832, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID2[] = {960, };

extern "C" __global__ void neuronSpikeQueueUpdateKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    if(id < 2) {
        struct MergedNeuronSpikeQueueUpdateGroup0 *group = &d_mergedNeuronSpikeQueueUpdateGroup0[id - 0]; 
        for(unsigned int batch = 0; batch < 128; batch++) {
             {
                // spike queue update 0
                group->spkCntSynSpike0[batch] = 0;
            }
        }
    }
}

extern "C" __global__ void updateNeuronsKernel(float t)
 {
    const unsigned int batch = blockIdx.y;
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[1][64];
    __shared__ unsigned int shSpkPos[1];
    __shared__ unsigned int shSpkCount[1];
    if (threadIdx.x == 0) {
        shSpkCount[0] = 0;
    }
    
    __syncthreads();
    // merged0
    if(id < 832) {
        const unsigned int lid = id - 0;
        struct MergedNeuronUpdateGroup0 *group = &d_mergedNeuronUpdateGroup0[0]; 
         {
            const unsigned int batchOffset = 784u * batch;
            if(lid < 784u) {
                float _lV = group->V[batchOffset + lid];
                const float _lInput = group->Input[batchOffset + lid];
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                _lV += _lInput;
                // test for and register a true spike
                if ((_lV >= 5.000000000e+00f)) {
                    const unsigned int eventIdx = atomicAdd(&shSpkCount[0], 1);
                    shSpk[0][eventIdx] = lid;
                    // spike reset code
                    _lV = 0.000000000e+00f;
                }
                group->V[batchOffset + lid] = _lV;
            }
            __syncthreads();
            if(threadIdx.x == 0) {
                 {
                    shSpkPos[0] = atomicAdd(&group->spkCntSynSpike0[batch], shSpkCount[0]);
                }
            }
            __syncthreads();
            if(threadIdx.x < shSpkCount[0]) {
                const unsigned int n = shSpk[0][threadIdx.x];
                 {
                    group->spkSynSpike0[batchOffset + shSpkPos[0] + threadIdx.x] = n;
                }
            }
        }
    }
    // merged1
    if(id >= 832 && id < 960) {
        const unsigned int lid = id - 832;
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
         {
            const unsigned int batchOffset = 128u * batch;
            if(lid < 128u) {
                float Isyn = 0;
                float _lV = group->V[batchOffset + lid];
                 {
                    // postsynaptic model 0
                    float linSyn = group->outPostInSyn0[batchOffset + lid];
                    Isyn += linSyn;
                    linSyn = 0;
                    group->outPostInSyn0[batchOffset + lid] = linSyn;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                _lV += Isyn;
                // test for and register a true spike
                if ((_lV >= 5.000000000e+00f)) {
                    const unsigned int eventIdx = atomicAdd(&shSpkCount[0], 1);
                    shSpk[0][eventIdx] = lid;
                    // spike reset code
                    _lV = 0.000000000e+00f;
                }
                group->V[batchOffset + lid] = _lV;
            }
            __syncthreads();
            if(threadIdx.x == 0) {
                 {
                    shSpkPos[0] = atomicAdd(&group->spkCntSynSpike0[batch], shSpkCount[0]);
                }
            }
            __syncthreads();
            if(threadIdx.x < shSpkCount[0]) {
                const unsigned int n = shSpk[0][threadIdx.x];
                 {
                    group->spkSynSpike0[batchOffset + shSpkPos[0] + threadIdx.x] = n;
                }
            }
        }
    }
    // merged2
    if(id >= 960 && id < 1024) {
        const unsigned int lid = id - 960;
        struct MergedNeuronUpdateGroup2 *group = &d_mergedNeuronUpdateGroup2[0]; 
         {
            const unsigned int batchOffset = 10u * batch;
            if(lid < 10u) {
                float Isyn = 0;
                float _lV = group->V[batchOffset + lid];
                uint32_t _lScount = group->Scount[batchOffset + lid];
                 {
                    // postsynaptic model 0
                    float linSyn = group->outPostInSyn0[batchOffset + lid];
                    Isyn += linSyn;
                    linSyn = 0;
                    group->outPostInSyn0[batchOffset + lid] = linSyn;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                _lV += Isyn;
                // test for and register a true spike
                if ((_lV >= 5.000000000e+00f)) {
                    // spike reset code
                    _lV = 0.000000000e+00f;
                    _lScount++;
                }
                group->V[batchOffset + lid] = _lV;
                group->Scount[batchOffset + lid] = _lScount;
            }
            __syncthreads();
        }
    }
}
void updateNeurons(float t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        neuronSpikeQueueUpdateKernel<<<grid, threads>>>();
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(64, 1);
        const dim3 grid(16, 128);
        updateNeuronsKernel<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
    }
}
