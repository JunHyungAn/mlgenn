#include "definitions.h"

struct MergedPresynapticUpdateGroup0
 {
    float* Gradient;
    float* LambdaI_post;
    float* g;
    float* outPost;
    uint32_t* srcSpk;
    uint32_t* srcSpkCnt;
    uint32_t* srcSpkCntEvent;
    uint32_t* srcSpkEvent;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    float* Gradient;
    float* LambdaI_post;
    float* LambdaV_post;
    float* g;
    float* outPost;
    float* outPre;
    uint32_t* srcSpk;
    uint32_t* srcSpkCnt;
    uint32_t* srcSpkCntEvent;
    uint32_t* srcSpkEvent;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[1];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* Gradient, float* LambdaI_post, float* g, float* outPost, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t* srcSpkCntEvent, uint32_t* srcSpkEvent) {
    MergedPresynapticUpdateGroup0 group = {Gradient, LambdaI_post, g, outPost, srcSpk, srcSpkCnt, srcSpkCntEvent, srcSpkEvent, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedPresynapticUpdateGroup1 d_mergedPresynapticUpdateGroup1[1];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* Gradient, float* LambdaI_post, float* LambdaV_post, float* g, float* outPost, float* outPre, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t* srcSpkCntEvent, uint32_t* srcSpkEvent) {
    MergedPresynapticUpdateGroup1 group = {Gradient, LambdaI_post, LambdaV_post, g, outPost, outPre, srcSpk, srcSpkCnt, srcSpkCntEvent, srcSpkEvent, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup1, &group, sizeof(MergedPresynapticUpdateGroup1), idx * sizeof(MergedPresynapticUpdateGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID1[] = {128, };
extern "C" __global__ void updatePresynapticKernel(float t)
 {
    const unsigned int batch = blockIdx.y;
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shSpkEvent[32];
    // merged0
    if(id < 128) {
        const unsigned int lid = id - 0;
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[0]; 
        const unsigned int preBatchOffset = 784u * batch;
        const unsigned int postBatchOffset = 128u * batch;
        const uint32_t synBatchOffset = (uint32_t)preBatchOffset * 128u;
        float linSyn = 0;
         {
            const unsigned int numSpikes = group->srcSpkCntEvent[batch];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpkEvent[preBatchOffset + (r * 32) + threadIdx.x];
                    shSpkEvent[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 128u) {
                        const uint32_t synAddress = ((uint32_t)shSpkEvent[j] * 128u) + lid;
                        group->Gradient[synBatchOffset + synAddress] -= (group->LambdaI_post[postBatchOffset + lid] * 5.000000000e+00f);
                    }
                }
            }
        }
         {
            const unsigned int numSpikes = group->srcSpkCnt[batch];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[preBatchOffset + (r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 128u) {
                        const uint32_t synAddress = ((uint32_t)shSpk[j] * 128u) + lid;
                        linSyn += (group->g[synAddress]);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 128u) {
            group->outPost[postBatchOffset + lid] += linSyn;
        }
    }
    // merged1
    if(id >= 128 && id < 160) {
        const unsigned int lid = id - 128;
        struct MergedPresynapticUpdateGroup1 *group = &d_mergedPresynapticUpdateGroup1[0]; 
        const unsigned int preBatchOffset = 128u * batch;
        const unsigned int postBatchOffset = 10u * batch;
        const uint32_t synBatchOffset = (uint32_t)preBatchOffset * 10u;
        float linSyn = 0;
         {
            const unsigned int numSpikes = group->srcSpkCntEvent[batch];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpkEvent[preBatchOffset + (r * 32) + threadIdx.x];
                    shSpkEvent[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    float lOutPre = 0.000000000e+00f;
                    // only work on existing neurons
                    if (lid < 10u) {
                        const uint32_t synAddress = ((uint32_t)shSpkEvent[j] * 10u) + lid;
                        group->Gradient[synBatchOffset + synAddress] -= (group->LambdaI_post[postBatchOffset + lid] * 5.000000000e+00f);
                        lOutPre += (group->g[synAddress] * (group->LambdaV_post[postBatchOffset + lid] - group->LambdaI_post[postBatchOffset + lid]));
                    }
                    lOutPre += __shfl_down_sync(0xFFFFFFFF, lOutPre, 16);
                    lOutPre += __shfl_down_sync(0xFFFFFFFF, lOutPre, 8);
                    lOutPre += __shfl_down_sync(0xFFFFFFFF, lOutPre, 4);
                    lOutPre += __shfl_down_sync(0xFFFFFFFF, lOutPre, 2);
                    lOutPre += __shfl_down_sync(0xFFFFFFFF, lOutPre, 1);
                    if((threadIdx.x % 32) == 0) {
                        atomicAdd(&group->outPre[preBatchOffset + shSpkEvent[j]], lOutPre);
                    }
                }
            }
        }
         {
            const unsigned int numSpikes = group->srcSpkCnt[batch];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[preBatchOffset + (r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < 10u) {
                        const uint32_t synAddress = ((uint32_t)shSpk[j] * 10u) + lid;
                        linSyn += (group->g[synAddress]);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < 10u) {
            group->outPost[postBatchOffset + lid] += linSyn;
        }
    }
}
void updateSynapses(float t) {
     {
        CHECK_RUNTIME_ERRORS(cudaEventRecord(presynapticUpdateStart));
        const dim3 threads(32, 1);
        const dim3 grid(5, 32);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(presynapticUpdateStop));
    }
}
