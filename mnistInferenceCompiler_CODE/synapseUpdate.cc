#include "definitions.h"

struct MergedPresynapticUpdateGroup0
 {
    float* g;
    float* outPost;
    uint32_t* srcSpk;
    uint32_t* srcSpkCnt;
    uint32_t numSrcNeurons;
    uint32_t numTrgNeurons;
    uint32_t rowStride;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[2];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* g, float* outPost, uint32_t* srcSpk, uint32_t* srcSpkCnt, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride) {
    MergedPresynapticUpdateGroup0 group = {g, outPost, srcSpk, srcSpkCnt, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, 128, };
extern "C" __global__ void updatePresynapticKernel(float t)
 {
    const unsigned int batch = blockIdx.y;
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    // merged0
    if(id < 160) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedPresynapticUpdateGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedPresynapticUpdateGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        const unsigned int preBatchOffset = group->numSrcNeurons * batch;
        const unsigned int postBatchOffset = group->numTrgNeurons * batch;
        float linSyn = 0;
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
                    if (lid < group->rowStride) {
                        const uint32_t synAddress = ((uint32_t)shSpk[j] * group->rowStride) + lid;
                        linSyn += (group->g[synAddress]);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < group->numTrgNeurons) {
            group->outPost[postBatchOffset + lid] += linSyn;
        }
    }
}
void updateSynapses(float t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(5, 128);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
    }
}
