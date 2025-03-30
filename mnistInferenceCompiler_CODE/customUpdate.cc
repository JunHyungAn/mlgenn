#include "definitions.h"

struct MergedCustomUpdateGroup0
 {
    float* V;
    uint32_t numNeurons;
    
}
;
struct MergedCustomUpdateGroup1
 {
    uint32_t* Scount;
    float* V;
    
}
;
__device__ __constant__ MergedCustomUpdateGroup0 d_mergedCustomUpdateGroup0[2];
void pushMergedCustomUpdateGroup0ToDevice(unsigned int idx, float* V, uint32_t numNeurons) {
    MergedCustomUpdateGroup0 group = {V, numNeurons, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup0, &group, sizeof(MergedCustomUpdateGroup0), idx * sizeof(MergedCustomUpdateGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup1 d_mergedCustomUpdateGroup1[1];
void pushMergedCustomUpdateGroup1ToDevice(unsigned int idx, uint32_t* Scount, float* V) {
    MergedCustomUpdateGroup1 group = {Scount, V, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup1, &group, sizeof(MergedCustomUpdateGroup1), idx * sizeof(MergedCustomUpdateGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID0[] = {0, 106496, };
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID1[] = {122880, };
extern "C" __global__ void customUpdateReset(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged0
    if(id < 122880) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateGroup0 *group = &d_mergedCustomUpdateGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        const unsigned int paddedSize = 64 * ((group->numNeurons + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        const unsigned int batchOffset = group->numNeurons * batch;
        // only do this for existing neurons
        if(bid < group->numNeurons) {
            float _lV;
            _lV = 0.000000000e+00f;
            group->V[batchOffset + bid] = _lV;
        }
    }
    // merged1
    if(id >= 122880 && id < 131072) {
        const unsigned int lid = id - 122880;
        struct MergedCustomUpdateGroup1 *group = &d_mergedCustomUpdateGroup1[0]; 
        const unsigned int paddedSize = 64 * ((10u + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        const unsigned int batchOffset = 10u * batch;
        // only do this for existing neurons
        if(bid < 10u) {
            uint32_t _lScount;
            float _lV;
            _lV = 0.000000000e+00f;
            _lScount = 0u;
            group->Scount[batchOffset + bid] = _lScount;
            group->V[batchOffset + bid] = _lV;
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateReset(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(2048, 1);
        customUpdateReset<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
    }
}
