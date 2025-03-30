#include "definitions.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    uint8_t* BackSpike;
    uint32_t* EndSpike;
    int32_t* RingReadEndOffset;
    int32_t* RingReadOffset;
    int32_t* RingWriteOffset;
    int32_t* RingWriteStartOffset;
    uint32_t* StartSpike;
    uint32_t* spkCntEventSynSpikeEvent0;
    uint32_t* spkCntSynSpike0;
    uint32_t* spkEventSynSpikeEvent0;
    uint32_t* spkSynSpike0;
    
}
;
struct MergedNeuronInitGroup1
 {
    uint8_t* BackSpike;
    float* LambdaI;
    float* LambdaV;
    int32_t* RingReadEndOffset;
    int32_t* RingReadOffset;
    int32_t* RingWriteOffset;
    int32_t* RingWriteStartOffset;
    float* V;
    float* outPostInSyn0;
    float* outPreOutSyn0;
    uint32_t* spkCntEventSynSpikeEvent0;
    uint32_t* spkCntSynSpike0;
    uint32_t* spkEventSynSpikeEvent0;
    uint32_t* spkSynSpike0;
    
}
;
struct MergedNeuronInitGroup2
 {
    float* LambdaI;
    float* LambdaV;
    float* Softmax;
    float* V;
    float* VAvg;
    uint8_t* YTrue;
    uint8_t* YTrueBack;
    float* outPostInSyn0;
    
}
;
struct MergedSynapseInitGroup0
 {
    float* Gradient;
    float* g;
    float meang;
    uint32_t numSrcNeurons;
    uint32_t numTrgNeurons;
    uint32_t rowStride;
    float sdg;
    
}
;
struct MergedCustomUpdateInitGroup0
 {
    float* MaxVal;
    
}
;
struct MergedCustomUpdateInitGroup1
 {
    float* SumExpVal;
    
}
;
struct MergedCustomWUUpdateInitGroup0
 {
    float* ReducedGradient;
    uint32_t numSrcNeurons;
    uint32_t numTrgNeurons;
    uint32_t rowStride;
    
}
;
struct MergedCustomWUUpdateInitGroup1
 {
    float* M;
    float* V;
    uint32_t numSrcNeurons;
    uint32_t numTrgNeurons;
    uint32_t rowStride;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, uint8_t* BackSpike, uint32_t* EndSpike, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, uint32_t* StartSpike, uint32_t* spkCntEventSynSpikeEvent0, uint32_t* spkCntSynSpike0, uint32_t* spkEventSynSpikeEvent0, uint32_t* spkSynSpike0) {
    MergedNeuronInitGroup0 group = {BackSpike, EndSpike, RingReadEndOffset, RingReadOffset, RingWriteOffset, RingWriteStartOffset, StartSpike, spkCntEventSynSpikeEvent0, spkCntSynSpike0, spkEventSynSpikeEvent0, spkSynSpike0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, uint8_t* BackSpike, float* LambdaI, float* LambdaV, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, float* V, float* outPostInSyn0, float* outPreOutSyn0, uint32_t* spkCntEventSynSpikeEvent0, uint32_t* spkCntSynSpike0, uint32_t* spkEventSynSpikeEvent0, uint32_t* spkSynSpike0) {
    MergedNeuronInitGroup1 group = {BackSpike, LambdaI, LambdaV, RingReadEndOffset, RingReadOffset, RingWriteOffset, RingWriteStartOffset, V, outPostInSyn0, outPreOutSyn0, spkCntEventSynSpikeEvent0, spkCntSynSpike0, spkEventSynSpikeEvent0, spkSynSpike0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedNeuronInitGroup2 d_mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, float* LambdaI, float* LambdaV, float* Softmax, float* V, float* VAvg, uint8_t* YTrue, uint8_t* YTrueBack, float* outPostInSyn0) {
    MergedNeuronInitGroup2 group = {LambdaI, LambdaV, Softmax, V, VAvg, YTrue, YTrueBack, outPostInSyn0, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup2, &group, sizeof(MergedNeuronInitGroup2), idx * sizeof(MergedNeuronInitGroup2), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedSynapseInitGroup0 d_mergedSynapseInitGroup0[2];
void pushMergedSynapseInitGroup0ToDevice(unsigned int idx, float* Gradient, float* g, float meang, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride, float sdg) {
    MergedSynapseInitGroup0 group = {Gradient, g, meang, numSrcNeurons, numTrgNeurons, rowStride, sdg, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseInitGroup0, &group, sizeof(MergedSynapseInitGroup0), idx * sizeof(MergedSynapseInitGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateInitGroup0 d_mergedCustomUpdateInitGroup0[1];
void pushMergedCustomUpdateInitGroup0ToDevice(unsigned int idx, float* MaxVal) {
    MergedCustomUpdateInitGroup0 group = {MaxVal, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateInitGroup0, &group, sizeof(MergedCustomUpdateInitGroup0), idx * sizeof(MergedCustomUpdateInitGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateInitGroup1 d_mergedCustomUpdateInitGroup1[1];
void pushMergedCustomUpdateInitGroup1ToDevice(unsigned int idx, float* SumExpVal) {
    MergedCustomUpdateInitGroup1 group = {SumExpVal, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateInitGroup1, &group, sizeof(MergedCustomUpdateInitGroup1), idx * sizeof(MergedCustomUpdateInitGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomWUUpdateInitGroup0 d_mergedCustomWUUpdateInitGroup0[2];
void pushMergedCustomWUUpdateInitGroup0ToDevice(unsigned int idx, float* ReducedGradient, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride) {
    MergedCustomWUUpdateInitGroup0 group = {ReducedGradient, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomWUUpdateInitGroup0, &group, sizeof(MergedCustomWUUpdateInitGroup0), idx * sizeof(MergedCustomWUUpdateInitGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomWUUpdateInitGroup1 d_mergedCustomWUUpdateInitGroup1[2];
void pushMergedCustomWUUpdateInitGroup1ToDevice(unsigned int idx, float* M, float* V, uint32_t numSrcNeurons, uint32_t numTrgNeurons, uint32_t rowStride) {
    MergedCustomWUUpdateInitGroup1 group = {M, V, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomWUUpdateInitGroup1, &group, sizeof(MergedCustomWUUpdateInitGroup1), idx * sizeof(MergedCustomWUUpdateInitGroup1), cudaMemcpyHostToDevice, 0));
}
void initializeHost() {
}
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {832, };
__device__ unsigned int d_mergedNeuronInitGroupStartID2[] = {960, };
__device__ unsigned int d_mergedSynapseInitGroupStartID0[] = {1024, 1152, };
__device__ unsigned int d_mergedCustomUpdateInitGroupStartID0[] = {1216, };
__device__ unsigned int d_mergedCustomUpdateInitGroupStartID1[] = {1280, };
__device__ unsigned int d_mergedCustomWUUpdateInitGroupStartID0[] = {1344, 1472, };
__device__ unsigned int d_mergedCustomWUUpdateInitGroupStartID1[] = {1536, 1664, };

extern "C" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed) {
    if(threadIdx.x == 0) {
        curand_init(deviceRNGSeed, 0, 0, &d_rng);
    }
}

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 832) {
        const unsigned int lid = id - 0;
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        // only do this for existing neurons
        if(lid < 784u) {
             {
                uint32_t initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->StartSpike[(d * 784u) + lid] = initVal;
                }
            }
             {
                uint32_t initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->EndSpike[(d * 784u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingWriteOffset[(d * 784u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 4.990000000e+02f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingReadOffset[(d * 784u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 4.990000000e+02f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingWriteStartOffset[(d * 784u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 4.990000000e+02f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingReadEndOffset[(d * 784u) + lid] = initVal;
                }
            }
             {
                uint8_t initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->BackSpike[(d * 784u) + lid] = initVal;
                }
            }
            for(unsigned int d = 0; d < 32; d++) {
                group->spkSynSpike0[(d * 784u) + lid] = 0;
            }
            if(lid == 0) {
                for(unsigned int d = 0; d < 32; d++) {
                    group->spkCntSynSpike0[d] = 0;
                }
            }
            for(unsigned int d = 0; d < 32; d++) {
                group->spkEventSynSpikeEvent0[(d * 784u) + lid] = 0;
            }
            if(lid == 0) {
                for(unsigned int d = 0; d < 32; d++) {
                    group->spkCntEventSynSpikeEvent0[d] = 0;
                }
            }
        }
    }
    // merged1
    if(id >= 832 && id < 960) {
        const unsigned int lid = id - 832;
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        // only do this for existing neurons
        if(lid < 128u) {
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->V[(d * 128u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingWriteOffset[(d * 128u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 4.990000000e+02f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingReadOffset[(d * 128u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 4.990000000e+02f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingWriteStartOffset[(d * 128u) + lid] = initVal;
                }
            }
             {
                int32_t initVal;
                initVal = 4.990000000e+02f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->RingReadEndOffset[(d * 128u) + lid] = initVal;
                }
            }
             {
                uint8_t initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->BackSpike[(d * 128u) + lid] = initVal;
                }
            }
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->LambdaV[(d * 128u) + lid] = initVal;
                }
            }
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->LambdaI[(d * 128u) + lid] = initVal;
                }
            }
            for(unsigned int d = 0; d < 32; d++) {
                group->spkSynSpike0[(d * 128u) + lid] = 0;
            }
            if(lid == 0) {
                for(unsigned int d = 0; d < 32; d++) {
                    group->spkCntSynSpike0[d] = 0;
                }
            }
            for(unsigned int d = 0; d < 32; d++) {
                group->spkEventSynSpikeEvent0[(d * 128u) + lid] = 0;
            }
            if(lid == 0) {
                for(unsigned int d = 0; d < 32; d++) {
                    group->spkCntEventSynSpikeEvent0[d] = 0;
                }
            }
            for(unsigned int d = 0; d < 32; d++) {
                group->outPostInSyn0[(d * 128u) + lid] = 0.000000000e+00f;
            }
            for(unsigned int d = 0; d < 32; d++) {
                group->outPreOutSyn0[(d * 128u) + lid] = 0.000000000e+00f;
            }
        }
    }
    // merged2
    if(id >= 960 && id < 1024) {
        const unsigned int lid = id - 960;
        struct MergedNeuronInitGroup2 *group = &d_mergedNeuronInitGroup2[0]; 
        // only do this for existing neurons
        if(lid < 10u) {
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->V[(d * 10u) + lid] = initVal;
                }
            }
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->VAvg[(d * 10u) + lid] = initVal;
                }
            }
             {
                if(lid == 0) {
                    uint8_t initVal;
                    initVal = 0.000000000e+00f;
                    for(unsigned int d = 0; d < 32; d++) {
                        group->YTrue[d] = initVal;
                    }
                }
            }
             {
                if(lid == 0) {
                    uint8_t initVal;
                    initVal = 0.000000000e+00f;
                    for(unsigned int d = 0; d < 32; d++) {
                        group->YTrueBack[d] = initVal;
                    }
                }
            }
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->LambdaV[(d * 10u) + lid] = initVal;
                }
            }
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->LambdaI[(d * 10u) + lid] = initVal;
                }
            }
             {
                float initVal;
                initVal = 0.000000000e+00f;
                for(unsigned int d = 0; d < 32; d++) {
                    group->Softmax[(d * 10u) + lid] = initVal;
                }
            }
            for(unsigned int d = 0; d < 32; d++) {
                group->outPostInSyn0[(d * 10u) + lid] = 0.000000000e+00f;
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Synapse groups
    // merged0
    if(id >= 1024 && id < 1216) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedSynapseInitGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedSynapseInitGroup0 *group = &d_mergedSynapseInitGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedSynapseInitGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        if(lid < group->numTrgNeurons) {
            curandStatePhilox4_32_10_t localRNG = d_rng;
            skipahead_sequence((unsigned long long)id, &localRNG);
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    const unsigned int idSyn = (i * group->rowStride) + lid;
                    float initVal;
                    initVal = group->meang + (curand_normal(&localRNG) * group->sdg);
                    group->g[idSyn] = initVal;
                }
                 {
                    const unsigned int idSyn = (i * group->rowStride) + lid;
                    float initVal;
                    initVal = 0.000000000e+00f;
                    for(unsigned int d = 0; d < 32; d++) {
                        group->Gradient[(d * group->numSrcNeurons * group->rowStride) + idSyn] = initVal;
                    }
                }
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Custom update groups
    // merged0
    if(id >= 1216 && id < 1280) {
        const unsigned int lid = id - 1216;
        struct MergedCustomUpdateInitGroup0 *group = &d_mergedCustomUpdateInitGroup0[0]; 
        // only do this for existing variables
        if(lid < 10u) {
             {
                if(lid == 0) {
                    float initVal;
                    initVal = 0.000000000e+00f;
                    for(unsigned int d = 0; d < 32; d++) {
                        group->MaxVal[d] = initVal;
                    }
                }
            }
        }
    }
    // merged1
    if(id >= 1280 && id < 1344) {
        const unsigned int lid = id - 1280;
        struct MergedCustomUpdateInitGroup1 *group = &d_mergedCustomUpdateInitGroup1[0]; 
        // only do this for existing variables
        if(lid < 10u) {
             {
                if(lid == 0) {
                    float initVal;
                    initVal = 0.000000000e+00f;
                    for(unsigned int d = 0; d < 32; d++) {
                        group->SumExpVal[d] = initVal;
                    }
                }
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Custom WU update groups
    // merged0
    if(id >= 1344 && id < 1536) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomWUUpdateInitGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomWUUpdateInitGroup0 *group = &d_mergedCustomWUUpdateInitGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomWUUpdateInitGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        if(lid < group->numTrgNeurons) {
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    const unsigned int idSyn = (i * group->rowStride) + lid;
                    float initVal;
                    initVal = 0.000000000e+00f;
                    group->ReducedGradient[idSyn] = initVal;
                }
            }
        }
    }
    // merged1
    if(id >= 1536 && id < 1728) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomWUUpdateInitGroupStartID1[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomWUUpdateInitGroup1 *group = &d_mergedCustomWUUpdateInitGroup1[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomWUUpdateInitGroupStartID1[lo - 1];
        const unsigned int lid = id - groupStartID;
        if(lid < group->numTrgNeurons) {
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    const unsigned int idSyn = (i * group->rowStride) + lid;
                    float initVal;
                    initVal = 0.000000000e+00f;
                    group->M[idSyn] = initVal;
                }
                 {
                    const unsigned int idSyn = (i * group->rowStride) + lid;
                    float initVal;
                    initVal = 0.000000000e+00f;
                    group->V[idSyn] = initVal;
                }
            }
        }
    }
    
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
        std::random_device seedSource;
        uint32_t *deviceRNGSeedWord = reinterpret_cast<uint32_t*>(&deviceRNGSeed);
        for(int i = 0; i < 2; i++) {
            deviceRNGSeedWord[i] = seedSource();
        }
    }
    initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);
    CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
     {
        CHECK_RUNTIME_ERRORS(cudaEventRecord(initStart));
        const dim3 threads(64, 1);
        const dim3 grid(27, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(initStop));
        CHECK_RUNTIME_ERRORS(cudaEventSynchronize(initStop));
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, initStart, initStop));
        initTime += tmp / 1000.0;
    }
}

void initializeSparse() {
}
