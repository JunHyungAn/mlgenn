#include "definitions.h"

struct MergedCustomUpdateGroup0
 {
    int32_t* RingReadEndOffset;
    int32_t* RingReadOffset;
    int32_t* RingWriteOffset;
    int32_t* RingWriteStartOffset;
    uint32_t* StartSpike;
    
}
;
struct MergedCustomUpdateGroup1
 {
    float* OutPost;
    uint32_t numNeurons;
    
}
;
struct MergedCustomUpdateGroup2
 {
    float* LambdaI;
    float* LambdaV;
    int32_t* RingReadEndOffset;
    int32_t* RingReadOffset;
    int32_t* RingWriteOffset;
    int32_t* RingWriteStartOffset;
    float* V;
    
}
;
struct MergedCustomUpdateGroup3
 {
    float* LambdaI;
    float* LambdaV;
    float* V;
    float* VAvg;
    
}
;
struct MergedCustomUpdateGroup4
 {
    uint8_t* YTrue;
    uint8_t* YTrueBack;
    
}
;
struct MergedCustomUpdateGroup5
 {
    float* MaxVal;
    float* SoftmaxVal;
    float* SumExpVal;
    float* Val;
    
}
;
struct MergedCustomUpdateGroup6
 {
    float* MaxVal;
    float* Val;
    
}
;
struct MergedCustomUpdateGroup7
 {
    float* MaxVal;
    float* SumExpVal;
    float* Val;
    
}
;
struct MergedCustomUpdateWUGroup0
 {
    float* Gradient;
    float* ReducedGradient;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedCustomUpdateWUGroup1
 {
    float* Gradient;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
struct MergedCustomUpdateWUGroup2
 {
    float* Gradient;
    float* M;
    float* V;
    float* Variable;
    float Alpha;
    float MomentScale1;
    float MomentScale2;
    uint32_t numSrcNeurons;
    uint32_t rowStride;
    
}
;
__device__ __constant__ MergedCustomUpdateGroup0 d_mergedCustomUpdateGroup0[1];
void pushMergedCustomUpdateGroup0ToDevice(unsigned int idx, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, uint32_t* StartSpike) {
    MergedCustomUpdateGroup0 group = {RingReadEndOffset, RingReadOffset, RingWriteOffset, RingWriteStartOffset, StartSpike, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup0, &group, sizeof(MergedCustomUpdateGroup0), idx * sizeof(MergedCustomUpdateGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup1 d_mergedCustomUpdateGroup1[2];
void pushMergedCustomUpdateGroup1ToDevice(unsigned int idx, float* OutPost, uint32_t numNeurons) {
    MergedCustomUpdateGroup1 group = {OutPost, numNeurons, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup1, &group, sizeof(MergedCustomUpdateGroup1), idx * sizeof(MergedCustomUpdateGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup2 d_mergedCustomUpdateGroup2[1];
void pushMergedCustomUpdateGroup2ToDevice(unsigned int idx, float* LambdaI, float* LambdaV, int32_t* RingReadEndOffset, int32_t* RingReadOffset, int32_t* RingWriteOffset, int32_t* RingWriteStartOffset, float* V) {
    MergedCustomUpdateGroup2 group = {LambdaI, LambdaV, RingReadEndOffset, RingReadOffset, RingWriteOffset, RingWriteStartOffset, V, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup2, &group, sizeof(MergedCustomUpdateGroup2), idx * sizeof(MergedCustomUpdateGroup2), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup3 d_mergedCustomUpdateGroup3[1];
void pushMergedCustomUpdateGroup3ToDevice(unsigned int idx, float* LambdaI, float* LambdaV, float* V, float* VAvg) {
    MergedCustomUpdateGroup3 group = {LambdaI, LambdaV, V, VAvg, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup3, &group, sizeof(MergedCustomUpdateGroup3), idx * sizeof(MergedCustomUpdateGroup3), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup4 d_mergedCustomUpdateGroup4[1];
void pushMergedCustomUpdateGroup4ToDevice(unsigned int idx, uint8_t* YTrue, uint8_t* YTrueBack) {
    MergedCustomUpdateGroup4 group = {YTrue, YTrueBack, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup4, &group, sizeof(MergedCustomUpdateGroup4), idx * sizeof(MergedCustomUpdateGroup4), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup5 d_mergedCustomUpdateGroup5[1];
void pushMergedCustomUpdateGroup5ToDevice(unsigned int idx, float* MaxVal, float* SoftmaxVal, float* SumExpVal, float* Val) {
    MergedCustomUpdateGroup5 group = {MaxVal, SoftmaxVal, SumExpVal, Val, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup5, &group, sizeof(MergedCustomUpdateGroup5), idx * sizeof(MergedCustomUpdateGroup5), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup6 d_mergedCustomUpdateGroup6[1];
void pushMergedCustomUpdateGroup6ToDevice(unsigned int idx, float* MaxVal, float* Val) {
    MergedCustomUpdateGroup6 group = {MaxVal, Val, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup6, &group, sizeof(MergedCustomUpdateGroup6), idx * sizeof(MergedCustomUpdateGroup6), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateGroup7 d_mergedCustomUpdateGroup7[1];
void pushMergedCustomUpdateGroup7ToDevice(unsigned int idx, float* MaxVal, float* SumExpVal, float* Val) {
    MergedCustomUpdateGroup7 group = {MaxVal, SumExpVal, Val, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup7, &group, sizeof(MergedCustomUpdateGroup7), idx * sizeof(MergedCustomUpdateGroup7), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateWUGroup0 d_mergedCustomUpdateWUGroup0[2];
void pushMergedCustomUpdateWUGroup0ToDevice(unsigned int idx, float* Gradient, float* ReducedGradient, uint32_t numSrcNeurons, uint32_t rowStride) {
    MergedCustomUpdateWUGroup0 group = {Gradient, ReducedGradient, numSrcNeurons, rowStride, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup0, &group, sizeof(MergedCustomUpdateWUGroup0), idx * sizeof(MergedCustomUpdateWUGroup0), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateWUGroup1 d_mergedCustomUpdateWUGroup1[2];
void pushMergedCustomUpdateWUGroup1ToDevice(unsigned int idx, float* Gradient, uint32_t numSrcNeurons, uint32_t rowStride) {
    MergedCustomUpdateWUGroup1 group = {Gradient, numSrcNeurons, rowStride, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup1, &group, sizeof(MergedCustomUpdateWUGroup1), idx * sizeof(MergedCustomUpdateWUGroup1), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ MergedCustomUpdateWUGroup2 d_mergedCustomUpdateWUGroup2[2];
void pushMergedCustomUpdateWUGroup2ToDevice(unsigned int idx, float* Gradient, float* M, float* V, float* Variable, float Alpha, float MomentScale1, float MomentScale2, uint32_t numSrcNeurons, uint32_t rowStride) {
    MergedCustomUpdateWUGroup2 group = {Gradient, M, V, Variable, Alpha, MomentScale1, MomentScale2, numSrcNeurons, rowStride, };
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup2, &group, sizeof(MergedCustomUpdateWUGroup2), idx * sizeof(MergedCustomUpdateWUGroup2), cudaMemcpyHostToDevice, 0));
}
void pushMergedCustomUpdateWU2AlphaToDevice(unsigned int idx, float value) {
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup2, &value, sizeof(value), (sizeof(MergedCustomUpdateWUGroup2) * (idx)) + offsetof(MergedCustomUpdateWUGroup2, Alpha), cudaMemcpyHostToDevice, 0));
}
void pushMergedCustomUpdateWU2MomentScale1ToDevice(unsigned int idx, float value) {
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup2, &value, sizeof(value), (sizeof(MergedCustomUpdateWUGroup2) * (idx)) + offsetof(MergedCustomUpdateWUGroup2, MomentScale1), cudaMemcpyHostToDevice, 0));
}
void pushMergedCustomUpdateWU2MomentScale2ToDevice(unsigned int idx, float value) {
    CHECK_RUNTIME_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup2, &value, sizeof(value), (sizeof(MergedCustomUpdateWUGroup2) * (idx)) + offsetof(MergedCustomUpdateWUGroup2, MomentScale2), cudaMemcpyHostToDevice, 0));
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID6[] = {0, };
extern "C" __global__ void customUpdateBatchSoftmax1(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged6
    if(id < 1024) {
        const unsigned int lid = id - 0;
        struct MergedCustomUpdateGroup6 *group = &d_mergedCustomUpdateGroup6[0]; 
        const unsigned int batch = lid / 32;
        // only do this for existing neurons
        if(lid < 1024) {
            const unsigned int lane = lid % 32;
            const unsigned int batchOffset = 10u * batch;
            float _lrMaxVal = -3.402823466e+38f;
            for(unsigned int idx = lane; idx < 10u; idx += 32) {
                float _lMaxVal;
                const float _lVal = group->Val[batchOffset + idx];
                _lMaxVal = _lVal;
                _lrMaxVal = fmax(_lrMaxVal, _lMaxVal);
            }
            _lrMaxVal = fmax(_lrMaxVal, __shfl_down_sync(0xFFFFFFFF, _lrMaxVal, 16));
            _lrMaxVal = fmax(_lrMaxVal, __shfl_down_sync(0xFFFFFFFF, _lrMaxVal, 8));
            _lrMaxVal = fmax(_lrMaxVal, __shfl_down_sync(0xFFFFFFFF, _lrMaxVal, 4));
            _lrMaxVal = fmax(_lrMaxVal, __shfl_down_sync(0xFFFFFFFF, _lrMaxVal, 2));
            _lrMaxVal = fmax(_lrMaxVal, __shfl_down_sync(0xFFFFFFFF, _lrMaxVal, 1));
            if(lane == 0) {
                group->MaxVal[batch] = _lrMaxVal;
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateBatchSoftmax1(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(16, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateBatchSoftmax1Start));
        customUpdateBatchSoftmax1<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateBatchSoftmax1Stop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateBatchSoftmax1Stop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateBatchSoftmax1Start, customUpdateBatchSoftmax1Stop));
        customUpdateBatchSoftmax1Time += tmp / 1000.0;
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID7[] = {0, };
extern "C" __global__ void customUpdateBatchSoftmax2(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged7
    if(id < 1024) {
        const unsigned int lid = id - 0;
        struct MergedCustomUpdateGroup7 *group = &d_mergedCustomUpdateGroup7[0]; 
        const unsigned int batch = lid / 32;
        // only do this for existing neurons
        if(lid < 1024) {
            const unsigned int lane = lid % 32;
            const unsigned int batchOffset = 10u * batch;
            float _lrSumExpVal = 0.0f;
            for(unsigned int idx = lane; idx < 10u; idx += 32) {
                float _lSumExpVal;
                const float _lVal = group->Val[batchOffset + idx];
                const float _lMaxVal = group->MaxVal[batch];
                _lSumExpVal = exp((_lVal - _lMaxVal) / 1.000000000e+00f);
                _lrSumExpVal += _lSumExpVal;
            }
            _lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, _lrSumExpVal, 16);
            _lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, _lrSumExpVal, 8);
            _lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, _lrSumExpVal, 4);
            _lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, _lrSumExpVal, 2);
            _lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, _lrSumExpVal, 1);
            if(lane == 0) {
                group->SumExpVal[batch] = _lrSumExpVal;
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateBatchSoftmax2(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(16, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateBatchSoftmax2Start));
        customUpdateBatchSoftmax2<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateBatchSoftmax2Stop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateBatchSoftmax2Stop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateBatchSoftmax2Start, customUpdateBatchSoftmax2Stop));
        customUpdateBatchSoftmax2Time += tmp / 1000.0;
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID5[] = {0, };
extern "C" __global__ void customUpdateBatchSoftmax3(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged5
    if(id < 2048) {
        const unsigned int lid = id - 0;
        struct MergedCustomUpdateGroup5 *group = &d_mergedCustomUpdateGroup5[0]; 
        const unsigned int paddedSize = 64 * ((10u + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        const unsigned int batchOffset = 10u * batch;
        // only do this for existing neurons
        if(bid < 10u) {
            const float _lVal = group->Val[batchOffset + bid];
            float _lSoftmaxVal = group->SoftmaxVal[batchOffset + bid];
            const float _lMaxVal = group->MaxVal[batch];
            const float _lSumExpVal = group->SumExpVal[batch];
            _lSoftmaxVal = exp((_lVal - _lMaxVal) / 1.000000000e+00f) / _lSumExpVal;
            group->SoftmaxVal[batchOffset + bid] = _lSoftmaxVal;
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateBatchSoftmax3(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(32, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateBatchSoftmax3Start));
        customUpdateBatchSoftmax3<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateBatchSoftmax3Stop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateBatchSoftmax3Stop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateBatchSoftmax3Start, customUpdateBatchSoftmax3Stop));
        customUpdateBatchSoftmax3Time += tmp / 1000.0;
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateWUGroupStartID0[] = {0, 100352, };
extern "C" __global__ void customUpdateGradientBatchReduce(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // ------------------------------------------------------------------------
    // Custom WU updates
    // merged0
    if(id < 101632) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateWUGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateWUGroup0 *group = &d_mergedCustomUpdateWUGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateWUGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        const uint32_t size = (uint32_t)group->numSrcNeurons * group->rowStride;
        if (lid < size) {
            float _lrReducedGradient = 0.0f;
            for(unsigned int batch = 0; batch < 32; batch++) {
                const unsigned int batchOffset = size * batch;
                float _lReducedGradient;
                float _lGradient = group->Gradient[batchOffset + lid];
                _lReducedGradient = _lGradient;
                _lGradient = 0;
                _lrReducedGradient += _lReducedGradient;
                group->Gradient[batchOffset + lid] = _lGradient;
            }
            group->ReducedGradient[lid] = _lrReducedGradient;
        }
    }
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateGradientBatchReduce(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(1588, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateGradientBatchReduceStart));
        customUpdateGradientBatchReduce<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateGradientBatchReduceStop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateGradientBatchReduceStop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateGradientBatchReduceStart, customUpdateGradientBatchReduceStop));
        customUpdateGradientBatchReduceTime += tmp / 1000.0;
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateWUGroupStartID2[] = {0, 100352, };
extern "C" __global__ void customUpdateGradientLearn(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // ------------------------------------------------------------------------
    // Custom WU updates
    // merged2
    if(id < 101632) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateWUGroupStartID2[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateWUGroup2 *group = &d_mergedCustomUpdateWUGroup2[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateWUGroupStartID2[lo - 1];
        const unsigned int lid = id - groupStartID;
        const uint32_t size = (uint32_t)group->numSrcNeurons * group->rowStride;
        if (lid < size) {
            float _lM = group->M[lid];
            float _lV = group->V[lid];
            float _lVariable = group->Variable[lid];
            const float _lGradient = group->Gradient[lid];
            _lM = (9.000000000e-01f * _lM) + ((1.0f - 9.000000000e-01f) * _lGradient);
            _lV = (9.990000000e-01f * _lV) + ((1.0f - 9.990000000e-01f) * _lGradient * _lGradient);
            _lVariable -= (group->Alpha * _lM * group->MomentScale1) / (sqrt(_lV * group->MomentScale2) + 1.000000000e-08f);
            group->Variable[lid] = _lVariable;
            group->M[lid] = _lM;
            group->V[lid] = _lV;
        }
    }
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateGradientLearn(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(1588, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateGradientLearnStart));
        customUpdateGradientLearn<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateGradientLearnStop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateGradientLearnStop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateGradientLearnStart, customUpdateGradientLearnStop));
        customUpdateGradientLearnTime += tmp / 1000.0;
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID2[] = {26624, };
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID3[] = {30720, };
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID4[] = {32768, };
extern "C" __global__ void customUpdateReset(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged0
    if(id < 26624) {
        const unsigned int lid = id - 0;
        struct MergedCustomUpdateGroup0 *group = &d_mergedCustomUpdateGroup0[0]; 
        const unsigned int paddedSize = 64 * ((784u + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        const unsigned int batchOffset = 784u * batch;
        // only do this for existing neurons
        if(bid < 784u) {
            uint32_t _lStartSpike;
            int32_t _lRingReadOffset = group->RingReadOffset[batchOffset + bid];
            int32_t _lRingWriteStartOffset = group->RingWriteStartOffset[batchOffset + bid];
            int32_t _lRingWriteOffset = group->RingWriteOffset[batchOffset + bid];
            int32_t _lRingReadEndOffset = group->RingReadEndOffset[batchOffset + bid];
            _lStartSpike = 0u;
            _lRingReadOffset = _lRingWriteOffset - 1;
            if(_lRingReadOffset < 0)
             {
                _lRingReadOffset = 499;
            }
            
            _lRingReadEndOffset = _lRingWriteStartOffset;
            _lRingWriteStartOffset = _lRingReadOffset;
            group->StartSpike[batchOffset + bid] = _lStartSpike;
            group->RingReadOffset[batchOffset + bid] = _lRingReadOffset;
            group->RingWriteStartOffset[batchOffset + bid] = _lRingWriteStartOffset;
            group->RingWriteOffset[batchOffset + bid] = _lRingWriteOffset;
            group->RingReadEndOffset[batchOffset + bid] = _lRingReadEndOffset;
        }
    }
    // merged2
    if(id >= 26624 && id < 30720) {
        const unsigned int lid = id - 26624;
        struct MergedCustomUpdateGroup2 *group = &d_mergedCustomUpdateGroup2[0]; 
        const unsigned int paddedSize = 64 * ((128u + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        const unsigned int batchOffset = 128u * batch;
        // only do this for existing neurons
        if(bid < 128u) {
            int32_t _lRingWriteStartOffset = group->RingWriteStartOffset[batchOffset + bid];
            int32_t _lRingWriteOffset = group->RingWriteOffset[batchOffset + bid];
            float _lLambdaI;
            float _lV;
            int32_t _lRingReadOffset = group->RingReadOffset[batchOffset + bid];
            float _lLambdaV;
            int32_t _lRingReadEndOffset = group->RingReadEndOffset[batchOffset + bid];
            _lV = 0.000000000e+00f;
            _lLambdaV = 0.000000000e+00f;
            _lLambdaI = 0.000000000e+00f;
            _lRingReadOffset = _lRingWriteOffset - 1;
            if(_lRingReadOffset < 0)
             {
                _lRingReadOffset = 499;
            }
            
            _lRingReadEndOffset = _lRingWriteStartOffset;
            _lRingWriteStartOffset = _lRingReadOffset;
            group->RingWriteStartOffset[batchOffset + bid] = _lRingWriteStartOffset;
            group->RingWriteOffset[batchOffset + bid] = _lRingWriteOffset;
            group->LambdaI[batchOffset + bid] = _lLambdaI;
            group->V[batchOffset + bid] = _lV;
            group->RingReadOffset[batchOffset + bid] = _lRingReadOffset;
            group->LambdaV[batchOffset + bid] = _lLambdaV;
            group->RingReadEndOffset[batchOffset + bid] = _lRingReadEndOffset;
        }
    }
    // merged3
    if(id >= 30720 && id < 32768) {
        const unsigned int lid = id - 30720;
        struct MergedCustomUpdateGroup3 *group = &d_mergedCustomUpdateGroup3[0]; 
        const unsigned int paddedSize = 64 * ((10u + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        const unsigned int batchOffset = 10u * batch;
        // only do this for existing neurons
        if(bid < 10u) {
            float _lLambdaI;
            float _lV;
            float _lVAvg;
            float _lLambdaV;
            _lV = 0.000000000e+00f;
            _lVAvg = 0.0f;
            _lLambdaV = 0.000000000e+00f;
            _lLambdaI = 0.000000000e+00f;
            group->LambdaI[batchOffset + bid] = _lLambdaI;
            group->V[batchOffset + bid] = _lV;
            group->VAvg[batchOffset + bid] = _lVAvg;
            group->LambdaV[batchOffset + bid] = _lLambdaV;
        }
    }
    // merged4
    if(id >= 32768 && id < 32832) {
        const unsigned int lid = id - 32768;
        struct MergedCustomUpdateGroup4 *group = &d_mergedCustomUpdateGroup4[0]; 
        // only do this for existing neurons
        if(lid < 32) {
            uint8_t _lYTrueBack;
            const uint8_t _lYTrue = group->YTrue[lid];
            _lYTrueBack = _lYTrue;
            group->YTrueBack[lid] = _lYTrueBack;
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
        const dim3 grid(513, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateResetStart));
        customUpdateReset<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateResetStop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateResetStop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateResetStart, customUpdateResetStop));
        customUpdateResetTime += tmp / 1000.0;
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateWUGroupStartID1[] = {0, 3211264, };
extern "C" __global__ void customUpdateZeroGradient(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // ------------------------------------------------------------------------
    // Custom WU updates
    // merged1
    if(id < 3252224) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateWUGroupStartID1[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateWUGroup1 *group = &d_mergedCustomUpdateWUGroup1[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateWUGroupStartID1[lo - 1];
        const unsigned int lid = id - groupStartID;
        const uint32_t size = (uint32_t)group->numSrcNeurons * group->rowStride;
        const unsigned int paddedSize = 64 * ((size + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        if (bid < size) {
            const unsigned int batchOffset = size * batch;
            float _lGradient;
            _lGradient = 0.000000000e+00f;
            group->Gradient[batchOffset + bid] = _lGradient;
        }
    }
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateZeroGradient(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(50816, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateZeroGradientStart));
        customUpdateZeroGradient<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateZeroGradientStop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateZeroGradientStop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateZeroGradientStart, customUpdateZeroGradientStop));
        customUpdateZeroGradientTime += tmp / 1000.0;
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID1[] = {0, 4096, };
extern "C" __global__ void customUpdateZeroOutPost(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged1
    if(id < 6144) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateGroupStartID1[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateGroup1 *group = &d_mergedCustomUpdateGroup1[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateGroupStartID1[lo - 1];
        const unsigned int lid = id - groupStartID;
        const unsigned int paddedSize = 64 * ((group->numNeurons + 64 - 1) / 64);
        const unsigned int bid = lid % paddedSize;
        const unsigned int batch = lid / paddedSize;
        const unsigned int batchOffset = group->numNeurons * batch;
        // only do this for existing neurons
        if(bid < group->numNeurons) {
            float _lOutPost;
            _lOutPost = 0.000000000e+00f;
            group->OutPost[batchOffset + bid] = _lOutPost;
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
    // ------------------------------------------------------------------------
    // Custom connectivity updates
}
void updateZeroOutPost(unsigned long long timestep) {
    const float t = timestep * 1.000000000e+00f;
     {
        const dim3 threads(64, 1);
        const dim3 grid(96, 1);
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateZeroOutPostStart));
        customUpdateZeroOutPost<<<grid, threads>>>(t);
        CHECK_RUNTIME_ERRORS(cudaPeekAtLastError());
        CHECK_RUNTIME_ERRORS(cudaEventRecord(customUpdateZeroOutPostStop));
    }
    CHECK_RUNTIME_ERRORS(cudaEventSynchronize(customUpdateZeroOutPostStop)); 
     {
        float tmp;
        CHECK_RUNTIME_ERRORS(cudaEventElapsedTime(&tmp, customUpdateZeroOutPostStart, customUpdateZeroOutPostStop));
        customUpdateZeroOutPostTime += tmp / 1000.0;
    }
}
