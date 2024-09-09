import SVGFCommon;
import Utils.Debug.PixelDebug;

// I know I have an inconsistent naming scheme but I really couldn't care

// I do not trust uniform variables in slang
#define kMapDim 5
#define kNumPixels (kMapDim * kMapDim)
#define kKernelDistance 1
#define kKernelDim 3
#define kKernelSummationTerms (kKernelDim * kKernelDim)
#define kOutputMapsPerLayer 8
#ifdef FWD_PASS
#define kRingBufferSize (2 * kOutputMapsPerLayer + kKernelSummationTerms - 1) // minus one since for the last write index, we can simultaineously store/accum
#else
#define kRingBufferSize ((kNumLayers + 2) * kOutputMapsPerLayer + kKernelSummationTerms - 1) // plus two, one for inputs and outputs
#endif
#define kNumLayers 4
#define kNumOutputWeights kOutputMapsPerLayer
#define getRingBufferIndex(x) ((x) % kRingBufferSize)
#define arbuf(x) rbuf[getRingBufferIndex(x)]
#ifdef BWD_PASS
#define drbuf(x) dbuf[getRingBufferIndex(x)]
#endif
//#define GET_RAW_WEIGHTS
#define GET_PIXEL_PARAMS \
    const uint2 interleavedIndex = groupId.xy % gStepSize; \
    const uint2 jumpIndex = groupId.xy / gStepSize; \
    const uint2 basePixel = interleavedIndex + kKernelDim * jumpIndex; \
    const uint linearId = threadId.z; \
    const uint2 offset = uint2(linearId % kMapDim, linearId / kMapDim); \
    const uint2 curPixel = basePixel + offset;

#ifdef FWD_PASS
#define kWeightIndexOffset 0
#else
#define kWeightIndexOffset kNumOutputWeights
#endif

// #define ILLUM_ONLY_INPUT

struct CnnKernel
{
    // map, y first, x second
    float4 weights[(kOutputMapsPerLayer * kKernelDim * kKernelDim + 3) / 4];
    float bias;

    static int2 get_index(const int map, const int x, const int y)
    {
        const int linearIdx = kKernelDim * kKernelDim * map + kKernelDim * y + x;
        const int elemIdx = linearIdx / 4;
        const int chnlIdx = linearIdx % 4;
        return int2(elemIdx, chnlIdx);
    }

    float fetch_weight(const int map, const int x, const int y)
    {
        const int2 idx = get_index(map, x, y);
        return weights[idx[0]][idx[1]];
    }

    /*
    [mutating]
    void save_bwd_prop(const int map, const int x, const int y, float w)
    {
        const int2 idx = get_index(map, x, y);
        weights[idx[0]][idx[1]] = w;
    }
    */
};

// alternate name: CnnSummaryKernel
struct CnnPostconvolutionKernel
{
    float4 weights[(kMapDim * kMapDim + 3) / 4];

    float fetch_weight(const int x, const int y)
    {
        const int linearIdx = y * kMapDim + x;
        const int elemIdx = linearIdx / 4;
        const int chnlIdx = linearIdx % 4;
        return weights[elemIdx][chnlIdx];
    }

    /*
    [mutating]
    void save_bwd_prop(const int x, const int y, float w)
    {
        const int linearIdx = y * kMapDim + x;
        const int elemIdx = linearIdx / 4;
        const int chnlIdx = linearIdx % 4;
        weights[elemIdx][chnlIdx] = w;
    }
    */
};

struct CnnMap
{
    // indexing: first y, then x
    float m[kMapDim][kMapDim];
};

cbuffer PerImageCB
{
    // paramters
    uint2 gStepSize;

    CnnPostconvolutionKernel postconv[kNumOutputWeights];
    CnnKernel kernels[kOutputMapsPerLayer * kNumLayers];
    // input buffers
    texture2D gIllumination;
    texture2D gLinearZAndNormal;
    texture2D gWorldPosition;
    // output buffer
    RWTexture2D<float4> gFiltered;

    RWByteAddressBuffer drIllum;
    RWByteAddressBuffer daIllum;

    RWByteAddressBuffer daPostConv;
    RWByteAddressBuffer daKernel;
};

groupshared CnnMap rbuf[kRingBufferSize];
groupshared float4 inputlum[kMapDim][kMapDim];

#ifdef BWD_PASS
struct MultiAccum
{
    float memory[kMapDim * kMapDim * kMapDim * kMapDim * 4];

    static int get_prop_idx(int a0, int a1, int a2, int a3)
    {
        int idx = (kMapDim * kMapDim * kMapDim) * a0 +
                  (kMapDim * kMapDim) * a1 +
                  (kMapDim)*a2 +
                  a3;

        return idx;
    }

    static int get_kernel_idx(int a0, int a1, int a2, int a3)
    {
        int idx = (kMapDim * kKernelSummationTerms * kNumOutputWeights) * a0 +
                  (kKernelSummationTerms * kNumOutputWeights) * a1 +
                  (kNumOutputWeights)*a2 +
                  a3;

        return idx;
    }

    /*
    [mutating]
    void save_prop(int a0, int a1, int a2, int a3, float x)
    {
        int idx = get_prop_idx(a0, a1, a2, a3);

        memory[idx] = x;
    }
    */

    float read_prop(int a0, int a1, int a2, int a3)
    {
        int idx = get_prop_idx(a0, a1, a2, a3);

        return memory[idx];
    }

    /*
    [mutating]
    void save_prop4(int a0, int a1, int a2, int a3, float4 x)
    {
        int idx = get_prop_idx(a0, a1, a2, a3) * 4;

        for (int i = 0; i < 4; i++)
        {
            memory[idx + i] = x[i];
        }
    }
    */

    float4 read_prop4(int a0, int a1, int a2, int a3)
    {
        int idx = get_prop_idx(a0, a1, a2, a3) * 4;

        float4 x;
        for (int i = 0; i < 4; i++)
        {
            x[i] = memory[idx + i];
        }

        return x;
    }

    /*
    [mutating]
    void save_kernel(int a0, int a1, int a2, int a3, float x)
    {
        int idx = get_kernel_idx(a0, a1, a2, a3);

        memory[idx] = x;
    }
    */

    float read_kernel(int a0, int a1, int a2, int a3)
    {
        int idx = get_kernel_idx(a0, a1, a2, a3);

        return memory[idx];
    }
};

groupshared CnnMap dbuf[kRingBufferSize];
groupshared float4 dLdInputlum[kMapDim][kMapDim];
groupshared CnnKernel dLdKernel;
groupshared MultiAccum multiaccum;
groupshared float writeStaging[(kMapDim * kMapDim - 1) / 4 + 1]; // idk if staging is the write word

#define STORE_PROP(a0, a1, a2, a3, x) multiaccum.memory[multiaccum.get_prop_idx(a0, a1, a2, a3)] = x
#define STORE_PROP4(a0, a1, a2, a3, x) { int idx = multiaccum.get_prop_idx(a0, a1, a2, a3) * 4; for (int i = 0; i < 4; i++) { multiaccum.memory[idx + i] = x[i]; }}
#define STORE_KERNEL(a0, a1, a2, a3, x) multiaccum.memory[multiaccum.get_kernel_idx(a0, a1, a2, a3)] = x

static uint2 accumIndex;
#endif

void setup_network_inputs(const uint2 offset, const uint2 curPixel)
{
    float4 illumAtCurPixel = gIllumination[curPixel];
    float4 normalDepthAtCurPixel = gLinearZAndNormal[curPixel];
    float4 worldPosAtCurPixel = gWorldPosition[curPixel];

    inputlum[offset.y][offset.x] = illumAtCurPixel;

    for (int i = 0; i < kOutputMapsPerLayer; i++)
    {
        float writeVal;
#ifdef ILLUM_ONLY_INPUT
#error need to set this up

        if (i < 4)
        {
            writeVal = illumAtCurPixel[i];

        }
        else if (i < 8)
        {
            writeVal = normalDepthAtCurPixel[i - 4];
        }
        else if (i < 12)
        {
            writeVal = 0.0f;//worldPosAtCurPixel[i - 8];
        }
        else
        {
            writeVal = 0.0f;
        }

#else
        if (i < 4)
        {
            writeVal = illumAtCurPixel[i];
        }
        else
        {
            writeVal = 0.0f;
        }
#endif

        rbuf[i].m[offset.y][offset.x] = writeVal;
    }
}

void bwd_setup_network_inputs(const uint2 offset, const uint2 curPixel)
{
    float4 dLdIllum;
    for (int i = 0; i < kOutputMapsPerLayer; i++)
    {
#ifdef ILLUM_ONLY_INPUT
#error need to set this up
        if (i < 4)
        {
            writeVal = illumAtCurPixel[i];

        }
        else if (i < 8)
        {
            writeVal = normalDepthAtCurPixel[i - 4];
        }
        else if (i < 12)
        {
            writeVal = 0.0f;//worldPosAtCurPixel[i - 8];
        }
        else
        {
            writeVal = 0.0f;
        }

#else
        if (i < 4)
        {
            dLdIllum[i] = rbuf[i].m[offset.y][offset.x];
        }
        else
        {
        }
#endif
    }

    accumDerivBuf4(daIllum, curPixel, dLdIllum, gIllumination);
}

void clear_accumulation_area(uint2 srcPix, int writeIdx)
{
    // first things first, we need to zero out everything in accumulation block
    for (int i = 0; i < kKernelSummationTerms; i++)
    {
        arbuf(writeIdx + i).m[srcPix.y][srcPix.x] = 0.0f;
    }
}

void convolve_kernel(uint2 srcPix, int readIdx, int writeIdx, int kernelIdx)
{
    for (int y = -kKernelDistance; y <= kKernelDistance; y++)
    {
        for (int x = -kKernelDistance; x <= kKernelDistance; x++)
        {
            const int2 dstPixel = int2(srcPix) + int2(x, y);
            const bool inside = all(dstPixel >= int2(0)) && all(dstPixel < int2(kMapDim));

            if (inside)
            {
                float sum = 0.0f;
                // now, accumulate to our target pixel
                for (int srcLayer = 0; srcLayer < kOutputMapsPerLayer; srcLayer++)
                {
                    float mapVal = arbuf(readIdx + srcLayer).m[srcPix.y][srcPix.x];
                    sum += mapVal * kernels[kernelIdx].fetch_weight(srcLayer, kKernelDistance - x, kKernelDistance - y);
                }

                int offsetIdx = kKernelDim * (kKernelDistance - y) + (kKernelDistance - x);

                arbuf(writeIdx + offsetIdx).m[dstPixel.y][dstPixel.x] = sum;
            }
        }
    }

    // now sync for future passes
    GroupMemoryBarrierWithGroupSync();
}

#ifdef BWD_PASS
void bwd_convolve_kernel(uint2 srcPix, int readIdx, int writeIdx, int kernelIdx)
{
    for (int i = 0; i < kKernelSummationTerms; i++)
    {
        for (int j = 0; j < kOutputMapsPerLayer; j++)
        {
            STORE_KERNEL(srcPix.y, srcPix.x, i,  j, 0.0f);
        }
    }


    for (int y = -kKernelDistance; y <= kKernelDistance; y++)
    {
        for (int x = -kKernelDistance; x <= kKernelDistance; x++)
        {
            const int2 dstPixel = int2(srcPix) + int2(x, y);
            const bool inside = all(dstPixel >= int2(0)) && all(dstPixel < int2(kMapDim));

            int offsetIdx = kKernelDim * (kKernelDistance - y) + (kKernelDistance - x);

            if(srcPix.x == 4 && srcPix.y == 0 && kernelIdx == 0)
            {
                int4 con;

                con.x = dstPixel.x;
                con.y = dstPixel.y;
                con.z = int(inside);
                con.w = offsetIdx;

                print(" Stuff ", con);
            }

            if (inside)
            {

                float dLoss = drbuf(writeIdx + offsetIdx).m[dstPixel.y][dstPixel.x];

                // float sum = 0.0f;
                //  now, accumulate to our target pixel

                for (int srcLayer = 0; srcLayer < kOutputMapsPerLayer; srcLayer++)
                {


                    float mapVal = arbuf(readIdx + srcLayer).m[srcPix.y][srcPix.x];
                    //sum += mapVal * kernels[kernelIdx].fetch_weight(srcLayer, x + kKernelDistance, y + kKernelDistance);

                    float dLdMapVal = dLoss * kernels[kernelIdx].fetch_weight(srcLayer, kKernelDistance - x, kKernelDistance - y);
                    float dLdKernelWeight = dLoss * mapVal;

                    dLdKernelWeight = 1.0f;

                    STORE_KERNEL(srcPix.y, srcPix.x, offsetIdx, srcLayer, dLdKernelWeight);

                    drbuf(readIdx + srcLayer).m[srcPix.y][srcPix.x] += dLdMapVal;
                }

                //arbuf(writeIdx + offsetIdx).m[dstPixel.y][dstPixel.x] = sum;
            }
        }
    }

    // now sync for future passes
    GroupMemoryBarrierWithGroupSync();


    // accum the kernel terms
    if (srcPix.x < 3 && srcPix.y < 3)
    {
        for (int srcLayer = 0; srcLayer < kOutputMapsPerLayer; srcLayer++)
        {
            float dLsum = 0.0f;
            int offsetIdx = kKernelDim * (srcPix.y) + (srcPix.x);

            for (int y = 0; y < kMapDim; y++)
            {
                for (int x = 0; x < kMapDim; x++)
                {
                    dLsum += multiaccum.read_kernel(y, x, offsetIdx, srcLayer);

                    float3 con;
                    con.x = x;
                    con.y = y;
                    con.z = multiaccum.read_kernel(y, x, offsetIdx, srcLayer);

                    if(kernelIdx == 0)
                    print("\tRd", con);
                }
            }

            // dLdKernel[kernelIdx].save_bwd_prop(srcLayer, srcPix.y, srcPix.x, dLsum);
            //  hooray for abstraction!
            int2 kidx = dLdKernel.get_index(srcLayer, srcPix.x, srcPix.y);
            dLdKernel.weights[kidx[0]][kidx[1]] = dLsum;
        }

    }

    // get all writes to sync, now we have to write it
    GroupMemoryBarrierWithGroupSync();

    int linearIdx = accumIndex.y * kMapDim + accumIndex.x;
    const int kernelStride = ((sizeof(CnnKernel) / 4 + 3) / 4);
    while (linearIdx < (kKernelDim * kKernelDim * kOutputMapsPerLayer + 3) / 4)
    {
        float4 dump = dLdKernel.weights[linearIdx];
        storeDerivBuf4(daKernel, int2(0, 0), dump, gIllumination, kernelStride * kernelIdx + linearIdx);
        linearIdx += kMapDim * kMapDim;
    }

    if (srcPix.x == 0 && srcPix.y == 0)
    {
        storeDerivBuf4( daKernel, int2(0, 0), float4(dLdKernel.bias, -999999.0f.xxx), gIllumination, kernelStride * kernelIdx + (kKernelDim * kKernelDim * kOutputMapsPerLayer + 3) / 4);
    }

}
#endif

void reduce_and_activate(uint2 offset, int writeIdx, int kernelIdx)
{
    // no fancy parallel reduction for now, just plain "linear" accumulation
    int dstIdx = getRingBufferIndex(writeIdx);

    for (int i = 1; i < kKernelSummationTerms; i++)
    {
        rbuf[dstIdx].m[offset.y][offset.x] += arbuf(writeIdx + i).m[offset.y][offset.x];
    }

    // now apply bias
    rbuf[dstIdx].m[offset.y][offset.x] += kernels[kernelIdx].bias;

    // apply ReLU
    rbuf[dstIdx].m[offset.y][offset.x] = max(rbuf[dstIdx].m[offset.y][offset.x], 0.0f);

    // resync for next layer
    GroupMemoryBarrierWithGroupSync();
}

#ifdef BWD_PASS
void bwd_reduce_and_activate(uint2 offset, int writeIdx, int kernelIdx)
{
    // no fancy parallel reduction for now, just plain "linear" accumulation
    int dstIdx = getRingBufferIndex(writeIdx);

    // expand the loss if ReLU didn't zero out our loss
    float dLoss = (rbuf[dstIdx].m[offset.y][offset.x] > 0.0f ? dbuf[dstIdx].m[offset.y][offset.x] : 0.0f);

    for (int i = 0; i < kKernelSummationTerms; i++)
    {
        drbuf(writeIdx + i).m[offset.y][offset.x] = dLoss;
    }

    // now apply bias
    STORE_PROP(0, 0, offset.y, offset.x, dLoss);

    // ensure everything has been written
    GroupMemoryBarrierWithGroupSync();
    if (offset.x == 0 && offset.y == 0)
    {
        float dLdBias = 0.0f;
        for (int y = 0; y < kMapDim; y++)
        {
            for (int x = 0; x < kMapDim; x++)
            {
                dLdBias += multiaccum.read_prop(0, 0, offset.y, offset.x);
            }
        }
        dLdKernel.bias = dLdBias;
    }

    // resync for next layer
    GroupMemoryBarrierWithGroupSync();
}
#endif

int execute_cnn(const uint2 offset)
{
    int currentReadIndex = 0;
    int currentWriteIndex = kOutputMapsPerLayer;
    int currentKernelIdx = 0;

    for (int layerIndex = 0; layerIndex < kNumLayers; layerIndex++)
    {
        for (int outputMapIndex = 0; outputMapIndex < kOutputMapsPerLayer; outputMapIndex++)
        {
            clear_accumulation_area(offset, currentWriteIndex);
            convolve_kernel(offset, currentReadIndex, currentWriteIndex, currentKernelIdx);
            reduce_and_activate(offset, currentWriteIndex, currentKernelIdx);
            currentWriteIndex++;
            currentKernelIdx++;
        }
        currentReadIndex += kOutputMapsPerLayer;
    }

    return currentReadIndex;
}

#ifdef BWD_PASS
void bwd_execute_cnn(const uint2 offset)
{
    int currentReadIndex = kNumLayers * kOutputMapsPerLayer;
    int currentWriteIndex = kOutputMapsPerLayer;
    int currentKernelIdx = kNumLayers * kOutputMapsPerLayer;

    for (int layerIndex = kNumLayers - 1; layerIndex >= 0; layerIndex--)
    {
        currentReadIndex -= kOutputMapsPerLayer;

        for (int i = 0; i < kOutputMapsPerLayer; i++)
        {
            drbuf(currentReadIndex + i).m[offset.y][offset.x] = 0.0f;
        }

        for (int outputMapIndex = kOutputMapsPerLayer - 1; outputMapIndex >= 0; outputMapIndex--)
        {
            currentKernelIdx--;
            currentWriteIndex--;

            bwd_reduce_and_activate(offset, currentWriteIndex, currentKernelIdx);
            bwd_convolve_kernel(offset, currentReadIndex, currentWriteIndex, currentKernelIdx);
        }
    }
}
#endif

float softmax_unorm_weights(const uint2 offset, int currentReadIndex)
{
    // softmax numerical stability trick I stole from "Deep Learning", MIT Press
    float maxRawOut = 0.0f;
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        maxRawOut = max(maxRawOut, arbuf(currentReadIndex + i).m[offset.y][offset.x]);
    }

    float totalWeight;

#ifndef GET_RAW_WEIGHTS
    totalWeight = 0.0f;
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        float expw = exp(arbuf(currentReadIndex + i).m[offset.y][offset.x] - maxRawOut);
        arbuf(currentReadIndex + i + kWeightIndexOffset).m[offset.y][offset.x] = expw;
        totalWeight += expw;
    }
#else
    totalWeight = 1.0f;
#endif

    return totalWeight;
}

#ifdef BWD_PASS
void bwd_softmax_unorm_weights(const uint2 offset, int currentReadIndex, float dLdTotalWeight)
{
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        //float expw = exp(arbuf(currentReadIndex + i).m[offset.y][offset.x] - maxRawOut);
        //arbuf(currentReadIndex + i + kWeightIndexOffset).m[offset.y][offset.x] = expw;
        //totalWeight += expw;

        float dLdExpW = drbuf(currentReadIndex + i + kWeightIndexOffset).m[offset.y][offset.x] + dLdTotalWeight;
        drbuf(currentReadIndex + i).m[offset.y][offset.x] = dLdExpW;
    }
}
#endif

float4 calc_postconv(int pcIndex)
{
    float4 tempAccumIllum = float4(0.0f);
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            tempAccumIllum += postconv[pcIndex].fetch_weight(x, y) * inputlum[y][x];
        }
    }

    return tempAccumIllum;
}

#ifdef BWD_PASS
void bwd_calc_postconv(int pcIndex, float4 dLoss)
{
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            float dLdWxy = dot(dLoss * inputlum[y][x], 1.0f.xxxx);
            //multiaccum.save_prop(y, x, accumIndex.y, accumIndex.x, dLdWxy);
            STORE_PROP(y, x, accumIndex.y, accumIndex.x, dLdWxy);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    float dLdW = 0.0f;
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            float dLdWxy = multiaccum.read_prop(accumIndex.y, accumIndex.x, y, x);
            dLdW += dLdWxy;
        }
    }

    int linearIdx = accumIndex.y * kMapDim + accumIndex.x;
    writeStaging[linearIdx] = dLdW;
    GroupMemoryBarrierWithGroupSync();

    if (linearIdx % 4 == 0)
    {
        float4 dump;
        for (int i = 0; i < 4; i++)
        {
            dump[i] = writeStaging[linearIdx + i];
        }

        const int pcStride = (25 - 1) / 4 + 1;
        storeDerivBuf4(daPostConv, int2(0, 0), dump, gIllumination, pcStride * pcIndex + linearIdx / 4);
    }

    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            float4 dLdLumxy = dLoss * postconv[pcIndex].fetch_weight(x, y);
            STORE_PROP4(y, x, accumIndex.y, accumIndex.x, dLdLumxy);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    float4 dLdLum = float4(0.0f);
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            dLdLum += multiaccum.read_prop4(accumIndex.y, accumIndex.x, y, x);
        }
    }

    dLdInputlum[accumIndex.y][accumIndex.x] = dLdLum;
    GroupMemoryBarrierWithGroupSync(); // resync to prevent any issues
}
#endif

float4 filter_luminances(const uint2 offset, int weightIndex, float weightNorm)
{

    float4 convIllum = float4(0.0f);
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        float4 tempAccumIllum = calc_postconv(i);

        float weight = arbuf(weightIndex + i + kWeightIndexOffset).m[offset.y][offset.x] / weightNorm;

#ifndef GET_RAW_WEIGHTS
        convIllum += weight * tempAccumIllum;
#else
        if (i < 4)
        {
            convIllum[i] = weight;
        }
        else
        {
            break;
        }
#endif
    }

    return convIllum;
}

#ifdef BWD_PASS
void bwd_filter_luminances(const uint2 offset, int weightIndex, float weightNorm, out float dLdWeightNorm, float4 dLoss)
{
    dLdWeightNorm = 0.0f;

    float4 convIllum = float4(0.0f);
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        float4 tempAccumIllum = calc_postconv(i);

        float weight = arbuf(weightIndex + i + kWeightIndexOffset).m[offset.y][offset.x] / weightNorm;

        // convIllum += weight * tempAccumIllum;
        float dLdWeight = dot(dLoss * tempAccumIllum, 1.0f.xxxx);
        drbuf(weightIndex + i + kWeightIndexOffset).m[offset.y][offset.x] = dLdWeight / weightNorm;

        dLdWeightNorm += dLdWeight * -arbuf(weightIndex + i + kWeightIndexOffset).m[offset.y][offset.x] / (weightNorm * weightNorm);

        float4 dLdTempAccumIllum = dLoss * weight;
        bwd_calc_postconv(i, dLdTempAccumIllum);
    }
}
#endif

float4 execute_final_filtering(const uint2 offset, const int weightIndex)
{
    float totalWeight = softmax_unorm_weights(offset, weightIndex);
    float4 convIllum = filter_luminances(offset, weightIndex, totalWeight);

    return convIllum;
}

#ifdef BWD_PASS
void bwd_execute_final_filtering(const uint2 offset, const int weightIndex, float4 dLoss)
{
    float totalWeight = 0.0f;
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        totalWeight += arbuf(weightIndex + i + kWeightIndexOffset).m[offset.y][offset.x];
    }

    float dLdTotalWeight;
    bwd_filter_luminances(offset, weightIndex, totalWeight, dLdTotalWeight, dLoss);
    bwd_softmax_unorm_weights(offset, weightIndex, dLdTotalWeight);
}
#endif

void akpcnn(uint3 threadId, uint3 groupId, uint3 globalId)
{
    // first step: we need to figure out which kernel we are operating upon
    GET_PIXEL_PARAMS;

    printSetPixel(offset);

    // second step: load in all important information
    setup_network_inputs(offset, curPixel);

    // third step: exectue the kpcnn
    int weightIndex = execute_cnn(offset);

    // fourth step: use the generated kernel to convolve the original patch
    float4 convIllum = execute_final_filtering(offset, weightIndex);

    // final step: write the convoluted illum to memory
    gFiltered[curPixel] = convIllum;
}

#ifdef BWD_PASS
void bwd_prop_akpcnn(uint3 threadId, uint3 groupId, uint3 globalId)
{
    // first do foward pass, saving inputs as we go
    GET_PIXEL_PARAMS;


    printSetPixel(offset);
    setup_network_inputs(offset, curPixel);
    int weightIndex = execute_cnn(offset);
    float4 convIllum = execute_final_filtering(offset, weightIndex);


    // now, backward pass time
    accumIndex = offset;

    // init some variables and then sync
    dLdInputlum[accumIndex.y][accumIndex.x] = float4(0.0f);
    GroupMemoryBarrierWithGroupSync();

    float4 dLoss = readDerivBuf4(drIllum, curPixel, gIllumination);

    bwd_execute_final_filtering(offset, weightIndex, dLoss);
    bwd_execute_cnn(offset);
    bwd_setup_network_inputs(offset, curPixel);
}
#endif
