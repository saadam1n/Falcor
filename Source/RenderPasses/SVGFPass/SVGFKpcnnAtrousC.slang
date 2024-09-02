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
#define GET_RAW_WEIGHTS
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

struct CnnKernel
{
    // map, y first, x second
    float4 weights[(kOutputMapsPerLayer * kKernelDim * kKernelDim + 3) / 4];
    float bias;

    float fetch_weight(const int map, const int x, const int y)
    {
        const int linearIdx = kKernelDim * kKernelDim * map + kKernelDim * y + x;
        const int elemIdx = linearIdx / 4;
        const int chnlIdx = linearIdx % 4;
        return weights[elemIdx][chnlIdx];
    }
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

    [mutating]
    float save_bwd_prop(const int x, const int y, float w)
    {
        const int linearIdx = y * kMapDim + x;
        const int elemIdx = linearIdx / 4;
        const int chnlIdx = linearIdx % 4;
        weights[elemIdx][chnlIdx] = w;
    }
};

struct CnnMap
{
    // indexing: first y, then x
    float m[kMapDim][kMapDim];
};

cbuffer PerImageCB
{
    CnnPostconvolutionKernel postconv[kNumOutputWeights];
    CnnKernel kernels[kOutputMapsPerLayer * kNumLayers];
    // input buffers
    texture2D gIllumination;
    texture2D gLinearZAndNormal;
    texture2D gWorldPosition;
    // output buffer
    RWTexture2D<float4> gFiltered;
    // paramters
    uint2 gStepSize;

    RWByteAddressBuffer drIllum;
};

groupshared CnnMap rbuf[kRingBufferSize];
groupshared float4 inputlum[kMapDim][kMapDim];

#ifdef BWD_PASS
groupshared CnnMap dbuf[kRingBufferSize];
groupshared float4 dLdInputlum[kMapDim][kMapDim];
groupshared CnnPostconvolutionKernel dLdPostconv[kNumOutputWeights];
groupshared float4 propaccum[kMapDim][kMapDim][kMapDim][kMapDim];

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
#if 0
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
                    sum += mapVal * kernels[kernelIdx].fetch_weight(srcLayer, x + kKernelDistance, y + kKernelDistance);
                }

                int offsetIdx = kKernelDim * (y + kKernelDistance) + (x + kKernelDistance);

                arbuf(writeIdx + offsetIdx).m[dstPixel.y][dstPixel.x] = sum;
            }
        }
    }

    // now sync for future passes
    GroupMemoryBarrierWithGroupSync();
}

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

void bwd_calc_postconv(int pcIndex, float4 dLoss)
{
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            float dLdWxy = dot(dLoss * inputlum[y][x], 1.0f.xxxx);
            propaccum[y][x][accumIndex.y][accumIndex.x].x = dLdWxy;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    float dLdW = 0.0f;
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            dLdW += propaccum[accumIndex.y][accumIndex.x][x][y].x;
        }
    }
    dLdPostconv[pcIndex].save_bwd_prop(accumIndex.y, accumIndex.x, dLdW);

    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            float4 dLdLumxy = dLoss * postconv[pcIndex].fetch_weight(x, y);
            propaccum[y][x][accumIndex.y][accumIndex.x] = dLdLumxy;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    float4 dLdLum = float4(0.0f);
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            dLdLum += propaccum[accumIndex.y][accumIndex.x][x][y];
        }
    }

    dLdInputlum[accumIndex.y][accumIndex.x] += dLdLum;
    GroupMemoryBarrierWithGroupSync(); // resync to prevent any issues
}

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

float4 execute_final_filtering(const uint2 offset, const int weightIndex)
{
    float totalWeight = softmax_unorm_weights(offset, weightIndex);
    float4 convIllum = filter_luminances(offset, weightIndex, totalWeight);

    return convIllum;
}

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


}
