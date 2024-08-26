import SVGFCommon;

// I know I have an inconsistent naming scheme but I really couldn't care

// I do not trust uniform variables in slang
#define kMapDim 5
#define kNumPixels (kMapDim * kMapDim)
#define kKernelDistance 1
#define kKernelDim 3
#define kKernelSummationTerms (kKernelDim * kKernelDim)
#define kOutputMapsPerLayer 1
#define kRingBufferSize (2 * kOutputMapsPerLayer + kKernelSummationTerms - 1) // minus one since for the last write index, we can simultaineously store/accum
#define kNumLayers 8
#define kNumOutputWeights kOutputMapsPerLayer

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

};

groupshared CnnMap rbuf[kRingBufferSize];
groupshared float4 inputlum[kMapDim][kMapDim];

void clear_accumulation_area(uint2 srcPix, int writeIdx)
{
    // first things first, we need to zero out everything in accumulation block
    for (int i = 0; i < kKernelSummationTerms; i++)
    {
        rbuf[(writeIdx + i) % kRingBufferSize].m[srcPix.y][srcPix.x] = 0.0f;
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
                    float mapVal = rbuf[(readIdx + srcLayer) % kRingBufferSize].m[srcPix.y][srcPix.x];
                    sum += mapVal * kernels[kernelIdx].fetch_weight(srcLayer, x + kKernelDistance, y + kKernelDistance);
                }

                int offsetIdx = kKernelDim * (y + kKernelDistance) + (x + kKernelDistance);

                rbuf[(writeIdx + offsetIdx) % kRingBufferSize].m[dstPixel.y][dstPixel.x] = sum;
            }
        }
    }

    // now sync for future passes
    GroupMemoryBarrierWithGroupSync();
}

void reduce_and_activate(uint2 offset, int writeIdx, int kernelIdx)
{
    // no fancy parallel reduction for now, just plain "linear" accumulation
    int dstIdx = writeIdx % kRingBufferSize;
    for (int i = 1; i < kKernelSummationTerms; i++)
    {
        rbuf[dstIdx].m[offset.y][offset.x] += rbuf[(writeIdx + i) % kRingBufferSize].m[offset.y][offset.x];
    }
    // now apply bias
    rbuf[dstIdx].m[offset.y][offset.x] += kernels[kernelIdx].bias;

    // apply ReLU
    rbuf[dstIdx].m[offset.y][offset.x] = max(rbuf[dstIdx].m[offset.y][offset.x], 0.0f);

    // resync for next layer
    GroupMemoryBarrierWithGroupSync();
}

void akpcnn(uint3 threadId, uint3 groupId, uint3 globalId)
{
    // first step: we need to figure out which kernel we are operating upon
    const uint2 interleavedIndex = groupId.xy % gStepSize;
    const uint2 jumpIndex = groupId.xy / gStepSize;

    const uint2 basePixel = interleavedIndex + kKernelDim * jumpIndex;

    const uint linearId = threadId.z;
    const uint2 offset = uint2(linearId % kMapDim, linearId / kMapDim);
    const uint2 curPixel = basePixel + offset;

    // second step: load in all important information
    float4 illumAtCurPixel = gIllumination[curPixel];
    float4 normalDepthAtCurPixel = gLinearZAndNormal[curPixel];
    float4 worldPosAtCurPixel = gWorldPosition[curPixel];

    inputlum[offset.y][offset.x] = illumAtCurPixel;

    for (int i = 0; i < kOutputMapsPerLayer; i++)
    {
        float writeVal;
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

        rbuf[i].m[offset.y][offset.x] = writeVal;
    }

    // third step: exectue the kpcnn
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

    // fourth step: use the generated kernel to convolve the original patch

    // softmax numerical stability trick I stole from "Deep Learning", MIT Press
    float maxRawOut = 0.0f;
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        maxRawOut = max(maxRawOut, rbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x]);
    }

    float totalWeight = 0.0f;
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        rbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x] = exp(rbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x] - maxRawOut);
        totalWeight += rbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x];
    }

    float4 convIllum = float4(0.0f);
    for (int i = 0; i < kNumOutputWeights; i++)
    {
        float4 tempAccumIllum = float4(0.0f);
        for (int y = 0; y < kMapDim; y++)
        {
            for (int x = 0; x < kMapDim; x++)
            {
                tempAccumIllum += postconv[i].fetch_weight(x, y) * inputlum[y][x];
            }
        }

        float weight = rbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x] / totalWeight;
        convIllum += weight * tempAccumIllum;
    }

    // final step: write the convoluted illum to memory
    gFiltered[curPixel] = convIllum;
}

void bwd_prop_akpcnn(uint3 threadId, uint3 groupId, uint3 globalId)
{
    // todo: fill in code later
}
