import SVGFCommon;

// I know I have an inconsistent naming scheme but I really couldn't care

// I do not trust uniform variables in slang
#define kMapDim 5
#define kNumPixels (kMapDim * kMapDim)
#define kKernelDistance 1
#define kKernelDim 3
#define kKernelSummationTerms (kKernelDim * kKernelDim)
#define kOutputMapsPerLayer 8
#define kRingBufferSize (2 * kOutputMapsPerLayer + kKernelSummationTerms)
#define kNumLayers 4
#define kNumOutputWeights 8

struct CnnKernel
{
    // map, y first, x second
    float weights[kOutputMapsPerLayer][3][3];
    float bias;
};

// alternate name: CnnSummaryKernel
struct CnnPostconvolutionKernel
{
    float weights[kMapDim][kMapDim];
};

struct CnnMap
{
    // indexing: first y, then x
    float m[5][5];
};

cbuffer PerImageCB
{
    // input buffers
    texture2D gIllumination;
    texture2D gLinearZAndNormal;
    texture2D gWorldPosition;
    // output buffer
    RWTexture2D<float4> gFiltered;
    // paramters
    uint2 gStepSize;
    CnnPostconvolutionKernel postconv[8];
    CnnKernel kernels[512];
};

groupshared CnnMap rbuf[kRingBufferSize];
groupshared float4 inputlum[kMapDim][kMapDim];

void convolve_kernel(uint2 srcPix, int readIdx, int writeIdx, int kernelIdx)
{
    // first things first, we need to zero out everything in accumulation block
    for (int i = 0; i < kKernelSummationTerms; i++)
    {
        rbuf[(writeIdx + i) % kRingBufferSize].m[srcPix.y][srcPix.x] = 0.0f;
    }

    for (int y = -kKernelDistance; y <= kKernelDistance; y++)
    {
        for (int x = -kKernelDistance; x <= kKernelDistance; x++)
        {
            const uint2 dstPixel = srcPix + int2(x, y);
            const bool inside = all(dstPixel >= int2(0)) && all(dstPixel < int2(kMapDim));

            if (inside)
            {
                float sum = 0.0f;
                // now, accumulate to our target pixel
                for (int srcLayer = 0; srcLayer < kOutputMapsPerLayer; srcLayer++)
                {
                    float mapVal = rbuf[(readIdx + srcLayer) % kRingBufferSize].m[srcPix.y][srcPix.x];
                    sum += mapVal * kernels[kernelIdx].weights[srcLayer][y + kKernelDistance][x + kKernelDistance]; 
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
        rbuf[dstIdx].m[offset.y][offset.x] += rbuf[(writeIdx = i) % kRingBufferSize].m[offset.y][offset.x];
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
    const uint2 offset = uint2(linearId % kKernelDim, linearId / kKernelDim);
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
            writeVal = worldPosAtCurPixel[i - 8];
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
            convolve_kernel(offset, currentReadIndex, currentWriteIndex, currentKernelIdx);
            reduce_and_activate(offset, currentWriteIndex, currentKernelIdx);
            currentReadIndex++;
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
                tempAccumIllum += postconv[i].weights[y][x] * inputlum[y][x];
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
