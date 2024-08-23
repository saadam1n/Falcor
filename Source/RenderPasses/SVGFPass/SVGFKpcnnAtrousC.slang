import SVGFCommon;

// I know I have an inconsistent naming scheme but I really couldn't care

// I do not trust uniform variables in slang
#define kMapDim 5
#define kNumPixels (kMapDim * kMapDim)
#define kKernelDim 3
#define kKernelSummationTerms (kKernelDim * kKernelDim)
#define kOutputMapsPerLayer 8
#define kRingBufferSize (2 * kOutputMapsPerLayer + kKernelSummationTerms)
#define kNumLayers 4

struct CnnKernel
{
    // y first, x second
    float weights[3][3];
    float bias;
};

struct CnnMap
{
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
    uint gNumLayers;
    CnnKernel kernels[512];
};

groupshared CnnMap rbuf[kRingBufferSize];

void convolve_kernel(int readIdx, int writeIdx, int kernelIdx)
{
    
}

void reduce_accumulation(int writeIdx)
{

}

void akpcnn(uint3 threadId, uint3 groupId, uint3 globalId)
{
    // first step: we need to figure out which kernel we are operating upon
    const uint2 interleavedIndex = groupId.xy % gStepSize;
    const uint2 jumpIndex = groupId.xy / gStepSize;

    const uint2 basePixel = interleavedIndex + kKernelSize * jumpIndex;

    const uint linearId = threadId.z;
    const uint2 offset = uint2(linearId % kKernelSize, linearId / kKernelSize);
    const uint2 curPixel = basePixel + offset;

    // second step: load in all important information
    float4 illumAtCurPixel = gIllumination[curPixel];
    float4 normalDepthAtCurPixel = gLinearZAndNormal[curPixel];
    float4 worldPosAtCurPixel = gWorldPosition[curPixel];
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
            convolve_kernel(currentReadIndex, currentWriteIndex, currentKernelIdx);
            reduce_accumulation(currentWriteIndex);
            currentReadIndex++;
            currentWriteIndex++;
            currentKernelIdx++;
        }
        currentReadIndex += kOutputMapsPerLayer;
    }





    // fourth step: use the generated kernel to convolve the original patch

    // softmax numerical stability trick I stole from "Deep Learning"
    float maxRawOut = 0.0f;
    for (int i = 0; i < kNumPixels; i++)
    {
        maxRawOut = max(maxRawOut, cnnS[r][i][offset.y][offset.x]);
    }

    float totalWeight = 0.0f;
    for (int i = 0; i < kNumPixels; i++)
    {
        cnnS[r][i][offset.y][offset.x] = exp(cnnS[r][i][offset.y][offset.x] - maxRawOut);
        totalWeight += cnnS[r][i][offset.y][offset.x];
    }

    float4 convIllum = float4(0.0f);
    for (int i = 0; i < kKernelSize; i++)
    {
        for (int j = 0; j < kKernelSize; j++)
        {
            int2 samplePixel = basePixel + int2(i, j) * gStepSize;
            float weight = cnnS[r][kKernelSize * j + i][offset.y][offset.x] / totalWeight;
            convIllum += gIllumination[samplePixel] * weight;
        }
    }

    // final step: write the convoluted illum to memory
    gFiltered[curPixel] = convIllum;
}

void bwd_prop_akpcnn(uint3 threadId, uint3 groupId, uint3 globalId)
{
    // todo: fill in code later
}
