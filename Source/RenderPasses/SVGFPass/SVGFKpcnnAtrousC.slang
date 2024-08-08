import SVGFCommon;

// I know I have an inconsistent naming scheme but I really couldn't care

// I do not trust uniform variables in slang
#define kKernelSize 5
#define kNumPixels (kKernelSize * kKernelSize)
#define kNumSlices kNumPixels

struct CnnKernel
{
    // y first, x second
    float w[kNumSlices][3][3];
};

cbuffer PerImageCB
{
    // input buffers
    texture2D gIllumination;
    // output buffer
    RWTexture2D<float4> gFiltered;
    // paramters
    uint2 gStepSize;
};

groupshared float cnnOut[kNumSlices][kKernelSize][kKernelSize];
groupshared float cnnIn[kNumSlices][kKernelSize][kKernelSize];

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
    for (int i = 0; i < kNumSlices; i++)
    {
        cnnIn[i][offset.y][offset.x] = (i < 4 ? illumAtCurPixel[i] : 0.0f);
    }

    // third step: exectue the kpcnn


    // fourth step: use the generated kernel to convolve the original patch

    // softmax numerical stability trick I stole from "Deep Learning"
    float maxRawOut = 0.0f;
    for (int i = 0; i < kNumPixels; i++)
    {
        maxRawOut = max(maxRawOut, cnnIn[i][offset.y][offset.x]);
    }

    float totalWeight = 0.0f;
    for (int i = 0; i < kNumPixels; i++)
    {
        cnnIn[i][offset.y][offset.x] = exp(cnnIn[i][offset.y][offset.x] - maxRawOut);
        totalWeight += cnnIn[i][offset.y][offset.x];
    }

    float4 convIllum = float4(0.0f);
    for (int i = 0; i < kKernelSize; i++)
    {
        for (int j = 0; j < kKernelSize; j++)
        {
            int2 samplePixel = basePixel + int2(i, j) * gStepSize;
            float weight = cnnIn[kKernelSize * j + i][offset.y][offset.x] / totalWeight;
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
