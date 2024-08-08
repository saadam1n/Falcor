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
    float bias;
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
    CnnKernel kernels[][kNumSlices];
};

groupshared float cnnS[2][kNumSlices][kKernelSize][kKernelSize];

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
    for (int i = 0; i < kNumSlices; i++)
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

        cnnS[0][i][offset.y][offset.x] = writeVal;
    }

    // third step: exectue the kpcnn
    int r = 0;
    int w = 1;
    for (int i = 0; i < gNumLayers; i++)
    {
        // clear output layers
        for (int j = 0; j < kNumSlices; j++)
        {
            cnnS[w][j][offset.y][offset.x] = kernels[i][j].bias;
        }

        // ensure everything is written before proceeding
        GroupMemoryBarrierWithGroupSync();

        // use shared memory ops to speed this up
        for (int j = 0; j < kNumSlices; j++) // read index
        {
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    int2 wP = offset + int2(x, y);
                    if (any(wP < int2(0, 0)) || any(wP >= int2(kKernelSize, kKernelSize)))
                    {
                        continue;
                    }

                    for (int k = 0; k < kNumSlices; k++) // write index
                    {
                        // todo: figure out how to do this atomically
                        cnnS[w][k][wP.y][wP.x] += kernels[i][j].w[j][y + 1][x + 1] * cnnS[r][j][curPixel.y][curPixel.x];
                    }
                }
            }
        }

        // sync to apply ReLU
        GroupMemoryBarrierWithGroupSync();
        for (int j = 0; j < kNumSlices; j++)
        {
            cnnS[w][j][offset.y][offset.x] = max(cnnS[w][j][offset.y][offset.x], 0.0f);
        }

        // now swap the read and write indices
        int temp = r;
        r = w;
        w = temp;
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
