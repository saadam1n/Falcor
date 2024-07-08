import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D filteredImage;
    Texture2D referenceImage;

    Texture2D filteredGaussian;
    Texture2D referenceGaussian;

    RWByteAddressBuffer pdaFilteredImage;
    RWByteAddressBuffer pdaFilteredGaussian;
};

void d_getReferenceImage(int2 ipos, float4 val) {}
[BackwardDerivative(d_getReferenceImage)]
float4 getReferenceImage(int2 ipos)
{
    return referenceImage[ipos];
}

void d_getReferenceGaussian(int2 ipos, int nsi, float4 val) {}
[BackwardDerivative(d_getReferenceGaussian)]
float4 getReferenceGaussian(int2 ipos, int nsi)
{
    return referenceGaussian[ipos];
}

void d_getFilteredImage(int2 ipos, float4 val)
{
    accumDerivBuf4(pdaFilteredImage, ipos, val, filteredImage, 11);
}

[BackwardDerivative(d_getFilteredImage)]
float4 getFilteredImage(int2 ipos)
{
    return filteredImage[ipos];
}

void d_getFilteredGaussian(int2 ipos, int nsi, float4 val)
{
    accumDerivBuf4(pdaFilteredGaussian, ipos, val, filteredGaussian, nsi);
}

[BackwardDerivative(d_getFilteredGaussian)]
float4 getFilteredGaussian(int2 ipos, int nsi)
{
    return filteredGaussian[ipos];
}

[BackwardDifferentiable]
float calculateLoss(int2 ipos)
{
    const int2 screenSize = getTextureDims(referenceImage, 0);

    // why do academics call this a "laplacian"?
    // please, just call it an edge detection filter
    const float edgeDetectionKernel[3][3] =
    {
        { -1.0f, -1.0f, -1.0f },
        { -1.0f,  8.0f, -1.0f },
        { -1.0f, -1.0f, -1.0f },
    };

    float3 filteredGradient = float3(0.0f);
    float3 referenceGradient = float3(0.0f);

    int nsi = 0;
    for (int yy = -1; yy <= 1; yy++)
    {
        for (int xx = -1; xx <= 1; xx++)
        {
            int force_nsi = nsi++;

            int2 p = ipos + int2(xx, yy);
            const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);
            if (inside)
            {
                float k = edgeDetectionKernel[yy + 1][xx + 1];
                filteredGradient += getFilteredGaussian(p, force_nsi).rgb * k;
                referenceGradient += getReferenceGaussian(p, force_nsi).rgb * k;
            }
        }
    }

    float3 gradientLoss = abs(filteredGradient - referenceGradient);

    float3 filteredCenter = applyGamma(getFilteredImage(ipos)).rgb;
    float3 referenceCenter = applyGamma(getReferenceImage(ipos)).rgb;

    float3 centerLoss = abs(filteredCenter - referenceCenter);

    // TODO later
    float3 temporalLoss = float3(0.0f);

    const float3 weights = float3(0.1f, 0.9f, 0.0f);
    float3 totalPerChannelLoss = mul(matrix<float, 3, 3>(gradientLoss, centerLoss, temporalLoss), weights);

    // Give more weight to the channels the human eye is more sensitive to
    const float3 channelCoeffs = float3(0.2126f, 0.7152f, 0.0722f);
    float totalLoss = dot(totalPerChannelLoss, channelCoeffs);

    return totalLoss;
}

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    __bwd_diff(calculateLoss)(ipos, 1.0f);

    // return loss to debug
    float loss = calculateLoss(ipos);
    return loss;
}

/*
old loss code:
    // MSE loss isn't too great
    // human is very senstiive to changes in brightness though
    const float3 brightnessCoefficients = float3(0.2126, 0.7152, 0.0722);

    float fb = dot(filtered, brightnessCoefficients);
    float rb = dot(reference, brightnessCoefficients);

    float brightnessCost = fb - rb;
    brightnessCost = 5.0f * brightnessCost * brightnessCost;

    float3 colorCost4 = filtered - reference;
    float colorCost = dot(colorCost4 * colorCost4, float3(1.0f));

    float totalCost = brightnessCost + colorCost;
*/