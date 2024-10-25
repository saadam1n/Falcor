import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D filteredImage;
    Texture2D referenceImage;

    Texture2D prevFiltered;
    Texture2D prevReference;

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

void d_getPrevReference(int2 ipos, float4 val) {}
[BackwardDerivative(d_getPrevReference)]
float4 getPrevReference(int2 ipos)
{
    return prevReference[ipos];
}

void d_getReferenceGaussian(int2 ipos, int nsi, float4 val) {}
[BackwardDerivative(d_getReferenceGaussian)]
float4 getReferenceGaussian(int2 ipos, int nsi)
{
    return referenceGaussian[ipos];
}

void d_getFilteredImage(int2 ipos, float4 val)
{
    storeDerivBuf4(pdaFilteredImage, ipos, val, filteredImage, 11);
}

[BackwardDerivative(d_getFilteredImage)]
float4 getFilteredImage(int2 ipos)
{
    return filteredImage[ipos];
}

void d_getPrevFiltered(int2 ipos, float4 val)
{
    // we'll need a buffer for this eventually for frame-to-frame stuff
}

[BackwardDerivative(d_getPrevFiltered)]
float4 getPrevFiltered(int2 ipos)
{
    return prevFiltered[ipos];
}

void d_getFilteredGaussian(int2 ipos, int nsi, float4 val)
{
    storeDerivBuf4(pdaFilteredGaussian, ipos, val, filteredGaussian, nsi);
}

[BackwardDerivative(d_getFilteredGaussian)]
float4 getFilteredGaussian(int2 ipos, int nsi)
{
    return filteredGaussian[ipos];
}

[BackwardDifferentiable]
float4 calculateLoss(int2 ipos)
{
    const int2 screenSize = getTextureDims(referenceImage, 0);

    // first compute gradient loss

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

    gradientLoss = 0.0f; // for current testing

    // color diff loss
    float3 filteredCenter = applyLossColorTransform(getFilteredImage(ipos)).rgb;
    float3 referenceCenter = applyLossColorTransform(getReferenceImage(ipos)).rgb;

    float3 centerLoss = abs(filteredCenter - referenceCenter);

    // temporal loss
    const float K_FRAME_TIME = 1.0f / 60.0f;
    float3 filteredDerivative = (filteredCenter - applyLossColorTransform(getPrevFiltered(ipos)).rgb) / K_FRAME_TIME;
    float3 referenceDerivative = (referenceCenter - applyLossColorTransform(getPrevReference(ipos)).rgb) / K_FRAME_TIME;

    float3 temporalLoss = abs(filteredDerivative - referenceDerivative);

    temporalLoss = 0.0f; // for current testing

    const float3 unnormWeights = float3(2.8f, 0.1f, 1.5f);
    const float3 normWeights = unnormWeights / dot(unnormWeights, 1.0f.xxx);
    float3 totalPerChannelLoss = normWeights[0] * centerLoss + normWeights[1] * gradientLoss + normWeights[2] * temporalLoss;  //mul(matrix<float, 3, 3>(centerLoss, gradientLoss, temporalLoss), weights);

    // Give more weight to the channels the human eye is more sensitive to
    const float3 channelCoeffs = float3(0.2126f, 0.7152f, 0.0722f);
    float totalLoss = dot(totalPerChannelLoss, channelCoeffs);

    return float4(totalLoss, dot(centerLoss, channelCoeffs), dot(gradientLoss, channelCoeffs), dot(temporalLoss, channelCoeffs));
}

[BackwardDifferentiable]
float4 just_l1_loss(int2 ipos) {
    // color diff loss
    float3 filteredCenter = (getFilteredImage(ipos)).rgb;
    float3 referenceCenter = (getReferenceImage(ipos)).rgb;

    float3 centerLoss = abs(filteredCenter - referenceCenter);

    float totalLoss = dot(centerLoss, 1.0f.xxx);

    return float4(totalLoss, centerLoss);
}

struct FS_OUT
{
    float totalLoss : SV_TARGET0;
    float centerLoss : SV_TARGET1;
    float gradientLoss : SV_TARGET2;
    float temporalLoss : SV_TARGET3;
}

FS_OUT main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);

    if (isInPatch(ipos))
    {
        __bwd_diff(calculateLoss)(ipos, float4(1.0f, 0.0f.xxx));
    }

    // return loss to debug
    float4 loss = calculateLoss(ipos);

    FS_OUT fs_out;

    fs_out.totalLoss = loss.r;
    fs_out.centerLoss = loss.g;
    fs_out.gradientLoss = loss.b;
    fs_out.temporalLoss = loss.a;

    return fs_out;
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
