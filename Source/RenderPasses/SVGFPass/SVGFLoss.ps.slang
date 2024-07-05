import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D filteredImage;
    Texture2D referenceImage;

    RWByteAddressBuffer pdaFilteredImage;
};

[BackwardDifferentiable]
float calculateLoss(float4 filtered, float4 reference)
{
    // MSE loss isn't too great
    // human is very senstiive to changes in brightness though
    // coeffs copied from https://learnopengl.com/Advanced-Lighting/Bloom 
    const float4 brightnessCoefficients = float4(0.2126, 0.7152, 0.0722, 0.0);

    float fBrightness = dot(filtered, brightnessCoefficients);
    float rBrightness = dot(reference, brightnessCoefficients);

    float brightnessCost = fBrightness - rBrightness;
    brightnessCost = 5.0f * brightnessCost * brightnessCost;

    float4 colorCost4 = filtered - reference;
    float colorCost = dot(colorCost4 * colorCost4, float4(1.0f.xxx, 0.0f));

    float totalCost = brightnessCost + colorCost;

    return totalCost;
}

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 filtered = filteredImage[ipos];
    float4 reference = referenceImage[ipos];

    filtered = saturate(filtered);
    reference = saturate(reference);

    filtered.a = 0.0f;
    reference.a = 0.0f;

    var dFiltered = diffPair(filtered);
    var dReference = diffPair(reference);

    __bwd_diff(calculateLoss)(dFiltered, dReference, 1.0f);

    accumDerivBuf4(pdaFilteredImage, ipos, dFiltered.d, referenceImage);

    // return loss to debug
    float loss = calculateLoss(filtered, reference);
    return loss;
}