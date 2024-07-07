import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D filteredImage;
    Texture2D referenceImage;

    RWByteAddressBuffer pdaFilteredImage;
};

[BackwardDifferentiable]
float calculateLoss(float3 filtered, float3 reference)
{
    filtered = saturate(filtered);
    reference = saturate(reference);

    // MSE loss isn't too great
    // human is very senstiive to changes in brightness though
    // coeffs copied from https://learnopengl.com/Advanced-Lighting/Bloom 
    const float3 brightnessCoefficients = float3(0.2126, 0.7152, 0.0722);

    float fb = dot(filtered, brightnessCoefficients);
    float rb = dot(reference, brightnessCoefficients);

    float brightnessCost = fb - rb;
    brightnessCost = 5.0f * brightnessCost * brightnessCost;

    float3 colorCost4 = filtered - reference;
    float colorCost = dot(colorCost4 * colorCost4, float3(1.0f));

    float totalCost = brightnessCost + colorCost;

    return totalCost;
}

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    float3 filtered = filteredImage[ipos].rgb;
    float3 reference = referenceImage[ipos].rgb;

    var dFiltered = diffPair(filtered.rgb);
    var dReference = diffPair(reference.rgb);

    __bwd_diff(calculateLoss)(dFiltered, dReference, 1.0f);

    // set alpha loss to zero
    accumDerivBuf4(pdaFilteredImage, ipos, float4(dFiltered.d, 0.0f), referenceImage);

    // return loss to debug
    float loss = calculateLoss(filtered, reference);
    return loss;
}