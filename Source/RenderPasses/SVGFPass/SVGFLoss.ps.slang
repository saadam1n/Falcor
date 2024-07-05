import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D filteredImage;
    Texture2D referenceImage;

    RWByteAddressBuffer pdaFilteredImage;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 filtered = filteredImage[ipos];
    float4 reference = referenceImage[ipos];

    float4 diff = filtered - reference;
    diff.a = 0.0; // we do not want to minimize alpha

    float4 loss = diff * diff;
    float4 dLoss = 2 * diff;

    accumDerivBuf4(pdaFilteredImage, ipos, dLoss, referenceImage);

    return loss;
}