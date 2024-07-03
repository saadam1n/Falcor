import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D filteredImage;
    Texture2D referenceImage;

    RWByteAddressBuffer pdaFilteredImage;
};

void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 filtered = filteredImage[ipos];
    float4 reference = referenceImage[ipos];

    float4 diff = filtered - reference;
    float4 loss = diff * diff;

    // we do not want to minimize alpha
    loss.a = 0.0;

    accumDerivBuf4(pdaFilteredImage, ipos, loss, referenceImage);
}