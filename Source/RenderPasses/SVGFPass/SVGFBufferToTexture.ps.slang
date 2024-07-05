import SVGFCommon;

cbuffer ConversionCB
{
    RWByteAddressBuffer drIllumination;
    Texture2D dummy;
};

float4 main(FullScreenPassVsOut vsOut): SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    return readDerivBuf4(drIllumination, ipos, dummy);
}
