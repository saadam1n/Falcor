import SVGFCommon;

cbuffer ConversionCB
{
    RWByteAddressBuffer drIllumination;
    int index;
    Texture2D dummy;
};

float4 main(FullScreenPassVsOut vsOut): SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);
    if (!isInPatch(ipos)) return float4(0.0f);

    float4 val = readDerivBuf4(drIllumination, ipos, dummy, index);

    for (int i = 0; i < 4; i++) {
        if (isnan(val[i]))
        {
            val[i] = 0.0f;
        }
    }

    return val;
}
