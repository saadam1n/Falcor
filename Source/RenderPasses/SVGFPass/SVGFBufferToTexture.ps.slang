import SVGFCommon;

cbuffer ConversionCB
{
    RWByteAddressBuffer drIllumination;
    Texture2D dummy;
};

float4 main(FullScreenPassVsOut vsOut): SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 val = readDerivBuf4(drIllumination, ipos, dummy);

    for (int i = 0; i < 4; i++) {
        if (isnan(val[i]))
        {
            val[i] = 0.0f;
        }
    }

    return val;
}
