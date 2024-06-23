import SVGFCommon;

cbuffer CompactingCB
{
    // raw output from Atrous pass
    RWByteAddressBuffer drIllumination;
    // compacted output
    RWByteAddressBuffer daIllumination;

    Texture2D gAlbedo;
};

void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 dIllum = float4(0.0f);
    for(int i = 0; i < 26 + 9; i++)
    {
        dIllum += readDerivBuf4_16(drIllumination, ipos, gAlbedo, i);
    }

    accumDerivBuf4(daIllumination, ipos, dIllum, gAlbedo);
}
