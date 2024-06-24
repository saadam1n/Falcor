import SVGFCommon;

cbuffer CompactingCB
{
    // raw output from Atrous pass
    RWByteAddressBuffer drIllumination;
    // compacted output
    RWByteAddressBuffer daIllumination;

    Texture2D gAlbedo;

    int elements;
};

void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 dIllum = float4(0.0f);
    for(int i = 0; i < elements; i++)
    {
        dIllum += readDerivBuf4(drIllumination, ipos, gAlbedo, i);
    }

    accumDerivBuf4(daIllumination, ipos, dIllum, gAlbedo);
}
