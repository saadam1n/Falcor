import Common;

cbuffer PerImageCB
{
    Texture2D src;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    return src[ipos];
}
