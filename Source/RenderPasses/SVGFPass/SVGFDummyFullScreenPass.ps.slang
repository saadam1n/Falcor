// useful when debugging in Nsight
import SVGFCommon;

cbuffer DummyBuffer
{
    Texture2D tex;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    return tex[ipos];
}
