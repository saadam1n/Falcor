import SVGFCommon;
import SVGFFinalModulateC;

cbuffer PerImageCB_D
{
    Texture2D   gResultColor;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    __bwd_diff(final_mod)(vsOut, gResultColor[ipos]);
}