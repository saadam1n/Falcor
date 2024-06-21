import SVGFCommon;
import SVGFFinalModulateC;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drFilteredImage;
};

void main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    __bwd_diff(final_mod)(vsOut,  float4(1.0f, 1.0f, 1.0f, 0.0f));
}