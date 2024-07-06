import SVGFCommon;
import SVGFFinalModulateC;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drFilteredImage;
};

void main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 dIllum = readDerivBuf4(drFilteredImage, ipos, gAlbedo);

    __bwd_diff(final_mod)(vsOut, dIllum);
}
