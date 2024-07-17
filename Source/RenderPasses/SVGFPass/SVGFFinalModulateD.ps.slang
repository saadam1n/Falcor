import SVGFCommon;
import SVGFFinalModulateC;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drFilteredImage;
};

void main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);
    if (!isInPatch(ipos)) return;
    
    float4 dIllum = readDerivBuf4(drFilteredImage, ipos, gAlbedo);

    uint2 dim = getTextureDims(gAlbedo, 0);
    uint numPixels = dim.x * dim.y;
    dIllum /= float(numPixels); 

    __bwd_diff(final_mod)(vsOut, dIllum);
}
