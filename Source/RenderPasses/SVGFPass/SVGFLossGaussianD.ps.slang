import SVGFCommon;
import SVGFLossGaussianC;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drIllumination;
};

void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 dIllum = readDerivBuf4(drIllumination, ipos, image);
    __bwd_diff(filter)(ipos, dIllum);
}