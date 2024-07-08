import SVGFCommon;
import SVGFLossGaussianC;

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    return filter(ipos);
}