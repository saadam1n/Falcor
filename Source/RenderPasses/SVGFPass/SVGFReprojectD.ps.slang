import SVGFCommon;
import SVGFReprojectC;

cbuffer PerImageCB_D
{
    Texture2D gMoments;
    Texture2D gIllum;
    Texture2D gHistoryLen;
};

void main(FullScreenPassVsOut vsOut) {
    DifferentialPair<float3> dpLuminanceParams = diffPair(dvLuminanceParams, float3(1.0, 0.0, 0.0));
    DifferentialPair<float[3]> dpReprojKernel = diffPair(dvReprojKernel, {0, 0, 0});
    DifferentialPair<float[4]> dpReprojParams = diffPair(dvReprojParams, {0, 0, 0, 0});
    DifferentialPair<float> dpAlpha = diffPair(dvAlpha, 0.0);
    DifferentialPair<float> dpMomentsAlpha = diffPair(dvMomentsAlpha, 0.0);

    REPROJ_OUT.Differential diffInput;

    const int2 ipos = int2(vsOut.posH.xy);

    diffInput.OutIllumination = gIllum[ipos];
    diffInput.OutMoments = gMoments[ipos].xy;
    diffInput.OutHistoryLength = gHistoryLen[ipos].x;

    __bwd_diff(reproject)(vsOut, dpLuminanceParams, dpReprojParams, dpReprojKernel, dpAlpha, dpMomentsAlpha, diffInput);
}