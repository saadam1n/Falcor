import SVGFCommon;
import SVGFReprojectC;

cbuffer PerImageCB_D
{
    Texture2D gIllumination;

    RWByteAddressBuffer drMoments;
    RWByteAddressBuffer drIllumination;
    RWByteAddressBuffer drHistoryLen;

    RWByteAddressBuffer daLuminanceParams;
    RWByteAddressBuffer daReprojKernel;
    RWByteAddressBuffer daReprojParams;
    RWByteAddressBuffer daAlpha;
    RWByteAddressBuffer daMomentsAlpha;
};

void main(FullScreenPassVsOut vsOut) {
    const int2 ipos = int2(vsOut.posH.xy);
    if (!isInPatch(ipos)) return;

    DifferentialPair<float3> dpLuminanceParams = diffPair(dvLuminanceParams, float3(1.0, 0.0, 0.0));
    DifferentialPair<float[3]> dpReprojKernel = diffPair(dvReprojKernel, {0, 0, 0});
    DifferentialPair<float[4]> dpReprojParams = diffPair(dvReprojParams, {0, 0, 0, 0});
    DifferentialPair<float> dpAlpha = diffPair(dvAlpha, 0.0);
    DifferentialPair<float> dpMomentsAlpha = diffPair(dvMomentsAlpha, 0.0);

    REPROJ_OUT.Differential diffInput;
    diffInput.OutIllumination = readDerivBuf4(drIllumination, ipos, gIllumination);
    diffInput.OutMoments = readDerivBuf4(drMoments, ipos, gIllumination).xy;
    diffInput.OutHistoryLength = readDerivBuf4(drHistoryLen, ipos, gIllumination).x;

    __bwd_diff(reproject )(vsOut, dpLuminanceParams, dpReprojParams, dpReprojKernel, dpAlpha, dpMomentsAlpha, diffInput);

    storeDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gIllumination);
    storeDerivBuf4(daReprojKernel, ipos, float4(dpReprojKernel.d[0], dpReprojKernel.d[1], dpReprojKernel.d[2], 0.0f), gIllumination);
    storeDerivBuf4(daReprojParams, ipos, float4(dpReprojParams.d[0], dpReprojParams.d[1], dpReprojParams.d[2], dpReprojParams.d[3]), gIllumination);
    storeDerivBuf4(daAlpha, ipos, float4(dpAlpha.d), gIllumination);
    storeDerivBuf4(daMomentsAlpha, ipos, float4(dpMomentsAlpha.d), gIllumination);
}
