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

    RWByteAddressBuffer daInputLayerWeights0;
    RWByteAddressBuffer daInputLayerWeights1;

    RWByteAddressBuffer daHiddenLayerWeights0;
    RWByteAddressBuffer daHiddenLayerWeights1;

    RWByteAddressBuffer daOutputLayerWeights;
};

void main(FullScreenPassVsOut vsOut) {
    const int2 ipos = int2(vsOut.posH.xy);
    if (!isInPatch(ipos)) return;

    DifferentialPair<float3> dpLuminanceParams = diffPair(dvLuminanceParams, float3(1.0, 0.0, 0.0));
    DifferentialPair<float[3]> dpReprojKernel = diffPair(dvReprojKernel, {0, 0, 0});
    DifferentialPair<float[4]> dpReprojParams = diffPair(dvReprojParams, {0, 0, 0, 0});
    DifferentialPair<float> dpAlpha = diffPair(dvAlpha, 0.0);
    DifferentialPair<float> dpMomentsAlpha = diffPair(dvMomentsAlpha, 0.0);

    DifferentialPair<float [2]> dpInputLayerWeights0 = diffPair(dvInputLayerWeights0);
    DifferentialPair<float [2]> dpInputLayerWeights1 = diffPair(dvInputLayerWeights1);
    DifferentialPair<float [4]> dpHiddenLayerWeights0 = diffPair(dvHiddenLayerWeights0);
    DifferentialPair<float [4]> dpHiddenLayerWeights1 = diffPair(dvHiddenLayerWeights1);
    DifferentialPair<float[4]> dpOutputLayerWeights = diffPair(dvOutputLayerWeights);

    REPROJ_OUT.Differential diffInput;
    diffInput.OutIllumination = readDerivBuf4(drIllumination, ipos, gIllumination);
    diffInput.OutMoments = readDerivBuf4(drMoments, ipos, gIllumination).xy;
    diffInput.OutHistoryLength = readDerivBuf4(drHistoryLen, ipos, gIllumination).x;

    __bwd_diff(reproject )(vsOut, dpLuminanceParams, dpReprojParams, dpReprojKernel, dpAlpha, dpMomentsAlpha,
        dpInputLayerWeights0, dpInputLayerWeights1, dpHiddenLayerWeights0, dpHiddenLayerWeights1, dpOutputLayerWeights, diffInput);

    storeDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gIllumination);
    storeDerivBuf4(daReprojKernel, ipos, float4(dpReprojKernel.d[0], dpReprojKernel.d[1], dpReprojKernel.d[2], 0.0f), gIllumination);
    storeDerivBuf4(daReprojParams, ipos, float4(dpReprojParams.d[0], dpReprojParams.d[1], dpReprojParams.d[2], dpReprojParams.d[3]), gIllumination);
    storeDerivBuf4(daAlpha, ipos, float4(dpAlpha.d), gIllumination);
    storeDerivBuf4(daMomentsAlpha, ipos, float4(dpMomentsAlpha.d), gIllumination);

    storeDerivBuf4(daInputLayerWeights0, ipos, float4(dpInputLayerWeights0.d[0], dpInputLayerWeights0.d[1], 0.0f.xx), gAlbedo);
    storeDerivBuf4(daInputLayerWeights1, ipos, float4(dpInputLayerWeights1.d[0], dpInputLayerWeights1.d[1], 0.0f.xx), gAlbedo);
    storeDerivBuf4( daHiddenLayerWeights0, ipos, float4(dpHiddenLayerWeights0.d[0], dpHiddenLayerWeights0.d[1], dpHiddenLayerWeights0.d[2], dpHiddenLayerWeights0.d[3]), gAlbedo );
    storeDerivBuf4( daHiddenLayerWeights1, ipos, float4(dpHiddenLayerWeights1.d[0], dpHiddenLayerWeights1.d[1], dpHiddenLayerWeights1.d[2], dpHiddenLayerWeights1.d[3]), gAlbedo );
    storeDerivBuf4( daOutputLayerWeights, ipos, float4(dpOutputLayerWeights.d[0], dpOutputLayerWeights.d[1], dpOutputLayerWeights.d[2], dpOutputLayerWeights.d[3]), gAlbedo );
}
