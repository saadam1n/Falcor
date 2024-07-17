import SVGFCommon;
import SVGFFilterMomentsC;

cbuffer PerImageCB_D
{
    // Technically the alpha channel also contains variance
    RWByteAddressBuffer drIllumination;

    RWByteAddressBuffer daSigma;

    RWByteAddressBuffer daVarianceBoostFactor;
    RWByteAddressBuffer daLuminanceParams;
    RWByteAddressBuffer daWeightFunctionParams;;
};

// compute moments
void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);
    if (!isInPatch(ipos)) return;

    var dpVarianceBoostFactor = diffPair(dvVarianceBoostFactor, 0.0);
    var dpLuminanceParams = diffPair(dvLuminanceParams, float3(0.0));
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams, {0.0, 0.0, 0.0});
    var dpSigmaL = diffPair(dvSigmaL, 0.0);
    var dpSigmaZ = diffPair(dvSigmaZ, 0.0);
    var dpSigmaN = diffPair(dvSigmaN, 0.0);

    float4 dIllum = readDerivBuf4(drIllumination, ipos, gIllumination);

     __bwd_diff(filter_moments)(vsOut, dpVarianceBoostFactor, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, dIllum);
    float bwd = dpVarianceBoostFactor.d;

    storeDerivBuf4(daSigma, ipos, float4(dpSigmaL.d, dpSigmaZ.d, dpSigmaN.d, 0.0f), gIllumination);
    storeDerivBuf4(daVarianceBoostFactor, ipos, float4(dpVarianceBoostFactor.d), gIllumination);
    storeDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gIllumination);
    storeDerivBuf4(daWeightFunctionParams, ipos, float4(dpWeightFunctionParams.d[0], dpWeightFunctionParams.d[1], dpWeightFunctionParams.d[2], 0.0f), gIllumination);
}
