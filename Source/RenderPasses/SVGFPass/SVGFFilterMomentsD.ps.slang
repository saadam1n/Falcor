import SVGFCommon;
import SVGFFilterMomentsC;

cbuffer PerImageCB_D
{
    // Technically the alpha channel also contains variance
    RWByteAddressBuffer drIllumination;

    RWByteAddressBuffer daSigmaL;
    RWByteAddressBuffer daSigmaZ;
    RWByteAddressBuffer daSigmaN;

    RWByteAddressBuffer daVarianceBoostFactor;
    RWByteAddressBuffer daLuminanceParams;
    RWByteAddressBuffer daWeightFunctionParams;;
};

// compute moments
void main(FullScreenPassVsOut vsOut)
{
    var dpVarianceBoostFactor = diffPair(dvVarianceBoostFactor, 1.0);
    var dpLuminanceParams = diffPair(dvLuminanceParams, float3(0.0));
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams, {0.0, 0.0, 0.0});
    var dpSigmaL = diffPair(dvSigmaL, 0.0);
    var dpSigmaZ = diffPair(dvSigmaZ, 0.0);
    var dpSigmaN = diffPair(dvSigmaN, 0.0);
    
    const int2 ipos = int2(vsOut.posH.xy);

     __bwd_diff(filter_moments)(vsOut, dpVarianceBoostFactor, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, readDerivBuf4(drIllumination, ipos, gIllumination));
    float bwd = dpVarianceBoostFactor.d;

    accumDerivBuf4(daSigmaL, ipos, float4(dpSigmaL.d), gIllumination);
    accumDerivBuf4(daSigmaZ, ipos, float4(dpSigmaZ.d), gIllumination);
    accumDerivBuf4(daSigmaN, ipos, float4(dpSigmaN.d), gIllumination);

    accumDerivBuf4(daVarianceBoostFactor, ipos, float4(dpVarianceBoostFactor.d), gIllumination);
    accumDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gIllumination);
    accumDerivBuf4(daWeightFunctionParams, ipos, float4(dpWeightFunctionParams.d[0], dpWeightFunctionParams.d[1], dpWeightFunctionParams.d[2], 0.0f), gIllumination);
}
