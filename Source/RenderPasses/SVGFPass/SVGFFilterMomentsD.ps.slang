import SVGFCommon;
import SVGFFilterMomentsC;

cbuffer PerImageCB_D
{
    // Technically the alpha channel also contains variance
    RWByteAddressBuffer drIllumination;
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

     __bwd_diff(filter_moments)(vsOut, dpVarianceBoostFactor, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaZ, readDerivBuf4(drIllumination, ipos, gIllumination));
    float bwd = dpVarianceBoostFactor.d;
}
