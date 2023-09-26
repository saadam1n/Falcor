import SVGFAtrousC;
import SVGFCommon;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drIllumination;
};

void main(FullScreenPassVsOut vsOut)
{
    var dpKernel = diffPair(dvKernel, {0.0, 0.0, 0.0});
    var dpVarianceKernel = diffPair(dvVarianceKernel, {{0.0, 0.0}, {0.0, 0.0}});
    var dpLuminanceParams =  diffPair(dvLuminanceParams, float3(0.0));
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams, {0.0, 0.0, 0.0});
    var dpSigmaL = diffPair(dvSigmaL, 0.0);
    var dpSigmaZ = diffPair(dvSigmaZ, 0.0);
    var dpSigmaN = diffPair(dvSigmaN, 0.0);
    
    const int2 ipos = int2(vsOut.posH.xy);

    __bwd_diff(atrous)(vsOut, dpKernel, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, readDerivBuf4(drIllumination, ipos, gAlbedo));    
}
