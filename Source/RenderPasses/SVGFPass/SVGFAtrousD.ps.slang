import SVGFAtrousC;
import SVGFCommon;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drIllumination;

    RWByteAddressBuffer daSigma;
    RWByteAddressBuffer daKernel;
    RWByteAddressBuffer daVarianceKernel;
    RWByteAddressBuffer daLuminanceParams;
    RWByteAddressBuffer daWeightFunctionParams;
};

void main(FullScreenPassVsOut vsOut)
{
    var dpKernel = diffPair(dvKernel);
    var dpVarianceKernel = diffPair(dvVarianceKernel);
    var dpLuminanceParams =  diffPair(dvLuminanceParams);
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams);
    var dpSigmaL = diffPair(dvSigmaL);
    var dpSigmaZ = diffPair(dvSigmaZ);
    var dpSigmaN = diffPair(dvSigmaN);
    
    const int2 ipos = int2(vsOut.posH.xy);

    float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);;
    
    __bwd_diff(atrous)(vsOut, dpKernel, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, dIllum);

    storeDerivBuf4(daSigma, ipos, float4(dpSigmaL.d, dpSigmaZ.d, dpSigmaN.d, 0.0f), gAlbedo);
    storeDerivBuf4(daKernel, ipos, float4(dpKernel.d[0], dpKernel.d[1], dpKernel.d[2], 0.0f), gAlbedo);
    storeDerivBuf4(daVarianceKernel, ipos, float4(dpVarianceKernel.d[0][0], dpVarianceKernel.d[0][1], dpVarianceKernel.d[1][0], dpVarianceKernel.d[1][1]), gAlbedo);
    storeDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gAlbedo);
    storeDerivBuf4(daWeightFunctionParams, ipos, float4(dpWeightFunctionParams.d[0], dpWeightFunctionParams.d[1], dpWeightFunctionParams.d[2], 0.0f), gAlbedo);
}