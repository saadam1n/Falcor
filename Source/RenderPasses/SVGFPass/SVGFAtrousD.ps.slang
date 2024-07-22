import SVGFAtrousC;
import SVGFCommon;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drIllumination;


    RWByteAddressBuffer daVarianceKernel;
    RWByteAddressBuffer daLuminanceParams;
    RWByteAddressBuffer daWeightFunctionParams;
};

void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);
    if (!isInPatch(ipos)) return;

    var dpVarianceKernel = diffPair(dvVarianceKernel);
    var dpLuminanceParams =  diffPair(dvLuminanceParams);
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams);

    float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);

    __bwd_diff(atrous)(vsOut, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dIllum);

    storeDerivBuf4(daVarianceKernel, ipos, float4(dpVarianceKernel.d[0][0], dpVarianceKernel.d[0][1], dpVarianceKernel.d[1][0], dpVarianceKernel.d[1][1]), gAlbedo);
    storeDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gAlbedo);
    storeDerivBuf4(daWeightFunctionParams, ipos, float4(dpWeightFunctionParams.d[0], dpWeightFunctionParams.d[1], dpWeightFunctionParams.d[2], 0.0f), gAlbedo);
}
