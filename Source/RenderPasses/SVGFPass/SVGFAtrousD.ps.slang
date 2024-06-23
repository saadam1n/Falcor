import SVGFAtrousC;
import SVGFCommon;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drIllumination;
};

struct FS_OUT
{
    float4 sigma                    : SV_TARGET0;
    float4 kernel                   : SV_TARGET1;
    float4 varianceKernel           : SV_TARGET2;
    float4 luminanceParams          : SV_TARGET3;
    float4 weightFunctionParams     : SV_TARGET4;
};

FS_OUT main(FullScreenPassVsOut vsOut)
{
    var dpKernel = diffPair(dvKernel, {0.0, 0.0, 0.0});
    var dpVarianceKernel = diffPair(dvVarianceKernel, {{0.0, 0.0}, {0.0, 0.0}});
    var dpLuminanceParams =  diffPair(dvLuminanceParams, float3(0.0));
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams, {0.0, 0.0, 0.0});
    var dpSigmaL = diffPair(dvSigmaL, 0.0);
    var dpSigmaZ = diffPair(dvSigmaZ, 0.0);
    var dpSigmaN = diffPair(dvSigmaN, 0.0);
    
    const int2 ipos = int2(vsOut.posH.xy);

    float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);;
    
    __bwd_diff(atrous)(vsOut, dpKernel, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, dIllum);

    FS_OUT fs_out;

    fs_out.sigma = float4(dpSigmaL.d, dpSigmaZ.d, dpSigmaN.d, 0.0f);
    fs_out.kernel = float4(dpKernel.d[0], dpKernel.d[1], dpKernel.d[2], 0.0f);
    fs_out.varianceKernel = float4(dpVarianceKernel.d[0][0], dpVarianceKernel.d[0][1], dpVarianceKernel.d[1][0], dpVarianceKernel.d[1][1]);
    fs_out.luminanceParams = float4(dpLuminanceParams.d, 0.0f);
    fs_out.weightFunctionParams = float4(dpWeightFunctionParams.d[0], dpWeightFunctionParams.d[1], dpWeightFunctionParams.d[2], 0.0f);

    return fs_out;
}