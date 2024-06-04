import SVGFAtrousC;
import SVGFCommon;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drIllumination;

    RWByteAddressBuffer daSigmaL;
    RWByteAddressBuffer daSigmaZ;
    RWByteAddressBuffer daSigmaN;

    RWByteAddressBuffer daKernel;
    RWByteAddressBuffer daVarianceKernel;
    RWByteAddressBuffer daLuminanceParams;
    RWByteAddressBuffer daWeightFunctionParams;
};

struct ParamDiffValues 
{
    float kernel[3];
    float varianceKernel[2][2];
    float3 luminanceParams;
    float weightFunctionParams[3];
    float sigmaL;
    float sigmaZ;
    float sigmaN;
};

ParamDiffValues calcAtrousDiffPairs(FullScreenPassVsOut vsOut, int2 offset, float4 dIllum)
{
    FullScreenPassVsOut jumpToLocation = vsOut;
    jumpToLocation.posH.xy += offset;

    var dpKernel = diffPair(dvKernel, {0.0, 0.0, 0.0});
    var dpVarianceKernel = diffPair(dvVarianceKernel, {{0.0, 0.0}, {0.0, 0.0}});
    var dpLuminanceParams =  diffPair(dvLuminanceParams, float3(0.0));
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams, {0.0, 0.0, 0.0});
    var dpSigmaL = diffPair(dvSigmaL, 0.0);
    var dpSigmaZ = diffPair(dvSigmaZ, 0.0);
    var dpSigmaN = diffPair(dvSigmaN, 0.0);
    
    const int2 ipos = int2(vsOut.posH.xy);

    __bwd_diff(atrous)(jumpToLocation, dpKernel, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, dIllum);

    ParamDiffValues pvd;

    for(int i = 0; i < 3; i++)
        pvd.kernel[i] = dpKernel.d[i];

    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            pvd.varianceKernel[i][j] = dpVarianceKernel.d[i][j];

    pvd.luminanceParams = dpLuminanceParams.d;

    for(int i = 0; i < 3; i++)
        pvd.weightFunctionParams[i] = dpWeightFunctionParams.d[i];

    pvd.sigmaL = dpSigmaL.d;
    pvd.sigmaZ = dpSigmaZ.d;
    pvd.sigmaN = dpSigmaN.d;

    return pvd;
}

void storeAtrousPdv(int2 ipos, ParamDiffValues pdv)
{
    accumDerivBuf4(daSigmaL, ipos, float4(pdv.sigmaL), gAlbedo);
    accumDerivBuf4(daSigmaZ, ipos, float4(pdv.sigmaZ), gAlbedo);
    accumDerivBuf4(daSigmaN, ipos, float4(pdv.sigmaN), gAlbedo);

    accumDerivBuf4(daKernel, ipos, float4(pdv.kernel[0], pdv.kernel[1], pdv.kernel[2], 0.0f), gAlbedo);
    accumDerivBuf4(daVarianceKernel, ipos, float4(pdv.varianceKernel[0][0], pdv.varianceKernel[0][1], pdv.varianceKernel[1][0], pdv.varianceKernel[1][1]), gAlbedo);
    accumDerivBuf4(daLuminanceParams, ipos, float4(pdv.luminanceParams, 0.0f), gAlbedo);
    accumDerivBuf4(daWeightFunctionParams, ipos, float4(pdv.weightFunctionParams[0], pdv.weightFunctionParams[1], pdv.weightFunctionParams[2], 0.0f), gAlbedo);
}