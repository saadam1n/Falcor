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

struct FS_OUT
{
    float4 diffdLdSumIllum : SV_TARGET0;
    float diffdLdSumWIllum : SV_TARGET1;
    float4 diffResult : SV_TARGET2;
    float diffWeights : SV_TARGET3;
};

FS_OUT main(FullScreenPassVsOut vsOut)
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

    float4 fwdResult = atrous(vsOut, dvKernel, dvVarianceKernel, dvLuminanceParams, dvWeightFunctionParams, dvSigmaL, dvSigmaZ, dvSigmaN);
    __bwd_diff(atrous)(vsOut, dpKernel, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, dIllum);

    storeDerivBuf4(daSigma, ipos, float4(dpSigmaL.d, dpSigmaZ.d, dpSigmaN.d, 0.0f), gAlbedo);
    storeDerivBuf4(daKernel, ipos, float4(dpKernel.d[0], dpKernel.d[1], dpKernel.d[2], 0.0f), gAlbedo);
    storeDerivBuf4(daVarianceKernel, ipos, float4(dpVarianceKernel.d[0][0], dpVarianceKernel.d[0][1], dpVarianceKernel.d[1][0], dpVarianceKernel.d[1][1]), gAlbedo);
    storeDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gAlbedo);
    storeDerivBuf4(daWeightFunctionParams, ipos, float4(dpWeightFunctionParams.d[0], dpWeightFunctionParams.d[1], dpWeightFunctionParams.d[2], 0.0f), gAlbedo);


    var dpad_sI = diffPair(autodiff_sumIllumination);
    var dpad_sW = diffPair(autodiff_sumWIllumination);

    __bwd_diff(getFilteredIllumination)(dpad_sI, dpad_sW, dIllum);

    FS_OUT fs_out;

    fs_out.diffdLdSumIllum = dpad_sI.d - manualdiff_dLdSumIllum;
    fs_out.diffdLdSumWIllum = dpad_sW.d - manualdiff_dLdSumW;
    fs_out.diffResult = tempResult - fwdResult;

    float errSum = 0.0f;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            errSum += abs(diffWeights[i][j]);

    fs_out.diffWeights = errSum;
    
    return fs_out;
}