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




void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);
    float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);

    var pvd = calcAtrousDiffPairs(vsOut, int2(0, 0), dIllum);

    accumDerivBuf4(daSigmaL, ipos, float4(pvd.sigmaL), gAlbedo);
    accumDerivBuf4(daSigmaZ, ipos, float4(pvd.sigmaZ), gAlbedo);
    accumDerivBuf4(daSigmaN, ipos, float4(pvd.sigmaN), gAlbedo);

    accumDerivBuf4(daKernel, ipos, float4(pvd.kernel[0], pvd.kernel[1], pvd.kernel[2], 0.0f), gAlbedo);
    accumDerivBuf4(daVarianceKernel, ipos, float4(pvd.varianceKernel[0][0], pvd.varianceKernel[0][1], pvd.varianceKernel[1][0], pvd.varianceKernel[1][1]), gAlbedo);
    accumDerivBuf4(daLuminanceParams, ipos, float4(pvd.luminanceParams, 0.0f), gAlbedo);
    accumDerivBuf4(daWeightFunctionParams, ipos, float4(pvd.weightFunctionParams[0], pvd.weightFunctionParams[1], pvd.weightFunctionParams[2], 0.0f), gAlbedo);
}
