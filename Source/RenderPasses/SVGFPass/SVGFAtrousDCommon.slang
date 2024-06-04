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