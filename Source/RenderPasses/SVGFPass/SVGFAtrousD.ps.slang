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

    int iteration;
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

ParamDiffValues addAtrousPdv(ParamDiffValues lhs, ParamDiffValues rhs)
{
    ParamDiffValues res;

    for(int i = 0; i < 3; i++)
        res.kernel[i] = lhs.kernel[i] + rhs.kernel[i];

    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            res.varianceKernel[i][j] = lhs.varianceKernel[i][j] + rhs.varianceKernel[i][j];


    res.luminanceParams = lhs.luminanceParams + rhs.luminanceParams;

    for(int i = 0; i < 3; i++)
        res.weightFunctionParams[i] = lhs.weightFunctionParams[i] + rhs.weightFunctionParams[i];

    res.sigmaL = lhs.sigmaL + rhs.sigmaL;
    res.sigmaZ = lhs.sigmaZ + rhs.sigmaZ;
    res.sigmaN = lhs.sigmaN + rhs.sigmaN;

    return res;
}

void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);
    if(iteration == 1)
    {
        float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);

        var pdv = calcAtrousDiffPairs(vsOut, int2(0, 0), dIllum);

        storeAtrousPdv(ipos, pdv);
    }
    else
    {
        // go through each and brute force it
        ParamDiffValues accum; 
        
        for(int i = 0; i < 3; i++)
            accum.kernel[i] = 0.0f;

        for(int i = 0; i < 2; i++)
            for(int j = 0; j < 2; j++)
                accum.varianceKernel[i][j] = 0.0f;


        accum.luminanceParams = float3(0.0f);

        for(int i = 0; i < 3; i++)
            accum.weightFunctionParams[i] = 0.0f;

        accum.sigmaL = 0.0f;
        accum.sigmaZ = 0.0f;
        accum.sigmaN = 0.0f;

        // our read index
        int nextReadIndex = 0;

        // let's first do the variance 3x3
        for (int yy = -1; yy <= 1; yy++)
        {
            break;
            for (int xx = -1; xx <= 1; xx++)
            {
                float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo, nextReadIndex++);
                var pdv = calcAtrousDiffPairs(vsOut, int2(xx, yy), dIllum);
                accum = addAtrousPdv(accum, pdv);
            }
        }

        // do center pixel now
        {
            float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo, nextReadIndex++);
            var pdv = calcAtrousDiffPairs(vsOut, int2(0, 0), dIllum);
            accum = addAtrousPdv(accum, pdv);
        }

        // do all remaining pixels now
        const int2 screenSize = getTextureDims(gAlbedo, 0);
        for (int yy = -2; yy <= 2; yy++)
        {
            for (int xx = -2; xx <= 2; xx++)
            {
                const int2 offset = int2(xx, yy) * gStepSize;
                const int2 p = ipos + int2(xx, yy) * gStepSize;
                const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);

                int nsi = nextStoreIndex++;

                if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
                {
                    float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo, nsi);
                    var pdv = calcAtrousDiffPairs(vsOut, offset, dIllum);
                    accum = addAtrousPdv(accum, pdv);
                }
            }
        }

        storeAtrousPdv(ipos, accum);
    }
}
