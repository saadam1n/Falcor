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

struct ParamDiffValues 
{
    float kernel[3];
    float varianceKernel[2][2];
    float3 luminanceParams;
    float weightFunctionParams[3];
    float sigmaL;
    float sigmaZ;
    float sigmaN;

    ImageReadData ird;
};

ParamDiffValues calcAtrousDiffPairs(FullScreenPassVsOut vsOut, ImageReadData ird, float4 dIllum)
{
    var dpKernel = diffPair(dvKernel, {0.0, 0.0, 0.0});
    var dpVarianceKernel = diffPair(dvVarianceKernel, {{0.0, 0.0}, {0.0, 0.0}});
    var dpLuminanceParams =  diffPair(dvLuminanceParams, float3(0.0));
    var dpWeightFunctionParams = diffPair(dvWeightFunctionParams, {0.0, 0.0, 0.0});
    var dpSigmaL = diffPair(dvSigmaL, 0.0);
    var dpSigmaZ = diffPair(dvSigmaZ, 0.0);
    var dpSigmaN = diffPair(dvSigmaN, 0.0);

    var dpIrdVariance = diffPair(ird.variance);
    var dpIrdColorV = diffPair(ird.color_v);

    __bwd_diff(atrous)(vsOut, dpKernel, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, dIllum);

    ParamDiffValues pdv;

    for(int i = 0; i < 3; i++)
        pdv.kernel[i] = dpKernel.d[i];

    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            pdv.varianceKernel[i][j] = dpVarianceKernel.d[i][j];

    pdv.luminanceParams = dpLuminanceParams.d;

    for(int i = 0; i < 3; i++)
        pdv.weightFunctionParams[i] = dpWeightFunctionParams.d[i];

    pdv.sigmaL = dpSigmaL.d;
    pdv.sigmaZ = dpSigmaZ.d;
    pdv.sigmaN = dpSigmaN.d;

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            pdv.ird.variance[i][j] = dpIrdVariance.d[i][j];
        }
    }

    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            pdv.ird.color_v[i][j] = dpIrdColorV.d[i][j];
        }
    }

    return pdv;
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

void storePdvCompressed(int2 ipos, int idx, ParamDiffValues pdv)
{
    // pk = packed
    float4 pkSigma = float4(pdv.sigmaL, pdv.sigmaZ, pdv.sigmaN, 0.0f);
    float4 pkKernel = float4(pdv.kernel[0], pdv.kernel[1], pdv.kernel[2], 0.0f);
    float4 pkVarianceKernel = float4(pdv.varianceKernel[0][0], pdv.varianceKernel[0][1], pdv.varianceKernel[1][0], pdv.varianceKernel[1][1]);
    float4 pkLuminanceParams = float4(pdv.luminanceParams, 0.0f);
    float4 pkWeightFunctionParams = float4(pdv.weightFunctionParams[0], pdv.weightFunctionParams[1], pdv.weightFunctionParams[2], 0.0f);

    // just store directly
    //float4 pkIrd[25 + 3];

    int k = 28 * idx;
    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            accumDerivBuf4(daIllumination, ipos, pdv.ird.color_v[i][j], gAlbedo, k++);
        }
    }

    accumDerivBuf4(daIllumination, ipos, float4(pdv.ird.variance[0][0], pdv.ird.variance[0][1], pdv.ird.variance[0][2], pdv.ird.variance[1][0]), gAlbedo, k++);
    accumDerivBuf4(daIllumination, ipos, float4(pdv.ird.variance[1][1], pdv.ird.variance[1][2], pdv.ird.variance[2][0], pdv.ird.variance[2][1]), gAlbedo, k++);
    accumDerivBuf4(daIllumination, ipos, float4(pdv.ird.variance[2][2],                   0.0f,                   0.0f,                   0.0f), gAlbedo, k++);


    accumDerivBuf4(daSigma, ipos, pkSigma, gAlbedo, idx);
    accumDerivBuf4(daKernel, ipos, pkKernel, gAlbedo, idx);
    accumDerivBuf4(daVarianceKernel, ipos, pkVarianceKernel, gAlbedo, idx);
    accumDerivBuf4(daLuminanceParams, ipos, pkLuminanceParams, gAlbedo, idx);
    accumDerivBuf4(daWeightFunctionParams, ipos, pkWeightFunctionParams, gAlbedo, idx);
}

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

    float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);
    __bwd_diff(atrous)(vsOut, dpKernel, dpVarianceKernel, dpLuminanceParams, dpWeightFunctionParams, dpSigmaL, dpSigmaZ, dpSigmaN, dIllum);

    accumDerivBuf4(daSigma, ipos, float4(dpSigmaL.d, dpSigmaZ.d, dpSigmaN.d, 0.0f), gAlbedo);
    accumDerivBuf4(daKernel, ipos, float4(dpKernel.d[0], dpKernel.d[1], dpKernel.d[2], 0.0f), gAlbedo);
    accumDerivBuf4(daVarianceKernel, ipos, float4(dpVarianceKernel.d[0][0], dpVarianceKernel.d[0][1], dpVarianceKernel.d[1][0], dpVarianceKernel.d[1][1]), gAlbedo);
    accumDerivBuf4(daLuminanceParams, ipos, float4(dpLuminanceParams.d, 0.0f), gAlbedo);
    accumDerivBuf4(daWeightFunctionParams, ipos, float4(dpWeightFunctionParams.d[0], dpWeightFunctionParams.d[1], dpWeightFunctionParams.d[2], 0.0f), gAlbedo);


}


/*
void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);

    var ird = prefetchIrd(ipos);

    // all these need to be stored separately
    ParamDiffValues pdv[4];
    pdv[0] = calcAtrousDiffPairs(vsOut, ird, float4(1.0f, 0.0f, 0.0f, 0.0f));
    pdv[1] = calcAtrousDiffPairs(vsOut, ird, float4(0.0f, 1.0f, 0.0f, 0.0f));
    pdv[2] = calcAtrousDiffPairs(vsOut, ird, float4(0.0f, 0.0f, 1.0f, 0.0f));
    pdv[3] = calcAtrousDiffPairs(vsOut, ird, float4(0.0f, 0.0f, 0.0f, 1.0f));

    // store them in their respective indices
    for(int i = 0; i < 4; i++)
    {
        storePdvCompressed(ipos, i, pdv[i]);
    }

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

// Brute force code
void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);
    if(iteration == 1)
    {
        float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);

        ParamDiffValues pdv;

        for(int i = 0; i < 3; i++)
            pdv.kernel[i] = 0.0f;

        for(int i = 0; i < 2; i++)
            for(int j = 0; j < 2; j++)
                pdv.varianceKernel[i][j] = 0.0f;


        pdv.luminanceParams = float3(0.0f);

        for(int i = 0; i < 3; i++)
            pdv.weightFunctionParams[i] = 0.0f;

        pdv.sigmaL = 0.0f;
        pdv.sigmaZ = 0.0f;
        pdv.sigmaN = 0.0f;

        // for each channel
        for(int i = 0; i < 4; i++)
        {
            // set every other channel to zero
            float4 dChannelMask;
            for(int j = 0; j < 4; j++)
            {
                if(i != j)
                {
                    dChannelMask[j] = 0.0f;
                }
                else
                {
                    dChannelMask[j] = 1.0f;
                }
            }

            var channelPdv = calcAtrousDiffPairs(vsOut, int2(0, 0), dChannelMask);

            // scale everything by this value
            float dChannelMultiplier = dIllum[i];

            // TODO: actually scale everything
            channelPdv.kernel[0] *= dChannelMultiplier;

            pdv = addAtrousPdv(pdv, channelPdv);
        }

        //pdv = calcAtrousDiffPairs(vsOut, int2(0, 0), dIllum);

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


        // start off with center pixel
        {
            float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo, nextReadIndex++);
            var pdv = calcAtrousDiffPairs(vsOut, int2(0, 0), dIllum);
            accum = addAtrousPdv(accum, pdv);
        }

        // now do variance 3x3
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

*/