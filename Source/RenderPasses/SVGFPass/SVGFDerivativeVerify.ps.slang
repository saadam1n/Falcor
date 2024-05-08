import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D gFuncOutputLower;
    Texture2D gFuncOutputUpper;

    float delta;

    RWByteAddressBuffer drBackwardsDiffBuffer;
};

bool isNonNegligble(float4 col)
{
    return (abs(dot(col, float4(1.0f))) < 0.1f);
}

struct PS_OUT
{
    float4 col : SV_TARGET0;
    float4 fdcol : SV_TARGET1;
    float4 bdcol : SV_TARGET2;
};

PS_OUT main(FullScreenPassVsOut vsOut) 
{
    const int2 ipos = int2(vsOut.posH.xy);
    const int2 texSize = getTextureDims(gFuncOutputLower, 0);

    float4 fdcol = (gFuncOutputUpper[ipos] - gFuncOutputLower[ipos]) / (2.0f * delta);
    fdcol = float4(dot(fdcol, float4(1.0f, 1.0f, 1.0f, 0.0f)));

    float4 bdcol = readDerivBuf4(drBackwardsDiffBuffer, ipos, gFuncOutputLower);
    bdcol = float4(bdcol[0]);

    float4 col;

    if (ipos.y < texSize.y / 2)
    {
        if (ipos.x < texSize.x / 2)
        {
            col = abs(fdcol);
        }
        else
        {
            col = abs(bdcol);
        }
    }
    else
    {
        if (isNonNegligble(fdcol) && isNonNegligble(bdcol))
        {
            col = float4(0.0f);
        }
        else
        {
            col = abs((bdcol - fdcol) / fdcol);
        }
    }

    PS_OUT output;

    output.col = col;
    output.fdcol = fdcol;
    output.bdcol = bdcol;

    return output;

}

