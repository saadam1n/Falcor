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
    return (abs(dot(col, float4(1.0f))) < 0.2f);
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

    #if 1
    float4 fdcol = (gFuncOutputUpper[ipos] - gFuncOutputLower[ipos]) / (2.0f * delta);
    fdcol = float4(dot(fdcol, float4(1.0f, 1.0f, 1.0f, 0.0f)));

    float4 bdcol = readDerivBuf4(drBackwardsDiffBuffer, ipos, gFuncOutputLower);
    for (int i = 0; i < 4; i++)
    {
        if (isnan(bdcol[i]))
        {
            bdcol[i] = 0.0f;
        }
    }
    bdcol = float4(bdcol[0]);
    #else
    float4 fdcol = float4(0.0f), bdcol = float4(0.0f);
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            int2 p = ipos + int2(i, j);
            float4 fditem = (gFuncOutputUpper[p] - gFuncOutputLower[p]) / (2.0f * delta);
            fditem = float4(dot(fditem, float4(1.0f, 1.0f, 1.0f, 0.0f)));
            fdcol += fditem;

            float4 bditem = readDerivBuf4(drBackwardsDiffBuffer, p, gFuncOutputLower);
            bditem = float4(bditem[0]);
            bdcol += bditem;
        }
    }

    fdcol /= 100.0f;
    bdcol /= 100.0f;
    #endif


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

