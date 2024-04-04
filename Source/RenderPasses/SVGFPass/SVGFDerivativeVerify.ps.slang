import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D gFuncOutputLower;
    Texture2D gFuncOutputUpper;

    float delta;

    RWByteAddressBuffer drBackwardsDiffBuffer;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);
    const int2 texSize = getTextureDims(gFuncOutputLower, 0);

    float4 col;

    if (ipos.x < texSize.x / 2)
    {
        col = (gFuncOutputUpper[ipos] - gFuncOutputLower[ipos]) / delta;
    }
    else
    {
        col = readDerivBuf4(drBackwardsDiffBuffer, ipos, gFuncOutputLower);
    }

    col = float4(0.5f);

    return col;

}

