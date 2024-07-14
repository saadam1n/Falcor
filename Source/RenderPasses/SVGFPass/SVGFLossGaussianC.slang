import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D image;
    RWByteAddressBuffer pdaIllumination;

    // first x-axis, then y-axis
    bool yaxis;
};

void d_getIllumination(int2 ipos, int nsi, float4 val)
{
    accumDerivBuf4(pdaIllumination, ipos, val, image);
}

[BackwardDerivative(d_getIllumination)]
float4 getIllumination(int2 ipos, int nsi)
{
    return image[ipos];
}

[BackwardDifferentiable]
float4 filter(int2 ipos)
{
    const float kernel[11] = {
        0.0012273699895602214, 0.008467421237028363, 0.03798436129141209, 0.11089218884877997, 0.21083796773361546, 0.2611813817992075, 0.21083796773361546, 0.11089218884877997, 0.03798436129141209, 0.008467421237028363, 0.0012273699895602214
    };

    const int2 offset = yaxis ? int2(0, 1) : int2(1, 0);

    float4 sum = float4(0.0f);
    for (int i = 0; i < 11; i++)
    {
        int2 p = ipos + (i - 5) * offset;

        float4 illum = getIllumination(p, i);

        if (!yaxis)
        {
            illum = applyLossColorTransform(illum);
        }

        sum += getIllumination(p, i) * kernel[i];
    }

    return sum;
}




