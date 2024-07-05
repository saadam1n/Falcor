import SVGFCommon;

cbuffer SummingCB
{
    RWByteAddressBuffer srcBuf;
    int srcOffset;
    int srcMax;

    RWByteAddressBuffer dstBuf;
    int dstOffset;
};

float4 read(int idx)
{
    if (idx > srcMax)
    {
        return float4(0.0f);
    }

    int address = sizeof(float4) * idx + srcOffset;

    uint status;
    uint4 raw = srcBuf.Load4(address, status);

    float4 val = reinterpret<float4>(raw);

    for (int i = 0; i < 4; i++)
    {
        if (isnan(val[i]))
        {
            // clear any nans
            // val[i] = 0.0f;
        }
    }

    return val;
}

void write(int idx, float4 val)
{
    int address = sizeof(float4) * idx + dstOffset;

    uint4 raw = reinterpret<uint4>(val);
    dstBuf.Store4(uint(address), raw);
}

groupshared float4 sum[128];

[numthreads(128, 1, 1)]
void main(uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID, uint3 globalId : SV_DispatchThreadID)
{
    sum[threadId.x] = read(globalId.x);

    // parallel reduction
    int offset = 64;

    while (offset > 0)
    {
        if (threadId.x < offset)
        {
            sum[threadId.x] += sum[threadId.x + offset];
        }

        offset /= 2;
    }

    if (threadId.x == 0)
    {
        write(groupId.x, sum[0]);
    }
}
