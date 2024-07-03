import SVGFCommon;

cbuffer SummingCB
{
    RWByteAddressBuffer srcBuf;
    int srcOffset;

    RWByteAddressBuffer dstBuf;
    int dstOffset;
};

float4 read(int idx)
{
    int address = sizeof(float4) * idx + srcOffset;

    uint status;
    uint4 raw = srcBuf.Load4(address, status);

    return reinterpret<float4>(raw);
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
    for (int i = 0; i < 10; i++)
    {
        // sync threads
        GroupMemoryBarrierWithGroupSync();

        int shift = (1 << i);

        if (threadId.x % (2 * shift) == 0)
        {
            float4 r = sum[threadId.x + shift];

            for (int j = 0; j < 4; j++)
            {
                // nan protection
                if (isnan(r[j]))
                {
                    r[j] = 0.0;
                }
            }

            sum[threadId.x] += r;
        }
    }

    if (threadId.x == 0)
    {
        write(groupId.x, sum[0]);
    }
}
