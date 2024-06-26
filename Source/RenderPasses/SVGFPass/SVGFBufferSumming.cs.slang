import SVGFCommon;

cbuffer SummingCB
{
    int2 screenDim;

    RWByteAddressBuffer srcBuf;
    int srcOffset;

    RWByteAddressBuffer dstBuf;
    int dstOffset;

    Texture2D gAlbedo;
};

[numthreads(32, 32, 1)]
void main(uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID, uint3 globalId : SV_DispatchThreadID)
{
    shared float4 sum[32 * 32];

    // idx is also the linear thread ID
    int idx = threadId.y * 32 + threadId.x;

    const int2 ipos = globalId.xy;

    sum[idx] = readDerivBuf4(srcBuf, ipos, gAlbedo, 0, srcOffset);

    // "construct" a segment tree to find the sum of all elements in sum in O(logn) "wall clock" time
    for (int i = 0; i < 10; i++)
    {
        // sync threads
        GroupMemoryBarrierWithGroupSync();

        int shift = (1 << i);

        if (idx % (2 * shift) == 0)
        {
            int rd = idx + shift;
            sum[idx] += sum[rd];
        }
    }

    if (idx == 0)
    {
        int2 storeIpos = int2(groupId.xy);
        accumDerivBuf4(dstBuf, storeIpos, sum[0], gAlbedo, 0, dstOffset);
    }
}
