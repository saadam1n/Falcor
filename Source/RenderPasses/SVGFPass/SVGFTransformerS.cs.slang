import SVGFTransformerC;
import SVGFCommon;

struct FS_OUT {
    float4 filter_result : SV_TARGET0;
    float4 debug_buf : SV_TARGET1;
};

[shader("compute")]
[numthreads(1, 1, 25)]
void main(uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID, uint3 globalId: SV_DispatchThreadID)
{
    //transformer(threadId.z);
    //return;
    simple_gemm_test(threadId.z, true);
}
