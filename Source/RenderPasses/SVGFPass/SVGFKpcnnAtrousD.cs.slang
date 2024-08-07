import SVGFCommon;
import SVGFKpcnnAtrousC;

[numthreads(128, 1, 1)]
void main(uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID, uint3 globalId: SV_DispatchThreadID)
{
    bwd_prop_akpcnn(threadId, groupId, globalId);
}
