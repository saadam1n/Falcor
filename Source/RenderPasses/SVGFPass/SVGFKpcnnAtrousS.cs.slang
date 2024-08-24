import SVGFCommon;
import SVGFKpcnnAtrousC;

[numthreads(1, 1, 25)]
void main(uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID, uint3 globalId: SV_DispatchThreadID)
{
    akpcnn(threadId, groupId, globalId);
}
