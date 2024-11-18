struct FullScreenPassVsOut
{
    float2 texC : TEXCOORD;
#ifndef _VIEWPORT_MASK
    float4 posH : SV_POSITION;
#else
    float4 posH : POSITION;
#endif
};

struct CbufferFloatArray<let N : int, let M : int> {
    float4 array[(N * M - 1) / 4 + 1];

    int getLinearIndex(int n, int m) {
        return M * n + m;
    }

    // returns [vectorIndex, channelIndex]
    int2 getIndices(int n, int m) {
        int linIdx = getLinearIndex(n, m);
        int vectorIdx = linIdx / 4;
        int channelIdx = linIdx % 4;
        return int2(vectorIdx, channelIdx);
    }

    __subscript(int n, int m) -> float {
        get {
            int2 indices = getIndices(n, m);
            return array[indices.x][indices.y];
        }
        set {
            int2 indices = getIndices(n, m);
            array[indices.x][indices.y] = newValue;
        }
    }
}
