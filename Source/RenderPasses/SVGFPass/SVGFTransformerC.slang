import SVGFCommon;
import Utils.Debug.PixelDebug;

#define kTransformerItems 25
#define kNumFeatures 4

/*
To make this as fast as possible, we need to ensure our stride between array accesses is always 1. Let X be a 25x8 matrix. Then:

Q=X*W_Q which as 1 array stride
K=X*W_K BUT MAKE SURE THAT THE MEMORY LAYOUT OF THE FINAL MATRIX IS THE SAME SHAPE AS Q
V=X*W_V BUT TOO MAKE SURE THAT THE MEMORY LAYOUT IS LIKE Q

unorm scores = QK, since memory layout is the same, just do a dot product with array stride of 1
final mtx = norm scores * V - likewise also do dot product with array stride of 1


*/

struct WeightMatrix {
    float4 weights[(kNumFeatures * kNumFeatures - 1) / 4 + 1];

    float fetch(int dst, int src) {
        const int idx = kNumFeatures * dst + src;

        const int vidx = idx / 4;
        const int cidx = idx % 4;

        return weights[vidx][cidx];
    }

};

cbuffer PerImageCB {
    WeightMatrix weights[3];

    texture2D gIllumination;
    RWTexture2D<float4> gFiltered;
};

struct EmbeddingMatrix {
    float m[kTransformerItems][kNumFeatures];
};

groupshared EmbeddingMatrix src;
groupshared EmbeddingMatrix xw[3];
groupshared float scores[kTransformerItems][kTransformerItems];
groupshared EmbeddingMatrix output;

static int2 workIdx;
static int threadIdx;

void gemm_src(int idx) {
    // take src and do dot product
    for (int i = 0; i < kNumFeatures; i++) {
        float sum = 0.0f;

        for (int j = 0; j < kNumFeatures; j++) {
            sum += weights[idx].fetch(i, j) * src.m[threadIdx][j];
        }

        xw[idx].m[threadIdx][i] = sum;
    }
}

void gemm_scores_and_smax() {
    float maxRawScore = 0.0f;

    for (int i = 0; i < kTransformerItems; i++) {
        // dot prod with this
        float sum = 0.0f;

        for (int j = 0; j < kNumFeatures; j++) {
            sum += xw[0].m[threadIdx][j] * xw[1].m[i][j];
        }

        scores[threadIdx][i] = sum;

        if (sum > maxRawScore) {
            maxRawScore = sum;
        }
    }

    float normFactor = 0.0f;
    for (int i = 0; i < kNumFeatures; i++) {
        scores[threadIdx][i] = exp(scores[threadIdx][i] - maxRawScore);
        normFactor += scores[threadIdx][i];
    }

    for (int i = 0; i < kNumFeatures; i++) {
        scores[threadIdx][i] /= normFactor;
    }

}



void gemm_output() {
    // mult the scores matrix with other stuff

    // first clear to zero
    for (int i = 0; i < kNumFeatures; i++) {
        output.m[threadIdx][i] = 0.0f;
    }

    for (int i = 0; i < kTransformerItems; i++) {
        for (int j = 0; j < kNumFeatures; j++) {
            // blend each feature
            output.m[threadIdx][j] += scores[threadIdx][i] * xw[2].m[i][j];
        }
    }

}

void transformer(int2 idx) {
    workIdx = idx;
    threadIdx = kTransformerItems * idx.y + idx.x;

    // load the values from meory
    float4 curIllum = gIllumination[idx];

    for (int i = 0; i < 4; i++) {
        src.m[threadIdx][i] = curIllum[i];
    }

    gemm_src(0); // query
    gemm_src(1); // key

    gemm_scores_and_smax();

    gemm_src(2); // values
    gemm_output();

    float4 filtered;
    for (int i = 0; i < 4; i++) {
        filtered[i] = output.m[threadIdx][i];
    }

    gFiltered[workIdx] = filtered;
}
