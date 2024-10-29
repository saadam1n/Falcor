import SVGFCommon;
import Utils.Debug.PixelDebug;

#define kTransformerItems 25
#define kNumFeatures 8

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
    texture2D gNormalZ;
    RWTexture2D<float4> gFiltered;
    RWTexture2D<float4> gDebugBuf;

    RWByteAddressBuffer daWeightMatrices;
    RWByteAddressBuffer drIllum;
    RWByteAddressBuffer daIllum;
};

struct EmbeddingMatrix {
    float m[kNumFeatures][kTransformerItems];
};

groupshared EmbeddingMatrix src;
groupshared EmbeddingMatrix xw[3];
groupshared float scores[kTransformerItems][kTransformerItems]; // how much the 2nd idx is related to 1st idx, technically index doesn't matter
groupshared EmbeddingMatrix output;

groupshared EmbeddingMatrix dLdOutput;
groupshared float dLdScores[kTransformerItems][kTransformerItems];
groupshared EmbeddingMatrix dLdxw[3];
groupshared float dLdTempWeightMatrix[kNumFeatures][kNumFeatures][kTransformerItems];
groupshared float dLdWeightMatrices[3][kNumFeatures][kNumFeatures];
groupshared float dLdSource[kNumFeatures][kTransformerItems];

static int2 workIdx;
static int threadIdx;


void gemm_src(int idx) {
    // take src and do dot product
    for (int i = 0; i < kNumFeatures; i++) {
        float sum = 0.0f;

        for (int j = 0; j < kNumFeatures; j++) {
            sum += weights[idx].fetch(i, j) * src.m[j][threadIdx];
        }

        xw[idx].m[i][threadIdx] = sum;
    }
}

void bwd_gemm_src(int idx) {
    for (int i = 0; i < kNumFeatures; i++) { // i is dst
        for (int j = 0; j < kNumFeatures; j++) { // j is src
            // destination loss times other factor
            dLdTempWeightMatrix[i][j][threadIdx] = dLdxw[idx].m[i][threadIdx] * src.m[j][threadIdx];
            dLdSource[j][threadIdx] += dLdxw[idx].m[i][threadIdx] * weights[idx].fetch(i, j);
        }
    }

    // after this, now need to reduce it. I am very lazy
    if (threadIdx < kNumFeatures * kNumFeatures) {
        float sumWeight = 0.0f;

        int dstFeature = threadIdx / 4;
        int srcFeature = threadIdx % 4;

        for (int i = 0; i < kTransformerItems; i++) {
            sumWeight += dLdTempWeightMatrix[dstFeature][srcFeature][i];
        }

        dLdWeightMatrices[idx][dstFeature][srcFeature] = sumWeight;
    }
}

void gemm_scores_and_smax() {
    float maxRawScore = 0.0f;

    for (int i = 0; i < kTransformerItems; i++) {
        // dot prod with this
        float sum = 0.0f;

        for (int j = 0; j < kNumFeatures; j++) {
            // our query times their key
            sum += xw[0].m[j][threadIdx] * xw[1].m[j][i];
        }

        scores[i][threadIdx] = sum;

        if (sum > maxRawScore) {
            maxRawScore = sum;
        }
    }

    float normFactor = 0.0f;
    for (int i = 0; i < kTransformerItems; i++) {
        scores[i][threadIdx] = exp(scores[i][threadIdx] - maxRawScore);
        normFactor += scores[i][threadIdx];
    }

    for (int i = 0; i < kTransformerItems; i++) {
        scores[i][threadIdx] /= normFactor;
    }

}

groupshared float dLdTempScores[kTransformerItems][kTransformerItems];
void bwd_gemm_scores_and_smax() {
    // for each score we need to add normFactor
    for (int i = 0; i < kTransformerItems; i++) {
        // uh, norm factor cancels out?
        float self_score = scores[i][threadIdx];
        float temp_loss = dLdScores[i][threadIdx] * self_score * (1.0f - self_score);

        for (int j = 0; j < kTransformerItems; j++) {
            if (i != j) {
                temp_loss += dLdScores[j][threadIdx] * self_score * -scores[j][threadIdx];
            }
        }

        dLdTempScores[i][threadIdx] = temp_loss;
    }

    for (int i = 0; i < kTransformerItems; i++) {
        dLdScores[i][threadIdx] = dLdTempScores[i][threadIdx];
    }

    GroupMemoryBarrierWithGroupSync();

    for (int i = 0; i < kNumFeatures; i++) {
        dLdxw[0].m[i][threadIdx] = 0.0;
        dLdxw[1].m[i][threadIdx] = 0.0;
    }

    for (int i = 0; i < kTransformerItems; i++) {
        // s_ij = sum of q_ij * k_ij
        // we are thread j in this case usually (according to memory layout of scores)
        // so when calculating s_ij loss for query, we just multiply it by the other thread keys
        // but what about key loss?
        // we have to flip the script and say we are thread i
        // how relevant was our key to the loss? it's that incoming loss times the other query value

        // remember, 0 is query and 1 is key

        // when we want to find the loss w.r.t to our query,
        // we worry about smax losses associated with our own thread
        float qloss = dLdScores[i][threadIdx];
        // when we want to find the loss w.r.t our key,
        // we worry about smax losses associated with other threads
        // this is very slow, 32 clock cycle read
        float kloss = dLdScores[threadIdx][i];

        for (int j = 0; j < kNumFeatures; j++) {
            // find other key weight for query
            dLdxw[0].m[j][threadIdx] += qloss * xw[1].m[j][i];
            // do same for key
            dLdxw[1].m[j][threadIdx] += kloss * xw[0].m[j][i];
        }
    }
}

void gemm_output() {
    // mult the scores matrix with other stuff

    // first clear to zero
    for (int i = 0; i < kNumFeatures; i++) {
        output.m[i][threadIdx] = 0.0f;
    }

    for (int i = 0; i < kTransformerItems; i++) {
        for (int j = 0; j < kNumFeatures; j++) {
            // blend each feature
            output.m[j][threadIdx] += scores[i][threadIdx] * xw[2].m[j][i];
        }
    }

}

void bwd_gemm_output() {
    // need to calculate backward scores and value matrix
    // we can do this as a gather operation instead

    // first clear to zero
    for (int i = 0; i < kNumFeatures; i++) {
        dLdxw[2].m[i][threadIdx] = 0.0f;
    }

    for (int i = 0; i < kTransformerItems; i++) {
        float dLdCurrentScore = 0.0f;
        for (int j = 0; j < kNumFeatures; j++) {
            // blend each feature
            // output.m[j][threadIdx] += scores[i][threadIdx] * xw[2].m[j][i];

            // score * value
            // thus, to get dL deposited on score, do dL_in * value
            // thus, to get dL deposited on value, do dL_in * score

            // we want to find the loss deposted on our scores and values
            // other threads can mind their own business doing the same thing
            // thus no need for parallel reduction

            // other loss times their score for us score gives us our value loss
            dLdxw[2].m[j][threadIdx] += dLdOutput.m[j][i] * scores[threadIdx][i];

            // our loss times times their value gives loss for our score on them
            dLdCurrentScore += dLdOutput.m[j][threadIdx] * xw[2].m[j][i];
        }
        dLdScores[i][threadIdx] = dLdCurrentScore;
    }

}

void transformer(const int idx) {
    threadIdx = idx;
    workIdx = uint2(idx % 5, idx / 5);

    printSetPixel(workIdx);

    // load the values from meory
    float4 curIllum = gIllumination[workIdx];
    for (int i = 0; i < 4; i++) {
        src.m[i][threadIdx] = curIllum[i];
    }

    float4 normZ = gNormalZ[workIdx];
    for (int i = 0; i < 4; i++) {
        src.m[i + 4][threadIdx] = normZ[i];
    }

    GroupMemoryBarrierWithGroupSync();

    gemm_src(0); // query
    gemm_src(1); // key
    gemm_src(2); // values

    GroupMemoryBarrierWithGroupSync();

    gemm_scores_and_smax();

    gemm_output();

    float4 filtered;
    for (int i = 0; i < 4; i++) {
        filtered[i] = output.m[i][threadIdx];
    }

    gFiltered[workIdx] = filtered;
}

void bwd_transformer(const int idx) {
    transformer(idx);

    float4 loss = readDerivBuf4(drIllum, workIdx, gIllumination, 0);
    for (int i = 0; i < 4; i++) {
        dLdOutput.m[i][threadIdx] = loss[i];
    }

    bwd_gemm_output();
    GroupMemoryBarrierWithGroupSync();

    bwd_gemm_scores_and_smax();
    GroupMemoryBarrierWithGroupSync();

    for (int i = 0; i < kNumFeatures; i++) {
        dLdSource[i][threadIdx] = 0.0f;
    }

    bwd_gemm_src(0);
    bwd_gemm_src(1);
    bwd_gemm_src(2);

    // now we need to dump this stuff somewhere
    float4 dLdCurPixel;
    for (int i = 0; i < 4; i++) {
        dLdCurPixel[i] = dLdSource[i][threadIdx];
    }

    GroupMemoryBarrierWithGroupSync();

    const int NUM_TOTAL_WEIGHTS = 3 * kNumFeatures * kNumFeatures;
    int readIdx = 4 * threadIdx;
    while (readIdx < NUM_TOTAL_WEIGHTS) {
        float4 writeVal;
        for (int i = readIdx; i < readIdx + 4; i++) {
            int colIdx = i % kNumFeatures;
            int rowIdx = (i / kNumFeatures) % kNumFeatures;
            int mtxIdx = (i / kNumFeatures) / kNumFeatures;

            writeVal[i % 4] = dLdWeightMatrices[mtxIdx][rowIdx][colIdx];
        }

        storeDerivBuf4(daWeightMatrices, int2(0, 0), writeVal, gIllumination, readIdx / 4);

        readIdx += 4 * kTransformerItems;
    }
}

void simple_gemm_test(const int idx, bool flushResult) {
    threadIdx = idx;
    workIdx = uint2(idx % 5, idx / 5);

    printSetPixel(workIdx);

    // load the values from meory
    float4 curIllum = gIllumination[workIdx];

    for (int i = 0; i < 4; i++) {
        src.m[i][threadIdx] = curIllum[i];
    }

    GroupMemoryBarrierWithGroupSync();

    gemm_src(2); // values

    if (flushResult) {
        float4 filtered;
        for (int i = 0; i < 4; i++) {
            filtered[i] = xw[2].m[i][threadIdx];
        }

        gFiltered[workIdx] = filtered;
    }
}

void bwd_simple_gemm_test(const int idx) {
    simple_gemm_test(idx, false);

    float4 loss = readDerivBuf4(drIllum, workIdx, gIllumination, 0);
    for (int i = 0; i < 4; i++) {
        dLdxw[2].m[i][threadIdx] = loss[i];
    }

    bwd_gemm_src(2);

    // now we need to store store derivatives somewhere

#if 0
    const int totalValuesToStore = 3 * kNumFeatures * kNumFeatures;

    int baseIdx = threadIdx * 4;
    while (baseIdx < totalValuesToStore) {
        float4 writeVal;

        for (int i = 0; i < 4; i++) {
            if (baseIdx + i < totalValuesToStore) {
                int linearIdx = baseIdx + i;

                int a = linearIdx % kNumFeatures;
                int b = linearIdx / kNumFeatures;
                int c = linearIdx / (kNumFeatures * kNumFeatures);

                writeVal[i] = dLdWeightMatrices[c][b][a];
            } else {
                break;
            }
        }

        storeDerivBuf4(daWeightMatrices, int2(0, 0), writeVal, gIllumination, baseIdx / 4);
        baseIdx += 4 * kTransformerItems;
    }
    #else
    // 12 4-vectors to store
    if (threadIdx < 12) {
        float4 writeVal;

        int rowIdx = threadIdx % 4;
        int mtxIdx = threadIdx / 4;
        for (int i = 0; i < 4; i++) {
            writeVal[i] = dLdWeightMatrices[mtxIdx][rowIdx][i];
        }

        storeDerivBuf4(daWeightMatrices, int2(0, 0), writeVal, gIllumination, threadIdx);
    }

    #endif

}
