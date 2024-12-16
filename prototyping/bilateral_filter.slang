// returns total weight
float bilteral_filter(
    int num_features,
    int width,
    int height,
    int b,
    int y,
    int x,
    TensorView<float> input,
    TensorView<float> params,
    TensorView<float> output,
    int kernel_boundary, // SIGNED!!!!!!
    int dialation,
    const bool normalize
) {
    float total_weight = 0.0f;

    for(int yoff = -kernel_boundary; yoff <= kernel_boundary; yoff++) {
        for (int xoff = -kernel_boundary; xoff <= kernel_boundary; xoff++) {
            int xp = x + xoff * dialation;
            int yp = y + yoff * dialation;

            // TODO: reflection padding
            if (xp < 0 || yp < 0 || xp >= width || yp >= height) {
                continue;
            }

            float sample_weight = 1.0;
            for (int i = 0; i < num_features + 2; i++) {
                float center_val, sample_val;

                if (i < num_features) {
                    center_val = input[b, i, y, x];
                    sample_val = input[b, i, yp, xp];
                } else {
                    center_val = 0;
                    sample_val = (i == num_features ? yoff : xoff); // positional encoding
                }


                float diff = sample_val - center_val;

                // in the original bilateral filter equations we divide by the variance
                // we choose to multiply by a learned value representing 1/variance because:
                // 1) computers are faster with multiplication than division
                // 2) gradient descent will in theory be more stable with multiplication
                // we also self-contain the negative within the weight
                // we will prevent positive weights via gradient clipping
                float weight = exp(params[i] * diff * diff);

                sample_weight *= weight;
            }

            for (int i = 0; i < num_features; i++) {
                float sample_val = input[b, i, yp, xp];
                output[b, i, y, x] += sample_weight * sample_val;
            }

            total_weight += sample_weight;
        }
    }

    total_weight = (total_weight < 0.001 ? 0.001 : total_weight);

    if (normalize) {
        for (int i = 0; i < num_features; i++) {
            output[b, i, y, x] /= total_weight;
        }
    }


    return total_weight;
}

#define GROUP_ICOUNT 256
#define MAX_PARAMS 24
groupshared float shr_dLdParam[GROUP_ICOUNT][MAX_PARAMS];

void bwd_bilteral_filter(
    int num_features,
    int width,
    int height,
    int b,
    int y,
    int x,
    TensorView<float> input,
    TensorView<float> input_grad,
    TensorView<float> params,
    TensorView<float> params_grad,
    TensorView<float> output,
    TensorView<float> output_grad,
    int kernel_boundary,
    int dialation
) {
    int shr_idx = cudaThreadIdx().x % GROUP_ICOUNT;

    if (cudaThreadIdx().x < GROUP_ICOUNT) {
        for (int i = 0; i < MAX_PARAMS; i++) {
            shr_dLdParam[cudaThreadIdx().x][i] = 0.0;
        }
    }

    // output grad is placeholder tensor
    float total_weight = bilteral_filter(num_features, width, height, b, y, x, input, params, output, kernel_boundary, dialation, false);

    float dLdTotalWeight = 0.0;
    for (int i = 0; i < num_features; i++) {
        dLdTotalWeight += -output_grad[b, i, y, x] * output[b, i, y, x] / (total_weight * total_weight);
    }

    for (int yoff = -kernel_boundary; yoff <= kernel_boundary; yoff++) {
        for (int xoff = -kernel_boundary; xoff <= kernel_boundary; xoff++) {
            int xp = x + xoff * dialation;
            int yp = y + yoff * dialation;

            // TODO: reflection padding
            if (xp < 0 || yp < 0 || xp >= width || yp >= height) {
                continue;
            }

            float sample_weight = 1.0;
            for (int i = 0; i < num_features + 2; i++) {
                float center_val, sample_val;

                if (i < num_features) {
                    center_val = input[b, i, y, x];
                    sample_val = input[b, i, yp, xp];
                } else {
                    center_val = 0;
                    sample_val = (i == num_features ? yoff : xoff); // positional encoding
                }

                float diff = sample_val - center_val;

                float weight = exp(params[i] * diff * diff);

                sample_weight *= weight;
            }

            float dLdSampleWeight = dLdTotalWeight;
            for (int i = 0; i < num_features; i++) {
                dLdSampleWeight += output_grad[b, i, y, x] * input[b, i, yp, xp] / total_weight;
            }

            // for each parameter and feature, compute the gradient
            for (int i = 0; i < num_features + 2; i++) {
                float center_val, sample_val;

                if (i < num_features) {
                    center_val = input[b, i, y, x];
                    sample_val = input[b, i, yp, xp];
                } else {
                    center_val = 0;
                    sample_val = (i == num_features ? yoff : xoff);
                }

                float diff = sample_val - center_val;

                float dLdParam = dLdSampleWeight * sample_weight * diff * diff;
                InterlockedAdd(shr_dLdParam[shr_idx][i], dLdParam);
                //params_grad.InterlockedAdd(i, dLdParam, junk);

                if (i < num_features) {
                    float junk;

                    float dLdCenterVal = dLdSampleWeight * sample_weight * params[i] * -2 * diff;
                    input_grad.InterlockedAdd(int4(b, i, y, x), dLdCenterVal, junk);

                    float dLdSampleVal = output_grad[b, i, y, x] * sample_weight / total_weight - dLdCenterVal;
                    input_grad.InterlockedAdd(int4(b, i, yp, xp), dLdSampleVal, junk);
                }

            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    int j = cudaThreadIdx().x;
    if (j < num_features + 2) {
        for (int i = 1; i < GROUP_ICOUNT; i++) {
            #if 1
            shr_dLdParam[0][j] += shr_dLdParam[i][j];
            #else
            float grad_val = shr_dLdParam[i][j];
            if(isnan(grad_val)) {
                grad_val = 0.0;
            }

            float next_val = shr_dLdParam[0][j] + grad_val;

            if(!isnan(next_val)) {
                shr_dLdParam[0][j] = next_val;
            }
            #endif
        }

        float junk;
        params_grad.InterlockedAdd(j, shr_dLdParam[0][j], junk);
    }



}

#define SETUP_PIXEL_CONTEXT()                                                 \
    int globalIdx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x; \
    int batch_size = input.size(0);                                          \
    int num_features = input.size(1);                                        \
    int width = input.size(3);                                               \
    int height = input.size(2);                                              \
    if (globalIdx >= batch_size * width * height) {                                                                         \
        return;                                                               \
    }                                                                         \
    int x = globalIdx % width;                                                \
    int y = (globalIdx / width) % height;                                     \
    int b = globalIdx / (width * height);

[CudaKernel]
[AutoPyBindCUDA]
void exec_bilateral_filter_wrapper(
    TensorView<float> input,
    TensorView<float> params,
    TensorView<float> output,
    int kernel_boundary,
    int dialation
) {
    SETUP_PIXEL_CONTEXT();
    // use junk inputs for gradient
    bilteral_filter(num_features, width, height, b, y, x, input, params, output, kernel_boundary, dialation, true);
}

[CudaKernel]
[AutoPyBindCUDA]
void bwd_bilateral_filter_wrapper(
    TensorView<float> input,
    TensorView<float> input_grad,
    TensorView<float> params,
    TensorView<float> params_grad,
    TensorView<float> output,
    TensorView<float> output_grad,
    int kernel_boundary,
    int dialation
) {
    SETUP_PIXEL_CONTEXT();

    bwd_bilteral_filter( num_features, width, height, b, y, x, input, input_grad, params, params_grad, output, output_grad, kernel_boundary, dialation);
}
