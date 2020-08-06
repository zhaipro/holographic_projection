#define CUDA_KERNEL_LOOP_x(i,n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

#define CUDA_KERNEL_LOOP_y(j,m) \
    for(int j = blockIdx.y * blockDim.y + threadIdx.y; \
        j < (m); \
        j += blockDim.y * gridDim.y)

#define FIXNUM(x) lrint((x) * (1 << 10))
#define RGB_TO_U(rgb) (((- FIXNUM(0.16874) * rgb[0] - FIXNUM(0.33126) * rgb[1] + FIXNUM(0.50000) * rgb[2] + (1 << 9) - 1) >> 10) + 128)
#define RGB_TO_V(rgb) (((  FIXNUM(0.50000) * rgb[0] - FIXNUM(0.41869) * rgb[1] - FIXNUM(0.08131) * rgb[2] + (1 << 9) - 1) >> 10) + 128)
#define av_clipd(v, v_min, v_max) (max(min(v, v_max), v_min))


// ---------------------------------------
__device__  float do_chromakey_pixel_diff(
    float similarity, float blend,
    float * diff_list)
{
    float diff = 0.0;
    int i;

    diff = 10000.0;
    for (i = 0; i < 9; i++) {
        if (diff_list[i] < diff)
        {
            diff = diff_list[i];
        }
    }

    float rst;
    if (blend > 0.0001) {
        rst = av_clipd((diff - similarity) / blend, 0.0, 1.0);
    }
    else {
        rst = (diff > similarity) ? 1 : 0;
    }

    return rst;
}

// ---------------------------------------
__global__ void rendering_kernel(const int h, const int w,
        const unsigned char * chromakey_rgb, const float * similarity_blend,
        const unsigned char * img_in,
        float * img_in_diff,
        unsigned char * img_out)
{
    const unsigned char * p_tmp;
    float uu, vv, du, dv, diff;

    // ---------------------------------------------------------
    unsigned char chromakey_uv[2];
    chromakey_uv[0] = RGB_TO_U(chromakey_rgb);
    chromakey_uv[1] = RGB_TO_V(chromakey_rgb);

    // ---------------------------------------------------------
    // rgb2uv
    CUDA_KERNEL_LOOP_y(jj, h){
        CUDA_KERNEL_LOOP_x(ii, w){
            int idx_base = jj * w + ii;
            p_tmp = img_in + idx_base * 3;
            uu = RGB_TO_U(p_tmp);
            vv = RGB_TO_V(p_tmp);

            // diff
            du = (float)uu - chromakey_uv[0];
            dv = (float)vv - chromakey_uv[1];
            diff = sqrt((du * du + dv * dv) / (255.0 * 255.0));
            img_in_diff[idx_base] = diff;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------
    int xo, yo;
    int x, y;

    float diff_list[9];

    CUDA_KERNEL_LOOP_y(jj, h){
        CUDA_KERNEL_LOOP_x(ii, w){
            int idx_base = jj * w + ii;
            for (yo = 0; yo < 3; ++yo) {
                for (xo = 0; xo < 3; ++xo) {
                    y = jj + yo - 1;
                    x = ii + xo - 1;

                    if (x < 0 || x >= w || y < 0 || y >= h)
                        continue;

                    int idx_base_tmp = w * y + x;
                    diff_list[yo * 3 + xo] = img_in_diff[idx_base_tmp];
                }
            }

            float similarity = similarity_blend[0];
            float blend = similarity_blend[1];
            float alpha = do_chromakey_pixel_diff(similarity, blend, diff_list);

            img_out[idx_base * 3 + 0] = img_in[idx_base * 3 + 0] * alpha;
            img_out[idx_base * 3 + 1] = img_in[idx_base * 3 + 1] * alpha;
            img_out[idx_base * 3 + 2] = img_in[idx_base * 3 + 2] * alpha;
        }
    }
}
