// https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_chromakey.c
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
#define BGR_TO_U(bgr) (((- FIXNUM(0.16874) * bgr[2] - FIXNUM(0.33126) * bgr[1] + FIXNUM(0.50000) * bgr[0] + (1 << 9) - 1) >> 10) + 128)
#define BGR_TO_V(bgr) (((  FIXNUM(0.50000) * bgr[2] - FIXNUM(0.41869) * bgr[1] - FIXNUM(0.08131) * bgr[0] + (1 << 9) - 1) >> 10) + 128)
#define av_clipd(v, v_min, v_max) (max(min(v, v_max), v_min))


__device__  float do_chromakey_pixel_diff(
    float similarity, float blend,
    float * diff_list)
{
    float diff = 0.0;
    int i;

    for (i = 0; i < 9; ++i) {
        diff += diff_list[i];
    }

    diff /= 9.0;

    if (blend > 0.0001) {
        return av_clipd((diff - similarity) / blend, 0.0, 1.0);
    } else {
        return (diff > similarity) ? 1.0 : 0.0;
    }
}

// ---------------------------------------
__global__ void rendering_kernel(const int h, const int w,
        const unsigned char * chromakey_bgr, const float * similarity_blend,
        unsigned char * img,
        float * img_diff)
{
    const unsigned char * p_tmp;
    int u, v, du, dv;
    float diff;

    // ---------------------------------------------------------
    unsigned char chromakey_uv[2];
    chromakey_uv[0] = BGR_TO_U(chromakey_bgr);
    chromakey_uv[1] = BGR_TO_V(chromakey_bgr);

    // ---------------------------------------------------------
    // rgb2uv
    CUDA_KERNEL_LOOP_y(jj, h){
        CUDA_KERNEL_LOOP_x(ii, w){
            int idx_base = jj * w + ii;
            p_tmp = img + idx_base * 3;
            u = BGR_TO_U(p_tmp);
            v = BGR_TO_V(p_tmp);

            // diff
            du = u - chromakey_uv[0];
            dv = v - chromakey_uv[1];
            diff = sqrt((du * du + dv * dv) / (255.0 * 255.0 * 2));
            img_diff[idx_base] = diff;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------
    int xo, yo;
    int x, y;

    float diff_list[9];
    const float similarity = similarity_blend[0];
    const float blend = similarity_blend[1];

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
                    diff_list[yo * 3 + xo] = img_diff[idx_base_tmp];
                }
            }

            float alpha = do_chromakey_pixel_diff(similarity, blend, diff_list);
            img[idx_base * 3 + 0] = img[idx_base * 3 + 0] * alpha;
            img[idx_base * 3 + 1] = img[idx_base * 3 + 1] * alpha;
            img[idx_base * 3 + 2] = img[idx_base * 3 + 2] * alpha;
        }
    }
}
