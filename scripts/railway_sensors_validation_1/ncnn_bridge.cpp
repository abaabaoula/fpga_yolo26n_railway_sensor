#include <net.h>
#include <mat.h>
#include <cstring>
#include <new>
struct Context { ncnn::Net net; };
extern "C" {
void* sv_create(const char* param, const char* bin, int threads) {
    Context* ctx = new (std::nothrow) Context;
    if (!ctx) return nullptr;
    ctx->net.opt.num_threads = threads;
    ctx->net.opt.use_vulkan_compute = false;
    ctx->net.opt.use_fp16_packed = false;
    ctx->net.opt.use_fp16_storage = false;
    ctx->net.opt.use_fp16_arithmetic = false;
    if (ctx->net.load_param(param) || ctx->net.load_model(bin)) { delete ctx; return nullptr; }
    return ctx;
}
int sv_run(void* handle, const float* input, int width, int height,
           float* output, int capacity, int* shape) {
    Context* ctx = static_cast<Context*>(handle);
    if (!ctx || !input || !output || !shape) return -1;
    ncnn::Mat image(width, height, 3);
    for (int c=0; c<3; ++c)
        std::memcpy(image.channel(c), input + c*width*height, width*height*sizeof(float));
    ncnn::Extractor ex = ctx->net.create_extractor();
    if (ex.input("in0", image)) return -2;
    ncnn::Mat result;
    if (ex.extract("out0", result)) return -3;
    shape[0] = result.dims; shape[1] = result.w; shape[2] = result.h; shape[3] = result.c;
    int count = result.w * result.h * result.c * result.d;
    if (result.elempack != 1 || result.elemsize != sizeof(float) || count > capacity) return -4;
    if (result.dims == 2) {
        std::memcpy(output, result.data, count*sizeof(float));
    } else if (result.dims == 3) {
        for (int c=0; c<result.c; ++c)
            std::memcpy(output+c*result.w*result.h, result.channel(c), result.w*result.h*sizeof(float));
    } else return -5;
    return count;
}
void sv_destroy(void* handle) { delete static_cast<Context*>(handle); }
}
