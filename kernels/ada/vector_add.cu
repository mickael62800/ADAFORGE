// Addition FP32 et résiduel, avec protection du dernier bloc partiel.
extern "C" __global__ void vector_add(
    const float* a, const float* b, float* output, unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = a[i] + b[i];
    }
}
