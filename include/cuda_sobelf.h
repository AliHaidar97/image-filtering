
typedef int pgrey;

#ifdef __cplusplus
extern "C" {
#endif

void apply_filter_cuda(pgrey *p, int width, int height, int position, int size, int threshold);

// return false if there are no gpu available
bool cuda_init(int mpirank);

#ifdef __cplusplus
}
#endif