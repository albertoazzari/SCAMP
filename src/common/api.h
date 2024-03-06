#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void compute_scamp(const double* ts, size_t ts_len, int window_size, float* mp, int* indexes);

#ifdef __cplusplus
}
#endif