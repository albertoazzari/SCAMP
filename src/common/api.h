#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void compute_mp(const double* ts_1, size_t ts1_len, const double* ts_2, size_t ts2_len, int window_size, float* mp, int* indexes);
void compute_selfmp(const double* ts, size_t ts_len, int window_size, float* mp, int* indexes);

#ifdef __cplusplus
}
#endif