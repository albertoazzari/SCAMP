#include <vector>
#include <cstring>
#include "api.h"
#include "common/common.h"
#include "common/scamp_args.h"
#include "common/scamp_interface.h"
#include "common/scamp_utils.h"

using namespace std;

void SplitProfile1NNINDEX_(const std::vector<uint64_t> profile,
                          vector<float>& nn, vector<int>& index,
                          bool output_pearson, int window) {
  auto nn_ptr = nn.data();
  auto index_ptr = index.data();
  int count = 0;
  for (auto& elem : profile) {
    SCAMP::mp_entry e;
    e.ulong = elem;
    if (output_pearson) {
      nn_ptr[count] = CleanupPearson(e.floats[0]);
    } else {
      nn_ptr[count] = ConvertToEuclidean(e.floats[0], window);
    }
    index_ptr[count] = e.ints[1];
    count++;
  }
}

SCAMP::SCAMPArgs GetDefaultSCAMPArgs_() {
  auto profile_type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  SCAMP::SCAMPArgs args;
  args.has_b = false;
  args.max_tile_size = 128000;
  args.distributed_start_row = -1;
  args.distributed_start_col = -1;
  args.distance_threshold = 0;
  args.precision_type = SCAMP::PRECISION_DOUBLE;
  args.profile_type = profile_type;
  args.computing_rows = true;
  args.computing_columns = true;
  args.keep_rows_separate = false;
  args.is_aligned = false;
  args.silent_mode = true;
  args.max_matches_per_column = 5;
  args.matrix_height = 50;
  args.matrix_width = 50;
  args.profile_a.type = profile_type;
  args.profile_b.type = profile_type;

  return args;
}

bool setup_and_do_SCAMP_(SCAMP::SCAMPArgs* args) {
  std::vector<int> gpus;
  int num_cpus = 0;
  bool pearson = false;

  if (gpus.empty() && num_cpus == 0) {
    SCAMP::do_SCAMP(args);
  } else {
    SCAMP::do_SCAMP(args, gpus, num_cpus);
  }
  return pearson;
}

void compute_mp(const double* ts_1, size_t ts1_len, const double* ts_2, size_t ts2_len, int window_size, float* mp, int* indexes) {
    SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs_();
    args.timeseries_a = vector<double>(ts_1, ts_1 + ts1_len);
    args.timeseries_b = vector<double>(ts_2, ts_2 + ts2_len);
    args.window = window_size;
    args.has_b = false;
    args.computing_rows = true;
    args.computing_columns = true;

    bool output_pearson = setup_and_do_SCAMP_(&args);

    vector<float> result_nn(args.profile_a.data[0].uint64_value.size());
    vector<int> result_index(args.profile_a.data[0].uint64_value.size());
    SplitProfile1NNINDEX_(args.profile_a.data[0].uint64_value, result_nn,
                        result_index, output_pearson, args.window);
    memcpy(mp, result_nn.data(), result_nn.size() * sizeof(float));
    memcpy(indexes, result_index.data(), result_index.size() * sizeof(int));
}

void compute_selfmp(const double* ts, size_t ts_len, int window_size, float* mp, int* indexes) {
    compute_mp(ts, ts_len, ts, ts_len, window_size, mp, indexes);
}