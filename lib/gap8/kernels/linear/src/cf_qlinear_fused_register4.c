#include <stdint.h>

#include "pmsis.h"

#include "cf_qlinear.h"

#define CF_QLINEAR_DOTP4(a, b, c) __builtin_pulp_sdotusp4((a), (b), (c))
#define CF_QLINEAR_CLIP_U8(value) __builtin_pulp_clipu_r((value), 255)
#define CF_QLINEAR_ALWAYS_INLINE static inline __attribute__((always_inline))

typedef void (*cf_qlinear_fused_worker_t)(void *argument);

CF_QLINEAR_ALWAYS_INLINE uint32_t cf_qlinear_fused_min_u32(uint32_t a,
                                                            uint32_t b) {
  return a < b ? a : b;
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_fused_output_range(
    uint32_t output_dim, uint32_t core, uint32_t cores, uint32_t *start,
    uint32_t *stop) {
  const uint32_t chunk = (output_dim + cores - 1u) / cores;

  *start = cf_qlinear_fused_min_u32(core * chunk, output_dim);
  *stop = cf_qlinear_fused_min_u32(*start + chunk, output_dim);
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_fused_block4_range(
    uint32_t output_dim, uint32_t core, uint32_t cores, uint32_t *first,
    uint32_t *last) {
  const uint32_t blocks = output_dim >> 2;
  const uint32_t chunk = (blocks + cores - 1u) / cores;

  *first = cf_qlinear_fused_min_u32(core * chunk, blocks);
  *last = cf_qlinear_fused_min_u32(*first + chunk, blocks);
}

CF_QLINEAR_ALWAYS_INLINE int32_t cf_qlinear_fused_bias(
    const cf_qlinear_u8s8_u8_job_t *job, uint32_t output_index) {
  return job->bias != 0 ? job->bias[output_index] : 0;
}

/* mode is a literal at every worker call site. always_inline lets GCC remove
 * this switch before emitting the output loop, so quantization flags are not
 * tested once per neuron. */
CF_QLINEAR_ALWAYS_INLINE uint8_t cf_qlinear_fused_quantize(
    const cf_qlinear_u8s8_u8_job_t *job, uint32_t output_index,
    int32_t accumulator, cf_qlinear_u8_epilogue_t mode) {
  switch (mode) {
    case CF_QLINEAR_U8_SHIFT_CLIP:
      return CF_QLINEAR_CLIP_U8(accumulator >> job->output_shift);
    case CF_QLINEAR_U8_REQUANT_RELU: {
      const int32_t multiplier = (int16_t)job->output_multiplier;
      const int16_t requantized =
          (int16_t)((multiplier * accumulator) >> job->output_shift);
      return CF_QLINEAR_CLIP_U8(requantized);
    }
    case CF_QLINEAR_U8_BATCHNORM_RELU: {
      const int32_t normalized =
          job->batch_norm_multiplier[output_index] * accumulator +
          job->batch_norm_bias[output_index];
      return CF_QLINEAR_CLIP_U8(normalized >> job->output_shift);
    }
    default:
      return 0u;
  }
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_fused_row(
    const cf_qlinear_u8s8_u8_job_t *job, uint32_t output_index,
    cf_qlinear_u8_epilogue_t mode) {
  const v4u *input = (const v4u *)job->input;
  const v4s *weight =
      (const v4s *)(job->weights +
                    output_index * job->weight_row_stride_bytes);
  const uint32_t words = (uint32_t)job->packed_input_dim >> 2;
  int32_t sum = cf_qlinear_fused_bias(job, output_index);
  uint32_t packed;

  for (packed = 0u; packed < words; ++packed) {
    sum = CF_QLINEAR_DOTP4(input[packed], weight[packed], sum);
  }
  job->output[output_index] =
      cf_qlinear_fused_quantize(job, output_index, sum, mode);
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_fused_tile4(
    const cf_qlinear_u8s8_u8_job_t *job, uint32_t output_index,
    cf_qlinear_u8_epilogue_t mode) {
  const v4u *input = (const v4u *)job->input;
  const v4s *weight0 =
      (const v4s *)(job->weights +
                    output_index * job->weight_row_stride_bytes);
  const v4s *weight1 =
      (const v4s *)(job->weights +
                    (output_index + 1u) * job->weight_row_stride_bytes);
  const v4s *weight2 =
      (const v4s *)(job->weights +
                    (output_index + 2u) * job->weight_row_stride_bytes);
  const v4s *weight3 =
      (const v4s *)(job->weights +
                    (output_index + 3u) * job->weight_row_stride_bytes);
  const uint32_t words = (uint32_t)job->packed_input_dim >> 2;
  int32_t sum0 = cf_qlinear_fused_bias(job, output_index);
  int32_t sum1 = cf_qlinear_fused_bias(job, output_index + 1u);
  int32_t sum2 = cf_qlinear_fused_bias(job, output_index + 2u);
  int32_t sum3 = cf_qlinear_fused_bias(job, output_index + 3u);
  uint8_t output0;
  uint8_t output1;
  uint8_t output2;
  uint8_t output3;
  uint32_t packed;

  for (packed = 0u; packed < words; ++packed) {
    const v4u activation = input[packed];
    sum0 = CF_QLINEAR_DOTP4(activation, weight0[packed], sum0);
    sum1 = CF_QLINEAR_DOTP4(activation, weight1[packed], sum1);
    sum2 = CF_QLINEAR_DOTP4(activation, weight2[packed], sum2);
    sum3 = CF_QLINEAR_DOTP4(activation, weight3[packed], sum3);
  }

  /* Keep mode-specific parameters live once for the whole register tile. The
   * literal mode passed by each specialized worker removes this switch. */
  switch (mode) {
    case CF_QLINEAR_U8_SHIFT_CLIP: {
      const int8_t shift = job->output_shift;
      output0 = CF_QLINEAR_CLIP_U8(sum0 >> shift);
      output1 = CF_QLINEAR_CLIP_U8(sum1 >> shift);
      output2 = CF_QLINEAR_CLIP_U8(sum2 >> shift);
      output3 = CF_QLINEAR_CLIP_U8(sum3 >> shift);
      break;
    }
    case CF_QLINEAR_U8_REQUANT_RELU: {
      const int32_t multiplier = (int16_t)job->output_multiplier;
      const int8_t shift = job->output_shift;
      output0 = CF_QLINEAR_CLIP_U8(
          (int16_t)((multiplier * sum0) >> shift));
      output1 = CF_QLINEAR_CLIP_U8(
          (int16_t)((multiplier * sum1) >> shift));
      output2 = CF_QLINEAR_CLIP_U8(
          (int16_t)((multiplier * sum2) >> shift));
      output3 = CF_QLINEAR_CLIP_U8(
          (int16_t)((multiplier * sum3) >> shift));
      break;
    }
    case CF_QLINEAR_U8_BATCHNORM_RELU: {
      const int32_t *multiplier =
          job->batch_norm_multiplier + output_index;
      const int32_t *bias = job->batch_norm_bias + output_index;
      const int8_t shift = job->output_shift;
      output0 = CF_QLINEAR_CLIP_U8((multiplier[0] * sum0 + bias[0]) >> shift);
      output1 = CF_QLINEAR_CLIP_U8((multiplier[1] * sum1 + bias[1]) >> shift);
      output2 = CF_QLINEAR_CLIP_U8((multiplier[2] * sum2 + bias[2]) >> shift);
      output3 = CF_QLINEAR_CLIP_U8((multiplier[3] * sum3 + bias[3]) >> shift);
      break;
    }
    default:
      return;
  }

  *((v4u *)(job->output + output_index)) =
      __builtin_pulp_pack4(output0, output1, output2, output3);
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_fused_register1(
    cf_qlinear_u8s8_u8_job_t *job, cf_qlinear_u8_epilogue_t mode) {
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  uint32_t start;
  uint32_t stop;
  uint32_t output_index;

  cf_qlinear_fused_output_range(job->output_dim, core, cores, &start, &stop);
  for (output_index = start; output_index < stop; ++output_index) {
    cf_qlinear_fused_row(job, output_index, mode);
  }
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_fused_register4_full(
    cf_qlinear_u8s8_u8_job_t *job, uint32_t core, uint32_t cores,
    cf_qlinear_u8_epilogue_t mode) {
  uint32_t first;
  uint32_t last;
  uint32_t block;

  cf_qlinear_fused_block4_range(job->output_dim, core, cores, &first, &last);
  for (block = first; block < last; ++block) {
    cf_qlinear_fused_tile4(job, block << 2, mode);
  }
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_fused_register4_tail(
    cf_qlinear_u8s8_u8_job_t *job, cf_qlinear_u8_epilogue_t mode) {
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();

  cf_qlinear_fused_register4_full(job, core, cores, mode);
  if (core == 0u) {
    const uint32_t start = (uint32_t)job->output_dim & ~3u;
    uint32_t output_index;
    for (output_index = start; output_index < job->output_dim; ++output_index) {
      cf_qlinear_fused_row(job, output_index, mode);
    }
  }
}

#define CF_QLINEAR_DEFINE_FUSED_WORKERS(suffix, mode)                         \
  static void cf_qlinear_fused_register1_##suffix(void *argument) {           \
    cf_qlinear_fused_register1((cf_qlinear_u8s8_u8_job_t *)argument, mode);    \
  }                                                                            \
  static void cf_qlinear_fused_register4_aligned_##suffix(void *argument) {    \
    cf_qlinear_u8s8_u8_job_t *job = (cf_qlinear_u8s8_u8_job_t *)argument;      \
    cf_qlinear_fused_register4_full(                                            \
        job, (uint32_t)pi_core_id(), (uint32_t)pi_cl_team_nb_cores(), mode);    \
  }                                                                            \
  static void cf_qlinear_fused_register4_tail_##suffix(void *argument) {       \
    cf_qlinear_fused_register4_tail((cf_qlinear_u8s8_u8_job_t *)argument,      \
                                    mode);                                      \
  }

CF_QLINEAR_DEFINE_FUSED_WORKERS(shift, CF_QLINEAR_U8_SHIFT_CLIP)
CF_QLINEAR_DEFINE_FUSED_WORKERS(requant, CF_QLINEAR_U8_REQUANT_RELU)
CF_QLINEAR_DEFINE_FUSED_WORKERS(batchnorm, CF_QLINEAR_U8_BATCHNORM_RELU)

static cf_qlinear_fused_worker_t cf_qlinear_fused_select_worker(
    const cf_qlinear_u8s8_u8_job_t *job) {
  const uint8_t small = job->output_dim < 4u * job->cores;
  const uint8_t tail = (job->output_dim & 3u) != 0u;

  switch (job->epilogue) {
    case CF_QLINEAR_U8_SHIFT_CLIP:
      return small ? cf_qlinear_fused_register1_shift
                   : tail ? cf_qlinear_fused_register4_tail_shift
                          : cf_qlinear_fused_register4_aligned_shift;
    case CF_QLINEAR_U8_REQUANT_RELU:
      return small ? cf_qlinear_fused_register1_requant
                   : tail ? cf_qlinear_fused_register4_tail_requant
                          : cf_qlinear_fused_register4_aligned_requant;
    case CF_QLINEAR_U8_BATCHNORM_RELU:
      return small ? cf_qlinear_fused_register1_batchnorm
                   : tail ? cf_qlinear_fused_register4_tail_batchnorm
                          : cf_qlinear_fused_register4_aligned_batchnorm;
    default:
      return 0;
  }
}

void cf_qlinear_u8s8_u8_register4_team(void *argument) {
  cf_qlinear_u8s8_u8_job_t *job = (cf_qlinear_u8s8_u8_job_t *)argument;
  cf_qlinear_fused_worker_t worker = cf_qlinear_fused_select_worker(job);

  worker(job);
}
