#include <limits.h>
#include <stdint.h>

#include "cf_qdepthwise.h"
#include "cf_qdepthwise_internal.h"

static uint8_t cf_qdepthwise_clip_u8(int32_t value) {
  if (value < 0) return 0u;
  if (value > UINT8_MAX) return UINT8_MAX;
  return (uint8_t)value;
}

static uint8_t cf_qdepthwise_quantize(
    const cf_qdepthwise_u8s8_u8_job_t *job, uint16_t channel,
    int32_t accumulator) {
  switch (job->epilogue) {
    case CF_QDEPTHWISE_U8_SHIFT_CLIP:
      return cf_qdepthwise_clip_u8(accumulator >> job->output_shift);
    case CF_QDEPTHWISE_U8_REQUANT_RELU: {
      const int16_t result =
          (int16_t)(((int16_t)job->output_multiplier * accumulator) >>
                    job->output_shift);
      return cf_qdepthwise_clip_u8(result);
    }
    case CF_QDEPTHWISE_U8_BATCHNORM_RELU:
      return cf_qdepthwise_clip_u8(
          (job->batch_norm_multiplier[channel] * accumulator +
           job->batch_norm_bias[channel]) >>
          job->output_shift);
    default:
      return 0u;
  }
}

void cf_qdepthwise_u8s8_u8_reference(
    const cf_qdepthwise_u8s8_u8_job_t *job) {
  uint16_t channel;

  if (job == 0 || job->input == 0 || job->weights == 0 || job->output == 0 ||
      job->input_width == 0u || job->input_height == 0u ||
      job->input_channels == 0u ||
      job->input_channels != job->output_channels ||
      job->output_width == 0u || job->output_height == 0u ||
      job->kernel_width == 0u || job->kernel_height == 0u ||
      job->stride_width == 0u || job->stride_height == 0u ||
      job->output_shift > 31u ||
      job->epilogue > CF_QDEPTHWISE_U8_BATCHNORM_RELU ||
      (job->epilogue == CF_QDEPTHWISE_U8_BATCHNORM_RELU &&
       (job->batch_norm_multiplier == 0 || job->batch_norm_bias == 0))) {
    return;
  }

  for (channel = 0u; channel < job->output_channels; ++channel) {
    const uint8_t *const input_plane =
        job->input +
        (uint32_t)channel * job->input_height * job->input_width;
    const int8_t *const weight =
        job->weights +
        (uint32_t)channel * job->kernel_height * job->kernel_width;
    uint16_t output_y;

    for (output_y = 0u; output_y < job->output_height; ++output_y) {
      uint16_t output_x;
      for (output_x = 0u; output_x < job->output_width; ++output_x) {
        int32_t sum = cf_qdepthwise_bias_at(job->bias, channel);
        uint16_t kernel_y;

        for (kernel_y = 0u; kernel_y < job->kernel_height; ++kernel_y) {
          const int32_t input_y =
              (int32_t)output_y * job->stride_height + kernel_y -
              job->padding_top;
          uint16_t kernel_x;
          for (kernel_x = 0u; kernel_x < job->kernel_width; ++kernel_x) {
            const int32_t input_x =
                (int32_t)output_x * job->stride_width + kernel_x -
                job->padding_left;
            if (input_y >= 0 && input_y < job->input_height && input_x >= 0 &&
                input_x < job->input_width) {
              sum += (int32_t)input_plane[(uint32_t)input_y *
                                              job->input_width +
                                          (uint32_t)input_x] *
                     weight[(uint32_t)kernel_y * job->kernel_width +
                            kernel_x];
            }
          }
        }
        job->output[((uint32_t)output_y * job->output_width + output_x) *
                        job->output_channels +
                    channel] = cf_qdepthwise_quantize(job, channel, sum);
      }
    }
  }
}
