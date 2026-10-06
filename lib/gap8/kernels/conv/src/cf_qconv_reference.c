#include <limits.h>
#include <stdint.h>

#include "cf_qconv.h"
#include "cf_qconv_bias.h"

static uint8_t cf_qconv_clip_u8(int32_t value) {
  if (value < 0) return 0u;
  if (value > UINT8_MAX) return UINT8_MAX;
  return (uint8_t)value;
}

static uint8_t cf_qconv_quantize(const cf_qconv_u8s8_u8_job_t *job,
                                 uint16_t channel, int32_t accumulator) {
  switch (job->epilogue) {
    case CF_QCONV_U8_SHIFT_CLIP:
      return cf_qconv_clip_u8(accumulator >> job->output_shift);
    case CF_QCONV_U8_REQUANT_RELU: {
      const int32_t multiplier = (int16_t)job->output_multiplier;
      const int16_t result =
          (int16_t)((multiplier * accumulator) >> job->output_shift);
      return cf_qconv_clip_u8(result);
    }
    case CF_QCONV_U8_BATCHNORM_RELU:
      return cf_qconv_clip_u8(
          (job->batch_norm_multiplier[channel] * accumulator +
           job->batch_norm_bias[channel]) >>
          job->output_shift);
    default:
      return 0u;
  }
}

void cf_qconv_u8s8_u8_reference(const cf_qconv_u8s8_u8_job_t *job) {
  uint16_t output_y;

  if (job == 0 || job->input == 0 || job->weights == 0 || job->output == 0 ||
      job->input_width == 0u || job->input_height == 0u ||
      job->input_channels == 0u || job->output_width == 0u ||
      job->output_height == 0u || job->output_channels == 0u ||
      job->kernel_width == 0u || job->kernel_height == 0u ||
      job->stride_width == 0u || job->stride_height == 0u ||
      job->output_shift > 31u ||
      job->epilogue > CF_QCONV_U8_BATCHNORM_RELU ||
      (job->epilogue == CF_QCONV_U8_BATCHNORM_RELU &&
       (job->batch_norm_multiplier == 0 || job->batch_norm_bias == 0))) {
    return;
  }

  for (output_y = 0u; output_y < job->output_height; ++output_y) {
    uint16_t output_x;
    for (output_x = 0u; output_x < job->output_width; ++output_x) {
      uint16_t output_channel;
      for (output_channel = 0u; output_channel < job->output_channels;
           ++output_channel) {
        const int8_t *weight =
            job->weights +
            (uint32_t)output_channel * job->kernel_height *
                job->kernel_width * job->input_channels;
        int32_t sum = cf_qconv_bias_at(job->bias, output_channel);
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
            uint16_t input_channel;
            for (input_channel = 0u; input_channel < job->input_channels;
                 ++input_channel) {
              uint8_t activation = 0u;
              if (input_y >= 0 && input_y < job->input_height &&
                  input_x >= 0 && input_x < job->input_width) {
                activation =
                    job->input[((uint32_t)input_y * job->input_width +
                                (uint32_t)input_x) *
                                   job->input_channels +
                               input_channel];
              }
              sum += (int32_t)activation * *weight++;
            }
          }
        }
        job->output[((uint32_t)output_y * job->output_width + output_x) *
                        job->output_channels +
                    output_channel] =
            cf_qconv_quantize(job, output_channel, sum);
      }
    }
  }
}
