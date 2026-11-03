#ifndef CF_LAB_NN_H
#define CF_LAB_NN_H

// Dense layers and small multilayer perceptrons that compute the same bits on the STM32 and on a host.
//
// Formats (cf_nn_format_t):
//   f32     float32. CF_NN_F32 and CF_NN_F32_X4 give identical bits (same order of operations per output, the second
//           computes four outputs per pass over the input); CF_NN_F32_FMA uses fmaf, which rounds once per multiply-
//           add on both platforms and so differs from the other two but agrees with itself across platforms.
//   s8      int8 activations and weights, the TFLite / CMSIS-NN arithmetic: asymmetric activations (zero points),
//           symmetric per-channel weights, int32 bias, Q31 multiplier and shift per output channel.
//   s16s8   int16 symmetric activations, int8 per-channel weights, int32 accumulation.
//   s16     int16 symmetric activations and weights, int64 accumulation.
// An activation after an integer layer is a lookup table from the requantized pre-activation to the output: 256
// entries for s8, CF_NN_LUT16_SIZE interpolated knots for the int16 formats (the host computes the tables).
//
// Layout (the cf_lab.nn exporter writes it): in_pad is the input count rounded up to a multiple of 4, with zero
// weights in the padding columns. The rows are stored so that the kernels walk the weights strictly sequentially, one
// word at a time, which keeps one pointer live instead of one per row and turns flash and cache traffic into whole
// consecutive lines:
//   CF_NN_F32               row-major [out][in_pad];
//   CF_NN_F32_X4, _FMA      blocks of four rows, [out/4][in_pad][4]: for each input, the four rows' weights;
//   CF_NN_S8, CF_NN_S16S8   blocks of four rows, [out/4][in_pad/4][4][4]: for each group of four inputs, each row's
//                           four int8 weights as one word, stored (w0, w2, w1, w3) so that SXTB16 of the word gives
//                           (w0, w1) and SXTB16 after an 8-bit rotation gives (w2, w3), multiplying the inputs in
//                           their natural order with SMLAD;
//   CF_NN_S16               blocks of two rows, [out/2][in_pad/4][2][4]: for each group of four inputs, each row's four
//                           int16 weights.
// A partial last block is padded with zero rows, and b (and mult, shift) hold a value for every padded row.
//
// On a Cortex-M4 (__ARM_FEATURE_DSP) the integer kernels use SXTB16 / SMLAD / SMLALD; elsewhere, or with
// CF_NN_PORTABLE defined, a plain C loop computes the same integer sums. Build with -ffp-contract=off and without
// -ffast-math (see cf_lab/fpmath.h). x and y of a layer must not overlap.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CF_NN_ACT_NONE = 0, CF_NN_ACT_RELU = 1, CF_NN_ACT_SILU = 2 } cf_nn_act_t;

typedef enum {
  CF_NN_F32 = 0,
  CF_NN_F32_X4 = 1,
  CF_NN_F32_FMA = 2,
  CF_NN_S8 = 3,
  CF_NN_S16S8 = 4,
  CF_NN_S16 = 5,
} cf_nn_format_t;

#define CF_NN_LUT16_SIZE 513  // 512 segments of 128 int16 steps

typedef struct {
  uint16_t in, in_pad, out;
  uint8_t act;  // cf_nn_act_t
  const float *w;
  const float *b;
} cf_nn_f32_layer_t;

typedef struct {
  uint16_t in, in_pad, out;
  int16_t in_offset;   // minus the input zero point
  int16_t out_offset;  // zero point of the requantized output (the pre-activation when lut is set)
  const int8_t *w;
  const int32_t *b;
  const int32_t *mult;   // Q31 multiplier per output channel
  const int32_t *shift;  // per output channel, CMSIS-NN convention (> 0 left, < 0 right); the exporter emits <= 0
  const int8_t *lut;     // 256 entries indexed by q + 128, or NULL
} cf_nn_s8_layer_t;

typedef struct {
  uint16_t in, in_pad, out;
  const int8_t *w;
  const int32_t *b;
  const int32_t *mult;   // y = (acc * mult + 2^(shift - 1)) >> shift per output channel, 0 < mult < 2^15
  const int32_t *shift;  // 1 .. 62
  const int16_t *lut;    // CF_NN_LUT16_SIZE knots, or NULL
} cf_nn_s16s8_layer_t;

typedef struct {
  uint16_t in, in_pad, out;
  const int16_t *w;
  const int64_t *b;
  const int32_t *mult;
  const int32_t *shift;
  const int16_t *lut;
} cf_nn_s16_layer_t;

typedef struct {
  uint8_t format;  // cf_nn_format_t
  uint8_t n_layers;
  uint16_t in, out, max_width;  // max_width: the largest in_pad or out of any layer
  const void *layers;           // n_layers layer structs of the format
  // Integer formats: input i is quantized as round(x_i * in_scale[i] + in_bias[i]) and the output dequantized as
  // (q - out_zero) * out_scale. Any input normalisation is folded into in_scale, in_bias and the first layer.
  const float *in_scale;
  const float *in_bias;
  float out_scale;
  int32_t out_zero;
} cf_nn_net_t;

void cf_nn_dense_f32(const cf_nn_f32_layer_t *layer, const float *x, float *y);
void cf_nn_dense_f32_x4(const cf_nn_f32_layer_t *layer, const float *x, float *y);
void cf_nn_dense_f32_fma(const cf_nn_f32_layer_t *layer, const float *x, float *y);
// scratch: layer->in_pad int16 values, 4-byte aligned (the input expanded with its offset).
void cf_nn_dense_s8(const cf_nn_s8_layer_t *layer, const int8_t *x, int8_t *y, int16_t *scratch);
void cf_nn_dense_s16s8(const cf_nn_s16s8_layer_t *layer, const int16_t *x, int16_t *y);
void cf_nn_dense_s16(const cf_nn_s16_layer_t *layer, const int16_t *x, int16_t *y);

// The requantization and table steps the kernels use, for tests and host tools.
int32_t cf_nn_requant_q31(int32_t acc, int32_t mult, int32_t shift);
int32_t cf_nn_requant_s64(int64_t acc, int32_t mult, int32_t shift);
int16_t cf_nn_lut16(const int16_t *lut, int32_t q);

// Bytes of 4-byte aligned work memory cf_nn_run needs for net.
uint32_t cf_nn_work_bytes(const cf_nn_net_t *net);
// Runs the network on net->in floats and writes net->out floats.
void cf_nn_run(const cf_nn_net_t *net, const float *x, float *y, void *work);
// cf_nn_run on n rows of x (row-major) into n rows of y.
void cf_nn_run_batch(const cf_nn_net_t *net, const float *x, float *y, int n, void *work);
// Index of the largest of n values, the first one on ties.
int cf_nn_argmax(const float *y, int n);

#ifdef __cplusplus
}
#endif

#endif
