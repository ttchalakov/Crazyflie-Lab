#include "cf_lab/nn.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "cf_lab/fpmath.h"

#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP == 1 && !defined(CF_NN_PORTABLE)
#define CF_NN_DSP 1
#include <arm_acle.h>
#else
#define CF_NN_DSP 0
#endif

static inline uint32_t ld32(const void *p) {
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

#if CF_NN_DSP
static inline uint32_t sxtb16(uint32_t w) {
  return (uint32_t)__sxtb16((int8x4_t)w);
}

// SXTB16 with its operand rotated by 8 bits, one instruction (GCC does not fold the rotation on its own).
static inline uint32_t sxtb16_ror8(uint32_t w) {
  uint32_t r;
  __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(w));
  return r;
}

static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc) {
  return __smlad((int16x2_t)a, (int16x2_t)b, acc);
}

static inline int64_t smlald(uint32_t a, uint32_t b, int64_t acc) {
  return __smlald((int16x2_t)a, (int16x2_t)b, acc);
}
#endif

static inline int8_t sat8(int32_t v) {
  return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
}

static inline int16_t sat16(int32_t v) {
  return (int16_t)(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
}

// Round half away from zero after clamping to [lo, hi]; a NaN maps to lo. No libm call, same bits everywhere.
static inline int32_t quantize(float v, float lo, float hi) {
  if (!(v >= lo)) {
    v = lo;
  }
  if (v > hi) {
    v = hi;
  }
  return (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
}

// ---- requantization ----

// CMSIS-NN's arm_nn_requantize: a saturating rounding doubling high multiply (VQRDMULH) followed by a rounding
// shift right with the midpoint away from zero. Written for any host: CMSIS divides an int64 by (1UL << 31), which
// is a signed division on the 32-bit Cortex-M4 but an unsigned one where unsigned long has 64 bits.
int32_t cf_nn_requant_q31(int32_t acc, int32_t mult, int32_t shift) {
  const int32_t left = shift > 0 ? shift : 0;
  const int32_t right = shift > 0 ? 0 : -shift;
  const int32_t m1 = (int32_t)((uint32_t)acc << left);
  int64_t prod = ((m1 < 0) != (mult < 0)) ? 1 - ((int64_t)1 << 30) : ((int64_t)1 << 30);
  prod += (int64_t)m1 * mult;
  int32_t hi = (int32_t)(prod / ((int64_t)1 << 31));
  if (m1 == mult && m1 == INT32_MIN) {
    hi = INT32_MAX;
  }
  const int32_t mask = (int32_t)(((int64_t)1 << right) - 1);
  const int32_t remainder = hi & mask;
  int32_t result = hi >> right;
  int32_t threshold = mask >> 1;
  if (result < 0) {
    threshold++;
  }
  if (remainder > threshold) {
    result++;
  }
  return result;
}

int32_t cf_nn_requant_s64(int64_t acc, int32_t mult, int32_t shift) {
  const int64_t v = (acc * (int64_t)mult + ((int64_t)1 << (shift - 1))) >> shift;
  return v < INT32_MIN ? INT32_MIN : (v > INT32_MAX ? INT32_MAX : (int32_t)v);
}

int16_t cf_nn_lut16(const int16_t *lut, int32_t q) {
  const uint32_t u = (uint32_t)(q + 32768);
  const uint32_t k = u >> 7;
  const int32_t f = (int32_t)(u & 127u);
  const int32_t a = lut[k];
  const int32_t b = lut[k + 1];
  return sat16(a + (((b - a) * f + 64) >> 7));
}

// ---- float32 ----

static inline float act_f32(float v, uint8_t act) {
  if (act == CF_NN_ACT_RELU) {
    return v > 0.0f ? v : 0.0f;
  }
  if (act == CF_NN_ACT_SILU) {
    return cf_siluf(v);
  }
  return v;
}

// Row-major weights, one output at a time.
void cf_nn_dense_f32(const cf_nn_f32_layer_t *layer, const float *x, float *y) {
  const uint32_t n = layer->in_pad;
  for (uint32_t o = 0; o < layer->out; o++) {
    const float *w = layer->w + (size_t)o * n;
    float acc = layer->b[o];
    for (uint32_t i = 0; i < n; i++) {
      acc += w[i] * x[i];
    }
    y[o] = act_f32(acc, layer->act);
  }
}

static inline void store4_f32(const cf_nn_f32_layer_t *layer, uint32_t o, const float acc[4], float *y) {
  for (uint32_t j = 0; j < 4 && o + j < layer->out; j++) {
    y[o + j] = act_f32(acc[j], layer->act);
  }
}

// Rows interleaved by four (see nn.h): each output still sums in the order of cf_nn_dense_f32.
void cf_nn_dense_f32_x4(const cf_nn_f32_layer_t *layer, const float *x, float *y) {
  const uint32_t n = layer->in_pad;
  for (uint32_t o = 0; o < layer->out; o += 4) {
    const float *w = layer->w + (size_t)o * n;
    float a0 = layer->b[o], a1 = layer->b[o + 1], a2 = layer->b[o + 2], a3 = layer->b[o + 3];
    for (uint32_t i = 0; i < n; i++, w += 4) {
      const float xi = x[i];
      a0 += w[0] * xi;
      a1 += w[1] * xi;
      a2 += w[2] * xi;
      a3 += w[3] * xi;
    }
    const float acc[4] = {a0, a1, a2, a3};
    store4_f32(layer, o, acc, y);
  }
}

void cf_nn_dense_f32_fma(const cf_nn_f32_layer_t *layer, const float *x, float *y) {
  const uint32_t n = layer->in_pad;
  for (uint32_t o = 0; o < layer->out; o += 4) {
    const float *w = layer->w + (size_t)o * n;
    float a0 = layer->b[o], a1 = layer->b[o + 1], a2 = layer->b[o + 2], a3 = layer->b[o + 3];
    for (uint32_t i = 0; i < n; i++, w += 4) {
      const float xi = x[i];
      a0 = fmaf(w[0], xi, a0);
      a1 = fmaf(w[1], xi, a1);
      a2 = fmaf(w[2], xi, a2);
      a3 = fmaf(w[3], xi, a3);
    }
    const float acc[4] = {a0, a1, a2, a3};
    store4_f32(layer, o, acc, y);
  }
}

// ---- int8 weights ----

// One block of four rows (see nn.h for the layout) times n int16 inputs in natural order, n a multiple of 4.
static inline void block4_w8(const int8_t *w, const int16_t *x, uint32_t n, int32_t acc[4]) {
#if CF_NN_DSP
  int32_t a0 = acc[0], a1 = acc[1], a2 = acc[2], a3 = acc[3];
  const int16_t *end = x + n;
  while (x < end) {
    const uint32_t xa = ld32(x);
    const uint32_t xb = ld32(x + 2);
    uint32_t u = ld32(w);
    a0 = smlad(sxtb16(u), xa, a0);
    a0 = smlad(sxtb16_ror8(u), xb, a0);
    u = ld32(w + 4);
    a1 = smlad(sxtb16(u), xa, a1);
    a1 = smlad(sxtb16_ror8(u), xb, a1);
    u = ld32(w + 8);
    a2 = smlad(sxtb16(u), xa, a2);
    a2 = smlad(sxtb16_ror8(u), xb, a2);
    u = ld32(w + 12);
    a3 = smlad(sxtb16(u), xa, a3);
    a3 = smlad(sxtb16_ror8(u), xb, a3);
    w += 16;
    x += 4;
  }
  acc[0] = a0;
  acc[1] = a1;
  acc[2] = a2;
  acc[3] = a3;
#else
  for (uint32_t g = 0; g < n; g += 4, w += 16) {
    for (uint32_t r = 0; r < 4; r++) {
      const int8_t *v = w + 4 * r;
      acc[r] += v[0] * x[g] + v[2] * x[g + 1] + v[1] * x[g + 2] + v[3] * x[g + 3];
    }
  }
#endif
}

void cf_nn_dense_s8(const cf_nn_s8_layer_t *layer, const int8_t *x, int8_t *y, int16_t *scratch) {
  const uint32_t n = layer->in_pad;
  for (uint32_t i = 0; i < n; i++) {
    scratch[i] = (int16_t)(x[i] + layer->in_offset);
  }
  for (uint32_t o = 0; o < layer->out; o += 4) {
    int32_t acc[4] = {layer->b[o], layer->b[o + 1], layer->b[o + 2], layer->b[o + 3]};
    block4_w8(layer->w + (size_t)o * n, scratch, n, acc);
    for (uint32_t j = 0; j < 4 && o + j < layer->out; j++) {
      const int8_t q = sat8(cf_nn_requant_q31(acc[j], layer->mult[o + j], layer->shift[o + j]) + layer->out_offset);
      y[o + j] = layer->lut != NULL ? layer->lut[(int32_t)q + 128] : q;
    }
  }
}

static inline int16_t out_s16(const int16_t *lut, int32_t q) {
  const int16_t v = sat16(q);
  return lut != NULL ? cf_nn_lut16(lut, v) : v;
}

void cf_nn_dense_s16s8(const cf_nn_s16s8_layer_t *layer, const int16_t *x, int16_t *y) {
  const uint32_t n = layer->in_pad;
  for (uint32_t o = 0; o < layer->out; o += 4) {
    int32_t acc[4] = {layer->b[o], layer->b[o + 1], layer->b[o + 2], layer->b[o + 3]};
    block4_w8(layer->w + (size_t)o * n, x, n, acc);
    for (uint32_t j = 0; j < 4 && o + j < layer->out; j++) {
      y[o + j] = out_s16(layer->lut, cf_nn_requant_s64(acc[j], layer->mult[o + j], layer->shift[o + j]));
    }
  }
}

// ---- int16 weights ----

// One block of two rows (see nn.h) times n int16 inputs, n a multiple of 4.
static inline void block2_w16(const int16_t *w, const int16_t *x, uint32_t n, int64_t acc[2]) {
#if CF_NN_DSP
  int64_t a0 = acc[0], a1 = acc[1];
  const int16_t *end = x + n;
  while (x < end) {
    const uint32_t xa = ld32(x);
    const uint32_t xb = ld32(x + 2);
    a0 = smlald(ld32(w), xa, a0);
    a0 = smlald(ld32(w + 2), xb, a0);
    a1 = smlald(ld32(w + 4), xa, a1);
    a1 = smlald(ld32(w + 6), xb, a1);
    w += 8;
    x += 4;
  }
  acc[0] = a0;
  acc[1] = a1;
#else
  for (uint32_t g = 0; g < n; g += 4, w += 8) {
    for (uint32_t r = 0; r < 2; r++) {
      const int16_t *v = w + 4 * r;
      acc[r] += (int64_t)v[0] * x[g] + (int64_t)v[1] * x[g + 1] + (int64_t)v[2] * x[g + 2] + (int64_t)v[3] * x[g + 3];
    }
  }
#endif
}

void cf_nn_dense_s16(const cf_nn_s16_layer_t *layer, const int16_t *x, int16_t *y) {
  const uint32_t n = layer->in_pad;
  for (uint32_t o = 0; o < layer->out; o += 2) {
    int64_t acc[2] = {layer->b[o], layer->b[o + 1]};
    block2_w16(layer->w + (size_t)o * n, x, n, acc);
    for (uint32_t j = 0; j < 2 && o + j < layer->out; j++) {
      y[o + j] = out_s16(layer->lut, cf_nn_requant_s64(acc[j], layer->mult[o + j], layer->shift[o + j]));
    }
  }
}

// ---- networks ----

static inline uint32_t padded_width(const cf_nn_net_t *net) {
  return ((uint32_t)net->max_width + 3u) & ~3u;
}

uint32_t cf_nn_work_bytes(const cf_nn_net_t *net) {
  const uint32_t w = padded_width(net);
  return 2u * w * (uint32_t)sizeof(float) + w * (uint32_t)sizeof(int16_t);
}

static void run_f32(const cf_nn_net_t *net, const float *x, float *y, float *a, float *b) {
  const cf_nn_f32_layer_t *layers = (const cf_nn_f32_layer_t *)net->layers;
  memcpy(a, x, net->in * sizeof(float));
  for (uint32_t l = 0; l < net->n_layers; l++) {
    const cf_nn_f32_layer_t *layer = &layers[l];
    for (uint32_t i = layer->in; i < layer->in_pad; i++) {
      a[i] = 0.0f;
    }
    if (net->format == CF_NN_F32_FMA) {
      cf_nn_dense_f32_fma(layer, a, b);
    } else if (net->format == CF_NN_F32_X4) {
      cf_nn_dense_f32_x4(layer, a, b);
    } else {
      cf_nn_dense_f32(layer, a, b);
    }
    float *t = a;
    a = b;
    b = t;
  }
  memcpy(y, a, net->out * sizeof(float));
}

static void run_s8(const cf_nn_net_t *net, const float *x, float *y, int8_t *a, int8_t *b, int16_t *scratch) {
  const cf_nn_s8_layer_t *layers = (const cf_nn_s8_layer_t *)net->layers;
  for (uint32_t i = 0; i < net->in; i++) {
    a[i] = (int8_t)quantize(x[i] * net->in_scale[i] + net->in_bias[i], -128.0f, 127.0f);
  }
  for (uint32_t l = 0; l < net->n_layers; l++) {
    const cf_nn_s8_layer_t *layer = &layers[l];
    for (uint32_t i = layer->in; i < layer->in_pad; i++) {
      a[i] = 0;
    }
    cf_nn_dense_s8(layer, a, b, scratch);
    int8_t *t = a;
    a = b;
    b = t;
  }
  for (uint32_t i = 0; i < net->out; i++) {
    y[i] = (float)((int32_t)a[i] - net->out_zero) * net->out_scale;
  }
}

static void run_s16(const cf_nn_net_t *net, const float *x, float *y, int16_t *a, int16_t *b) {
  for (uint32_t i = 0; i < net->in; i++) {
    a[i] = (int16_t)quantize(x[i] * net->in_scale[i] + net->in_bias[i], -32768.0f, 32767.0f);
  }
  for (uint32_t l = 0; l < net->n_layers; l++) {
    if (net->format == CF_NN_S16S8) {
      const cf_nn_s16s8_layer_t *layer = &((const cf_nn_s16s8_layer_t *)net->layers)[l];
      for (uint32_t i = layer->in; i < layer->in_pad; i++) {
        a[i] = 0;
      }
      cf_nn_dense_s16s8(layer, a, b);
    } else {
      const cf_nn_s16_layer_t *layer = &((const cf_nn_s16_layer_t *)net->layers)[l];
      for (uint32_t i = layer->in; i < layer->in_pad; i++) {
        a[i] = 0;
      }
      cf_nn_dense_s16(layer, a, b);
    }
    int16_t *t = a;
    a = b;
    b = t;
  }
  for (uint32_t i = 0; i < net->out; i++) {
    y[i] = (float)((int32_t)a[i] - net->out_zero) * net->out_scale;
  }
}

void cf_nn_run(const cf_nn_net_t *net, const float *x, float *y, void *work) {
  const uint32_t w = padded_width(net);
  switch (net->format) {
    case CF_NN_S8: {
      int8_t *a = (int8_t *)work;
      run_s8(net, x, y, a, a + w, (int16_t *)(void *)(a + 2u * w));
      break;
    }
    case CF_NN_S16S8:
    case CF_NN_S16: {
      int16_t *a = (int16_t *)work;
      run_s16(net, x, y, a, a + w);
      break;
    }
    default: {
      float *a = (float *)work;
      run_f32(net, x, y, a, a + w);
      break;
    }
  }
}

void cf_nn_run_batch(const cf_nn_net_t *net, const float *x, float *y, int n, void *work) {
  for (int i = 0; i < n; i++) {
    cf_nn_run(net, x + (size_t)i * net->in, y + (size_t)i * net->out, work);
  }
}

int cf_nn_argmax(const float *y, int n) {
  int k = 0;
  for (int i = 1; i < n; i++) {
    if (y[i] > y[k]) {
      k = i;
    }
  }
  return k;
}
