#include "cf_lab/fpmath.h"

#include <stdint.h>

// Round half away from zero. The float-to-int conversion truncates on both the Cortex-M4F (VCVT) and x86 (CVTTSS2SI),
// and no libm rounding function is involved.
static inline int32_t round_to_int(float x) {
  return (int32_t)(x + (x >= 0.0f ? 0.5f : -0.5f));
}

float cf_expf(float x) {
  // Inside [-87, 88] the result is a normal float, so the 2^n scaling below never needs subnormals or infinity.
  if (x > 88.0f) {
    x = 88.0f;
  }
  if (x < -87.0f) {
    x = -87.0f;
  }
  const int32_t n = round_to_int(x * 1.44269504088896341f);
  const float fn = (float)n;
  // x - n ln2 in two steps (Cody-Waite): the high part of ln2 has few bits, so fn * 0.693359375f is exact.
  float r = x - fn * 0.693359375f;
  r = r - fn * -2.12194440e-4f;
  const float r2 = r * r;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r2 + r;
  p = p + 1.0f;
  union {
    uint32_t u;
    float f;
  } scale = {.u = (uint32_t)(n + 127) << 23};
  return p * scale.f;
}

float cf_sigmoidf(float x) {
  return 1.0f / (1.0f + cf_expf(-x));
}

float cf_siluf(float x) {
  return x / (1.0f + cf_expf(-x));
}

float cf_tanhf(float x) {
  const float a = x >= 0.0f ? x : -x;
  const float t = cf_expf(-2.0f * a);
  const float y = (1.0f - t) / (1.0f + t);
  return x >= 0.0f ? y : -y;
}

// sin and cos of r for |r| <= pi/4.
static inline float sin_kernel(float r) {
  const float r2 = r * r;
  float p = -1.9515295891e-4f;
  p = p * r2 + 8.3321608736e-3f;
  p = p * r2 + -1.6666654611e-1f;
  return r + r * r2 * p;
}

static inline float cos_kernel(float r) {
  const float r2 = r * r;
  float p = 2.443315711809948e-5f;
  p = p * r2 + -1.388731625493765e-3f;
  p = p * r2 + 4.166664568298827e-2f;
  return 1.0f - 0.5f * r2 + r2 * r2 * p;
}

void cf_sincosf(float x, float *s, float *c) {
  const int32_t k = round_to_int(x * 0.636619772367581343f);  // x / (pi/2)
  const float fk = (float)k;
  // x - k pi/2 with pi/2 split into three parts (Cephes' DP1-3, doubled); the leading products are exact for the
  // moderate k this library sees, so the reduction loses no accuracy there.
  float r = x - fk * 1.5703125f;
  r = r - fk * 4.837512969970703125e-4f;
  r = r - fk * 7.54978995489188216e-8f;
  const float sr = sin_kernel(r);
  const float cr = cos_kernel(r);
  switch (k & 3) {
    case 0:
      *s = sr;
      *c = cr;
      break;
    case 1:
      *s = cr;
      *c = -sr;
      break;
    case 2:
      *s = -sr;
      *c = -cr;
      break;
    default:
      *s = -cr;
      *c = sr;
      break;
  }
}
