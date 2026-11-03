#ifndef CF_LAB_FPMATH_H
#define CF_LAB_FPMATH_H

// Bit-reproducible single-precision math: the same bits on the STM32 (Cortex-M4F) and on a host CPU.
//
// Only IEEE-754 single-precision +, -, *, / and integer operations are used, in a fixed order, so no result
// depends on a platform's libm (newlib's sinf and glibc's sinf can differ in the last bit). Build every file that
// must agree across platforms with -ffp-contract=off and never with -ffast-math: GCC otherwise fuses a * b + c into
// one rounding on the Cortex-M4F (VFMA) but not on a host without FMA, and the two builds drift apart.
//
// Accuracy: cf_expf is Cephes' expf (about 1 ulp); cf_sincosf uses Cephes' polynomials after a three-constant
// reduction by pi/2 and is accurate to about 1 ulp for |x| < 1e4.

#ifdef __cplusplus
extern "C" {
#endif

float cf_expf(float x);
float cf_sigmoidf(float x);
float cf_tanhf(float x);
float cf_siluf(float x);  // x * sigmoid(x)
void cf_sincosf(float x, float *s, float *c);

#ifdef __cplusplus
}
#endif

#endif
