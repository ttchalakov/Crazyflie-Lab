/*
 * jpeg_dct.c -- integer forward DCT.
 *
 * GAP8 has no FPU, so a float DCT would be software-emulated and cost hundreds
 * of cycles per multiply. This is the classic slow-but-accurate integer DCT
 * (Loeffler/Ligtenberg/Moschytz factorization, as used by libjpeg's
 * jpeg_fdct_islow): two passes of 8-point transforms, 13-bit fixed-point
 * constants, 12 multiplies per transform.
 *
 * Output convention: the result is the true DCT scaled up by 8. The quantizer
 * divides by (qtab[i] << 3) to undo it.
 *
 * "Basic and trivial" on purpose -- portable scalar C, identical on host and
 * GAP8, so byte-exact comparison works. The FC and cluster encoders both call
 * this one function.
 */

#include "jpeg_gray.h"

#define CONST_BITS 13
#define PASS1_BITS 2

#define FIX_0_298631336 2446
#define FIX_0_390180644 3196
#define FIX_0_541196100 4433
#define FIX_0_765366865 6270
#define FIX_0_899976223 7373
#define FIX_1_175875602 9633
#define FIX_1_501321110 12299
#define FIX_1_847759065 15137
#define FIX_1_961570560 16069
#define FIX_2_053119869 16819
#define FIX_2_562915447 20995
#define FIX_3_072711026 25172

#define DESCALE(x, n) (((x) + (((int32_t)1) << ((n) - 1))) >> (n))

void jpeg_fdct_8x8(int16_t data[64])
{
    int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int32_t tmp10, tmp11, tmp12, tmp13;
    int32_t z1, z2, z3, z4, z5;
    int32_t ws[64];
    int i;

    /* Pass 1: rows. Results are scaled up by 2**PASS1_BITS to keep precision. */
    for (i = 0; i < 8; i++) {
        const int16_t *p = data + i * 8;
        int32_t *w = ws + i * 8;

        tmp0 = p[0] + p[7];
        tmp7 = p[0] - p[7];
        tmp1 = p[1] + p[6];
        tmp6 = p[1] - p[6];
        tmp2 = p[2] + p[5];
        tmp5 = p[2] - p[5];
        tmp3 = p[3] + p[4];
        tmp4 = p[3] - p[4];

        /* even part */
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        w[0] = (tmp10 + tmp11) << PASS1_BITS;
        w[4] = (tmp10 - tmp11) << PASS1_BITS;

        z1 = (tmp12 + tmp13) * FIX_0_541196100;
        w[2] = DESCALE(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        w[6] = DESCALE(z1 - tmp12 * FIX_1_847759065, CONST_BITS - PASS1_BITS);

        /* odd part */
        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = (z3 + z4) * FIX_1_175875602;

        tmp4 = tmp4 * FIX_0_298631336;
        tmp5 = tmp5 * FIX_2_053119869;
        tmp6 = tmp6 * FIX_3_072711026;
        tmp7 = tmp7 * FIX_1_501321110;
        z1 = z1 * -FIX_0_899976223;
        z2 = z2 * -FIX_2_562915447;
        z3 = z3 * -FIX_1_961570560;
        z4 = z4 * -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        w[7] = DESCALE(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        w[5] = DESCALE(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        w[3] = DESCALE(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        w[1] = DESCALE(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    /* Pass 2: columns. Removes the PASS1_BITS scaling on the way out. */
    for (i = 0; i < 8; i++) {
        int32_t *w = ws + i;

        tmp0 = w[8 * 0] + w[8 * 7];
        tmp7 = w[8 * 0] - w[8 * 7];
        tmp1 = w[8 * 1] + w[8 * 6];
        tmp6 = w[8 * 1] - w[8 * 6];
        tmp2 = w[8 * 2] + w[8 * 5];
        tmp5 = w[8 * 2] - w[8 * 5];
        tmp3 = w[8 * 3] + w[8 * 4];
        tmp4 = w[8 * 3] - w[8 * 4];

        /* even part */
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        data[8 * 0 + i] = (int16_t)DESCALE(tmp10 + tmp11, PASS1_BITS);
        data[8 * 4 + i] = (int16_t)DESCALE(tmp10 - tmp11, PASS1_BITS);

        z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[8 * 2 + i] = (int16_t)DESCALE(z1 + tmp13 * FIX_0_765366865,
                                           CONST_BITS + PASS1_BITS);
        data[8 * 6 + i] = (int16_t)DESCALE(z1 - tmp12 * FIX_1_847759065,
                                           CONST_BITS + PASS1_BITS);

        /* odd part */
        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = (z3 + z4) * FIX_1_175875602;

        tmp4 = tmp4 * FIX_0_298631336;
        tmp5 = tmp5 * FIX_2_053119869;
        tmp6 = tmp6 * FIX_3_072711026;
        tmp7 = tmp7 * FIX_1_501321110;
        z1 = z1 * -FIX_0_899976223;
        z2 = z2 * -FIX_2_562915447;
        z3 = z3 * -FIX_1_961570560;
        z4 = z4 * -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        data[8 * 7 + i] = (int16_t)DESCALE(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        data[8 * 5 + i] = (int16_t)DESCALE(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        data[8 * 3 + i] = (int16_t)DESCALE(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        data[8 * 1 + i] = (int16_t)DESCALE(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }
}
