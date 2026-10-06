/*
 * fastmem.c -- word-at-a-time memcpy and memset for the GAP8 fabric controller.
 *
 * The GAP8 FreeRTOS SDK's own (vendors/gwt/libs/src/string.c) move one byte per
 * loop iteration, and every FC copy goes through them: each CPX packet is copied
 * three times on its way to the SPI link (FreeRTOS's queue copy included), the
 * JPEG segments are merged with them, a 3-level DORY input is staged with them.
 *
 * fastmem.mk links these in their place for every caller with --wrap, so nothing
 * else changes. The FC core (RI5CY) splits a misaligned word access into two
 * aligned ones in hardware, so only the destination is aligned here: a CPX
 * packet's payload sits at offset 22, which never lines up with a frame's.
 */
#include <stddef.h>
#include <stdint.h>

typedef uint32_t __attribute__((may_alias)) fastmem_word_t;

/* Keep GCC from turning the loops back into a call to memcpy/memset, which
 * --wrap would route here again. */
#define FASTMEM_NO_LIBCALL __attribute__((optimize("no-tree-loop-distribute-patterns")))

FASTMEM_NO_LIBCALL void *__wrap_memcpy(void *dst, const void *src, size_t n) {
  uint8_t *d = (uint8_t *)dst;
  const uint8_t *s = (const uint8_t *)src;
  while (n != 0u && ((uintptr_t)d & 3u) != 0u) {
    *d++ = *s++;
    n--;
  }
  fastmem_word_t *dw = (fastmem_word_t *)d;
  const fastmem_word_t *sw = (const fastmem_word_t *)s;
  while (n >= 16u) {
    const uint32_t a = sw[0], b = sw[1], c = sw[2], e = sw[3];
    dw[0] = a;
    dw[1] = b;
    dw[2] = c;
    dw[3] = e;
    dw += 4;
    sw += 4;
    n -= 16u;
  }
  while (n >= 4u) {
    *dw++ = *sw++;
    n -= 4u;
  }
  d = (uint8_t *)dw;
  s = (const uint8_t *)sw;
  while (n != 0u) {
    *d++ = *s++;
    n--;
  }
  return dst;
}

FASTMEM_NO_LIBCALL void *__wrap_memset(void *dst, int value, size_t n) {
  uint8_t *d = (uint8_t *)dst;
  const uint8_t byte = (uint8_t)value;
  while (n != 0u && ((uintptr_t)d & 3u) != 0u) {
    *d++ = byte;
    n--;
  }
  const uint32_t word = byte * 0x01010101u;
  fastmem_word_t *dw = (fastmem_word_t *)d;
  while (n >= 16u) {
    dw[0] = word;
    dw[1] = word;
    dw[2] = word;
    dw[3] = word;
    dw += 4;
    n -= 16u;
  }
  while (n >= 4u) {
    *dw++ = word;
    n -= 4u;
  }
  d = (uint8_t *)dw;
  while (n != 0u) {
    *d++ = byte;
    n--;
  }
  return dst;
}
