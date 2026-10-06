/*
 * dory_freertos_compat.h -- what DORY's generated runtime assumes, supplied.
 *
 * DORY emits code for a PULP-OS application, where the archi/ register-map
 * headers and the event-unit intrinsics are already in scope. A FreeRTOS build
 * of the same sources is missing both, and the failures are unhelpful:
 *
 *   dory_dma.c: 'ARCHI_MCHAN_DEMUX_ADDR' undeclared   <- cluster DMA base address
 *   dory_dma.c: 'ARCHI_CL_EVT_DMA0' undeclared        <- DMA-completion event bit
 *   mchan.h:    'eu_evt_maskWaitAndClr' undeclared    <- spelled differently here
 *
 * dory.mk force-includes this file (-include) into every translation unit rather
 * than patching DORY's sources, so the generated tree stays byte-identical to
 * what DORY produced and there is nothing to re-derive when DORY is updated.
 */
#ifndef CF_LAB_GAP8_DORY_DORY_FREERTOS_COMPAT_H
#define CF_LAB_GAP8_DORY_DORY_FREERTOS_COMPAT_H

/* -include reaches the assembler too, and the SDK builds .S files with the same
 * flags. Everything below is C. */
#ifndef __ASSEMBLER__

#include <math.h>

/* Register map and cluster properties: MCHAN's demux base and the event bit its
 * completion signal is delivered on.
 *
 * Guarded because -include applies to every translation unit in the build, SDK
 * sources included, and those are not compiled with gap_archi on their include
 * path. Without the guard the force-include breaks the SDK rather than the app. */
#if defined(__has_include)
#  if __has_include("archi/chips/gap8/memory_map.h")
#    include "archi/chips/gap8/memory_map.h"
#    include "archi/chips/gap8/properties.h"
#  endif
#endif

/* Same intrinsic, different spelling in the FreeRTOS SDK. */
#define eu_evt_maskWaitAndClr hal_eu_evt_mask_wait_and_clr

#endif /* __ASSEMBLER__ */
#endif /* CF_LAB_GAP8_DORY_DORY_FREERTOS_COMPAT_H */
