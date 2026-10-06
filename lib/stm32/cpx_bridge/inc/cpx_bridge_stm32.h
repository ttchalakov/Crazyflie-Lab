#ifndef CF_CPX_BRIDGE_STM32_H
#define CF_CPX_BRIDGE_STM32_H
#include "cf_lab/cpx_bridge.h"
bool cf_cpx_bridge_stm32_init(cf_cpx_bridge_t *bridge, CPXTarget_t peer);
/* In your registered CPX app handler, call cf_cpx_bridge_read to copy payload
 * bytes. The library does not replace the firmware's single global callback. */
#endif
