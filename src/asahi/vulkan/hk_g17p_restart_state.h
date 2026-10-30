/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Whether a native HAL200 direct indexed draw must be preceded by a restart
 * State Update. Every VDM State Update drains the vertex pipeline (on the
 * M4 Pro a run of 10k-triangle draws loses about 3.5 us per update), so the
 * record is only emitted when the stream does not already hold `index`
 * (known/known_index: hk_cs::restart_index_set/restart_index). A draw after a
 * fresh bind tail always carries it, keeping Metal's selector, state, draw
 * order there. */
static inline bool
hk_g17p_restart_state_needed(bool tail_emitted, bool known,
                             uint32_t known_index, uint32_t index)
{
   return tail_emitted || !known || known_index != index;
}
