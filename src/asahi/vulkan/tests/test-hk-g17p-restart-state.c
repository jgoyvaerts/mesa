/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_g17p_restart_state.h"

#include <assert.h>
#include <stdio.h>

int
main(void)
{
   /* A run of draws in one stream: only the first and the index changes
    * (u16 -> u32 -> u16, then a custom index of 0) carry the record. */
   bool known = false;
   uint32_t held = 0;
   const uint32_t seq[] = {0xffff, 0xffff, 0xffffffff, 0xffffffff, 0xffff, 0};
   const bool want[] = {true, false, true, false, true, true};
   for (unsigned i = 0; i < sizeof(seq) / sizeof(seq[0]); i++) {
      const bool need = hk_g17p_restart_state_needed(false, known, held, seq[i]);
      assert(need == want[i]);
      if (need) {
         known = true;
         held = seq[i];
      }
   }

   /* Zero is a legal custom index: an unset stream must not match it. */
   assert(hk_g17p_restart_state_needed(false, false, 0, 0));
   /* A fresh bind tail always carries it, even with the index held. */
   assert(hk_g17p_restart_state_needed(true, true, 0xffff, 0xffff));
   assert(!hk_g17p_restart_state_needed(false, true, 0xffff, 0xffff));

   printf("PASS\n");
   return 0;
}
