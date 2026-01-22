#ifndef HARDWARE_REGS_H
#define HARDWARE_REGS_H

#include <exec/types.h>

// Custom Chip Base Address
#define CUSTOM_BASE 0xdff000
#define AMIGA_CUSTOM ((volatile struct Custom *)CUSTOM_BASE)

// We can use the system headers <hardware/custom.h> which define 'struct Custom'
// This file is for additional helpers or if we want to avoid headers (but using AmigaOS headers is better)

#include <hardware/custom.h>
#include <hardware/dmabits.h>
#include <hardware/intbits.h>

// Helper macros
#define REG_COP1LC  (*(volatile ULONG *)(CUSTOM_BASE + 0x080))
#define REG_COP2LC  (*(volatile ULONG *)(CUSTOM_BASE + 0x084))
#define REG_COPJMP1 (*(volatile UWORD *)(CUSTOM_BASE + 0x088))
#define REG_COLOR00 (*(volatile UWORD *)(CUSTOM_BASE + 0x180))

#endif
