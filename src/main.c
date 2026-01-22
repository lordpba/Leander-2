#include <proto/exec.h>
#include <proto/dos.h>
#include <proto/graphics.h>
#include <proto/intuition.h>
#include <graphics/gfxbase.h>
#include <hardware/custom.h>
#include <hardware/dmabits.h>
#include <hardware/intbits.h>
#include "hardware_regs.h"

// Assets
#include "../assets/test_bg_pal.h"
#include "../assets/test_bg_data.h" 

struct Custom *custom = (struct Custom *)0xdff000;

// Simple Copper List 
struct PopCopper {
    UWORD instructions[200]; // Increased size
};

struct PopCopper copperList __attribute__((section(".chip"))); // Must be in Chip RAM

void create_display() {
    int i = 0;
    int p;
    
    // 1. Bitplane Pointers
    // We have 5 bitplanes (32 colors)
    // BPL1PTH/L to BPL5PTH/L
    // BPLxPT registers start at 0x0E0
    
    ULONG bitplane_addr = (ULONG)test_bg_data;
    ULONG plane_size = (320 / 8) * 256; // Bytes per plane
    
    for(p = 0; p < 5; p++) {
        ULONG addr = bitplane_addr + (p * plane_size);
        
        // BPLxPTH
        copperList.instructions[i++] = 0x00E0 + (p * 4);
        copperList.instructions[i++] = (UWORD)(addr >> 16);
        
        // BPLxPTL
        copperList.instructions[i++] = 0x00E2 + (p * 4);
        copperList.instructions[i++] = (UWORD)(addr & 0xFFFF);
    }
    
    // 2. Palette
    // Load from test_bg_pal
    for(p = 0; p < 32; p++) {
        copperList.instructions[i++] = 0x0180 + (p*2);
        copperList.instructions[i++] = test_bg_pal[p];
    }
    
    // 3. Bitplane Control (Screen resolution etc)
    // BPLCON0: 5 bitplanes, composite color enabled
    // 5 planes = 0x5000. COLOR = 0x0200. Total 0x5200.
    copperList.instructions[i++] = 0x0100; // BPLCON0
    copperList.instructions[i++] = 0x5200;
    
    // BPLCON1: Scrolling (0 for now)
    copperList.instructions[i++] = 0x0102;
    copperList.instructions[i++] = 0x0000;
    
    // BPL1MOD / BPL2MOD: Modulo (0 for now as width matches)
    copperList.instructions[i++] = 0x0108;
    copperList.instructions[i++] = 0x0000;
    copperList.instructions[i++] = 0x010A;
    copperList.instructions[i++] = 0x0000;
    
    // Display Window (DiwStrt, DiwStop, DdfStrt, DdfStop)
    // Standard PAL: 320x256
    // DIWSTRT: 0x2C81
    copperList.instructions[i++] = 0x008E;
    copperList.instructions[i++] = 0x2C81;
    
    // DIWSTOP: 0x2C + 256 lines = 0x12C. 0x81 + 320 pixels = ...
    // Let's use standard values
    copperList.instructions[i++] = 0x0090; // DIWSTOP
    copperList.instructions[i++] = 0xFFC1; // (Simplified high stop)

    // DDFSTRT: 0x0038
    copperList.instructions[i++] = 0x0092;
    copperList.instructions[i++] = 0x0038;
    
    // DDFSTOP: 0x00D0
    copperList.instructions[i++] = 0x0094;
    copperList.instructions[i++] = 0x00D0;

    // End of list
    copperList.instructions[i++] = 0xffff;
    copperList.instructions[i++] = 0xfffe;
}

int main(void) {
    struct Library *SysBase = * ((struct Library **) 4);
    struct Library *GfxBase = OpenLibrary((STRPTR)"graphics.library", 0);
    
    // Save old state
    struct View *oldView = ((struct GfxBase *)GfxBase)->ActiView;
    
    // 1. Allocate Chip RAM for Copper List and Bitplanes
    // The Custom Chips (Copper, Blitter, Denise) CANNOT see Fast RAM.
    // Static data often ends up in Fast RAM. We must copy it to Chip RAM.
    
    // Copper List
    struct PopCopper *myCop = (struct PopCopper *)AllocMem(sizeof(struct PopCopper), MEMF_CHIP | MEMF_CLEAR);
    if (!myCop) return 10; // Fail

    // Bitplanes
    ULONG plane_size = (320 / 8) * 256;
    ULONG total_planes_size = plane_size * 5;
    UBYTE *chipData = (UBYTE *)AllocMem(total_planes_size, MEMF_CHIP | MEMF_CLEAR);
    if (!chipData) {
        FreeMem(myCop, sizeof(struct PopCopper));
        return 20;
    }
    
    // Copy data from our static array (which might be in FastRAM) to Chip RAM
    // Note: This assumes test_bg_data is available globally
    ULONG *src = (ULONG *)test_bg_data;
    ULONG *dst = (ULONG *)chipData;
    int k;
    for(k=0; k < total_planes_size/4; k++) {
        dst[k] = src[k];
    }
    
    // 2. Prepare Copper Instructions in the allocated Chip RAM
    int i = 0;
    int p = 0;
    
    // Bitplane Pointers for INTERLEAVED Layout
    // The Python script produces: Line 0 P0, Line 0 P1, Line 0 P2...
    // Screen Width in Bytes = 320 / 8 = 40.
    // Total Bytes per Row (all planes) = 40 * 5 = 200.
    //
    // Pointers must point to the start of each plane for the first line.
    // P0: Start + 0
    // P1: Start + 40
    // P2: Start + 80
    // ...
    
    ULONG bitplane_addr = (ULONG)chipData;
    int bytes_per_row = 40; // 320 pixels
    
    for(p = 0; p < 5; p++) {
        ULONG addr = bitplane_addr + (p * bytes_per_row);
        
        myCop->instructions[i++] = 0x00E0 + (p * 4);
        myCop->instructions[i++] = (UWORD)(addr >> 16);
        
        myCop->instructions[i++] = 0x00E2 + (p * 4);
        myCop->instructions[i++] = (UWORD)(addr & 0xFFFF);
    }
    
    // Palette
    for(p = 0; p < 32; p++) {
        myCop->instructions[i++] = 0x0180 + (p*2);
        myCop->instructions[i++] = test_bg_pal[p];
    }
    
    // Control registers
    myCop->instructions[i++] = 0x0100; // BPLCON0
    myCop->instructions[i++] = 0x5200; // 5 bitplanes, composite color
    
    myCop->instructions[i++] = 0x0102; // BPLCON1
    myCop->instructions[i++] = 0x0000;
    
    // Modulo for INTERLEAVED
    // We fetch one line of one plane (40 bytes).
    // The next line for THIS plane is 4 lines of other planes away.
    // Jump over P1, P2, P3, P4 lines.
    // Modulo = 4 * 40 = 160.
    
    myCop->instructions[i++] = 0x0108; // Modulo Odd
    myCop->instructions[i++] = 160;
    myCop->instructions[i++] = 0x010A; // Modulo Even
    myCop->instructions[i++] = 160;
    
    myCop->instructions[i++] = 0x008E; // DIWSTRT
    myCop->instructions[i++] = 0x2C81;
    
    myCop->instructions[i++] = 0x0090; // DIWSTOP
    myCop->instructions[i++] = 0xFFC1;

    myCop->instructions[i++] = 0x0092; // DDFSTRT
    myCop->instructions[i++] = 0x0038;
    
    myCop->instructions[i++] = 0x0094; // DDFSTOP
    myCop->instructions[i++] = 0x00D0;

    // End of list
    myCop->instructions[i++] = 0xffff;
    myCop->instructions[i++] = 0xfffe;

    
    // 3. System Takeover
    LoadView(NULL);
    WaitTOF();
    WaitTOF();
    
    OwnBlitter();
    
    // Point Copper to our Chip RAM list
    custom->cop1lc = (ULONG)myCop;
    
    custom->dmacon = DMAF_SETCLR | DMAF_MASTER | DMAF_COPPER | DMAF_RASTER | DMAF_BLITTER;
    
    // 4. Main Loop
    volatile UBYTE *ciaa = (volatile UBYTE *)0xbfe001;
    
    while((*ciaa & 64) != 0) { 
        WaitTOF(); 
    }
    
    // 5. Cleanup
    LoadView(oldView);
    WaitTOF();
    WaitTOF();
    
    custom->dmacon = DMAF_SETCLR | DMAF_MASTER | 0x01FF; 
    custom->cop1lc = (ULONG)((struct GfxBase *)GfxBase)->copinit;
    
    DisownBlitter();
    
    // Free Memory
    FreeMem(chipData, total_planes_size);
    FreeMem(myCop, sizeof(struct PopCopper));
    
    CloseLibrary(GfxBase);
    
    return 0;
}
