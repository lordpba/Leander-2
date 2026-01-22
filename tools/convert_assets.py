import sys
from PIL import Image
import struct

def export_amiga_interleaved(image_path, output_base, depth=4):
    """
    Converts an image to Amiga Interleaved Bitplane format.
    """
    try:
        img = Image.open(image_path)
    except IOError:
        print(f"Error: Could not open {image_path}")
        return

    # Ensure palette mode
    if img.mode != 'P':
        img = img.quantize(colors=2**depth)
    
    width, height = img.size
    
    # Export Palette
    palette = img.getpalette() # [r,g,b, r,g,b, ...]
    if not palette:
        print("Error: No palette found")
        return

    # Create C header for Palette
    with open(f"{output_base}_pal.h", "w") as f:
        f.write(f"// Palette for {output_base}, {2**depth} colors\n")
        f.write("#include <exec/types.h>\n\n")
        f.write(f"UWORD {output_base.split('/')[-1]}_pal[] = {{\n")
        
        for i in range(2**depth):
            if i*3+2 < len(palette):
                r = palette[i*3]
                g = palette[i*3+1]
                b = palette[i*3+2]
                # Scale 8-bit to 4-bit (Amiga OCS/ECS)
                # 0-255 -> 0-15
                ar = r >> 4
                ag = g >> 4
                ab = b >> 4
                val = (ar << 8) | (ag << 4) | ab
                f.write(f"    0x{val:04X},")
                if (i+1) % 8 == 0:
                    f.write("\n")
        f.write("};\n")

    print(f"Palette saved to {output_base}_pal.h")

    # Export Bitplanes (Interleaved)
    # Interleaved means: Line 0 Plane 0, Line 0 Plane 1, ... Line 0 Plane N, Line 1 Plane 0 ...
    
    # Pad width to 16 pixels (word aligned)
    stride = (width + 15) // 16 * 2 # bytes per plane line
    
    raw_data = bytearray()
    
    pixels = img.load()
    
    for y in range(height):
        for plane in range(depth):
            # Construct line for this plane
            line_bytes = bytearray(stride)
            for x in range(width):
                color_index = pixels[x, y]
                if (color_index >> plane) & 1:
                    byte_index = x // 8
                    bit_index = 7 - (x % 8)
                    line_bytes[byte_index] |= (1 << bit_index)
            raw_data.extend(line_bytes)

    with open(f"{output_base}.raw", "wb") as f:
        f.write(raw_data)

    # Also export as C header for inclusion
    with open(f"{output_base}_data.h", "w") as f:
        f.write(f"// Raw Bitplane Data for {output_base}, {width}x{height}, {depth} planes\n")
        f.write(f"#include <exec/types.h>\n\n")
        f.write(f"__attribute__((section(\".chip\"))) UBYTE {output_base.split('/')[-1]}_data[] = {{\n")
        for i, b in enumerate(raw_data):
            f.write(f"0x{b:02X},")
            if (i+1) % 16 == 0:
                f.write("\n")
        f.write("};\n")
        
    print(f"Raw data saved to {output_base}.raw and {output_base}_data.h")
    print(f"Dimensions: {width}x{height} pixels, {depth} bitplanes")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_assets.py <input.png> <output_base> [depth]")
    else:
        depth = 5
        if len(sys.argv) > 3:
            depth = int(sys.argv[3])
        export_amiga_interleaved(sys.argv[1], sys.argv[2], depth)
