import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- CONFIGURATION ---
INPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/single_chip_exports/20210621_NO0099'
OUTPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/diagrams'

# B4 is placed at the end so it renders last (on top/in front)
RENDERING_ORDER = ['B12', 'B11', 'B10', 'B9', 'B8A', 'B8', 'B7', 'B6', 'B5', 'B3', 'B2', 'B1', 'B4']

def generate_flat_style(style_name, x_step, y_step, figsize=(15, 6)):
    """Generates and saves a clean, 2D flat layout of the image stack."""
    print(f"Generating {style_name}...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Track the maximum dimensions to adjust the plot boundaries later
    max_x = 0
    max_y = 0

    for i, band in enumerate(RENDERING_ORDER):
        img_path = os.path.join(INPUT_DIR, f"{band}.png")
        try:
            img = mpimg.imread(img_path) 
        except FileNotFoundError:
            print(f"  -> Error: Could not find {img_path}")
            continue

        # Calculate where the bottom-left corner of this image should be
        x_offset = i * x_step
        y_offset = i * y_step

        # Define the exact bounding box for this image [left, right, bottom, top]
        extent = [x_offset, x_offset + 128, y_offset, y_offset + 128]
        
        # Update maximums for the plot boundaries
        if extent[1] > max_x: max_x = extent[1]
        if extent[3] > max_y: max_y = extent[3]

        # Draw the flat 2D image
        ax.imshow(img, cmap='gray', origin='lower', extent=extent, alpha=0.95)

        # Draw a clean black border around each image
        rect = plt.Rectangle((x_offset, y_offset), 128, 128, fill=False, 
                             edgecolor='black', linewidth=1.0)
        ax.add_patch(rect)

        # Label the images (Adjusting text position based on layout)
        if style_name == "1_The_Line":
            # For a pure line, label every image at the top
            label_text = 'B4 (Red)' if band == 'B4' else band
            color = 'darkred' if band == 'B4' else 'black'
            ax.text(x_offset + 64, y_offset + 135, label_text, 
                    fontsize=12, fontweight='bold', color=color, ha='center')
        else:
            # For overlapping stacks, only label the back and front
            if i == 0:
                ax.text(x_offset - 10, y_offset + 135, RENDERING_ORDER[0], 
                        fontsize=12, fontweight='bold')
            elif i == 12: 
                ax.text(x_offset - 10, y_offset + 135, 'B4 (Red)', 
                        fontsize=14, fontweight='bold', color='darkred')

    # Add a slight margin to the plot boundaries
    ax.set_xlim(-20, max_x + 20)
    ax.set_ylim(-20, max_y + 30)
    
    ax.axis('off')

    # Save logic
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = os.path.join(OUTPUT_DIR, f"stack_style_{style_name}.png")
    
    plt.tight_layout()
    plt.savefig(out_name, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved to {out_name}\n")


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Directory not found: {INPUT_DIR}")
        return

    # --- STYLE OPTION 1: The Line ---
    # Pure side-by-side horizontal row. No overlapping at all.
    # We step the X-axis by exactly the width of the image (128) plus a small gap (10).
    generate_flat_style(
        style_name="1_The_Line",
        x_step=138, y_step=0, 
        figsize=(24, 4) # Very wide figure
    )

    # --- STYLE OPTION 2: Flat Horizontal Spread ---
    # Like a hand of cards fanned out sideways. Overlapping, but in a straight horizontal line.
    generate_flat_style(
        style_name="2_Horizontal_Spread",
        x_step=25, y_step=0, 
        figsize=(12, 4)
    )

    # --- STYLE OPTION 3: Flat Diagonal Stack ---
    # Steps up and to the right uniformly. Classic clean overlap.
    generate_flat_style(
        style_name="3_Diagonal_Stack",
        x_step=15, y_step=15, 
        figsize=(10, 8)
    )

if __name__ == "__main__":
    main()