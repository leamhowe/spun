import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- CONFIGURATION ---
INPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/single_chip_exports/20210621_NO0099'
OUTPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/diagrams'

# Standard sequential order from top to bottom
BANDS_ORDER = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

def create_vertical_column():
    print("Generating Vertical Column of 13 Bands...")

    # Create a figure with 13 rows and 1 column. 
    # figsize=(width, height) - narrow and very tall!
    fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(3, 26))

    for i, band in enumerate(BANDS_ORDER):
        ax = axes[i]
        img_path = os.path.join(INPUT_DIR, f"{band}.png")
        
        try:
            img = mpimg.imread(img_path) 
            ax.imshow(img, cmap='gray')
        except FileNotFoundError:
            print(f"  -> Error: Could not find {img_path}")
            # If a file is missing, it skips rendering it but keeps the space
            continue

        # Add the band label to the right side of the image
        label_color = 'darkred' if band == 'B4' else 'black'
        ax.text(1.1, 0.5, band, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='center', ha='left', color=label_color)

        # Draw a solid black border around the image
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
            spine.set_visible(True)

        # Remove the tick marks/numbers on the axes
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust vertical spacing between images (increase hspace for a bigger gap)
    plt.subplots_adjust(hspace=0.15) 

    # Save the output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = os.path.join(OUTPUT_DIR, "13_bands_vertical_column.png")
    
    # bbox_inches='tight' ensures the text on the right doesn't get cut off
    plt.savefig(out_name, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"Success! Saved vertical column to {out_name}")

if __name__ == "__main__":
    create_vertical_column()