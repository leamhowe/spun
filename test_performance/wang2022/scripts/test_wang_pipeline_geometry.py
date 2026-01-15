import numpy as np
import matplotlib.pyplot as plt
import torch

# --- CONFIGURATION ---
INPUT_SIZE = 256
ORIG_SIZE = 100

def test_pipeline():
    # 1. Create Synthetic Input (100x100)
    # Background 0, distinct pixels at corners = 1.0
    fake_input = np.zeros((4, 100, 100), dtype=np.float32)
    
    # Mark corners
    fake_input[:, 0, 0] = 1.0   # Top-Left
    fake_input[:, 0, 99] = 1.0  # Top-Right
    fake_input[:, 99, 0] = 1.0  # Bottom-Left
    fake_input[:, 99, 99] = 1.0 # Bottom-Right
    
    # Mark Center
    fake_input[:, 50, 50] = 1.0

    # 2. Simulate "Preprocessing" (Padding to 256)
    # The logic from generate_wang_predictions.py:
    padded = np.zeros((4, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    padded[:, :ORIG_SIZE, :ORIG_SIZE] = fake_input # Placed in Top-Left

    # 3. Simulate "Model" (Pass-through)
    # We assume the U-Net preserves spatial dimensions exactly (256 in -> 256 out)
    # We return the first channel as the "prediction"
    model_output = torch.from_numpy(padded).unsqueeze(0) # (1, 4, 256, 256)
    
    # 4. Simulate "Post-processing" (Cropping back to 100)
    out_numpy = model_output.cpu().numpy()
    pred_crop = out_numpy[0, :, :ORIG_SIZE, :ORIG_SIZE]
    
    # Take channel 0 as "result"
    result = pred_crop[0]
    
    # 5. Simulate "Saving" (The Vertical Flip)
    # In the main script: wang_scf[::-1, :]
    final_result = result[::-1, :]

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(fake_input[0], cmap='gray')
    ax[0].set_title("1. Original Input (100x100)\nTop-Left is (0,0)")
    
    ax[1].imshow(padded[0], cmap='gray')
    ax[1].set_title("2. Padded Input (256x256)\nShould be in Top-Left corner")
    
    ax[2].imshow(final_result, cmap='gray')
    ax[2].set_title("3. Final Result (Flipped)\nTop-Left should be Bottom-Left?")
    
    plt.tight_layout()
    plt.savefig("pipeline_geometry_test.png")
    print("Test image saved to pipeline_geometry_test.png")
    
    # --- VERIFICATION ---
    # We flipped the result, so:
    # Original (0,0) [Top-Left] -> Should move to (99,0) [Bottom-Left] in image coords
    
    print("\n--- COORDINATE CHECK ---")
    print(f"Original Top-Left (0,0) value: {fake_input[0,0,0]}")
    print(f"Final Bottom-Left (99,0) value: {final_result[99,0]}")
    
    if final_result[99,0] == 1.0:
        print("PASS: Top-Left pixel correctly moved to Bottom-Left after flip.")
    else:
        print("FAIL: Geometry lost!")

test_pipeline()