import os
import argparse
import subprocess
import glob

def main(args):
    """
    Finds all GeoTIFFs in the input directory and runs generate_spun_scene_predictions.py on each.
    """
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.tif')))
    
    if not input_files:
        print(f"Error: No .tif files found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Found {len(input_files)} files to process.")

    # Dynamically find the core prediction script in the same directory
    script_path = os.path.join(os.path.dirname(__file__), 'generate_spun_scene_predictions.py')

    for input_filepath in input_files:
        base_filename = os.path.basename(input_filepath)
        output_filename = f"pred_{base_filename}"
        output_filepath = os.path.join(args.output_dir, output_filename)

        print("-" * 50)
        print(f"Processing: {base_filename}")
        print(f"Output will be saved to: {output_filepath}")

        # Construct the command to call the prediction script
        command = [
            'python',
            script_path,
            '--input_geotiff', input_filepath,
            '--output_geotiff', output_filepath,
            '--model_path', args.model_path,
            '--credentials_path', args.credentials_path,
            '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers)
        ]

        if args.no_cloud_mask:
            command.append('--no_cloud_mask')


        try:
            # Execute the command
            subprocess.run(command, check=True, text=True)
            print(f"Successfully processed {base_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {base_filename}:")
            print(e)
            print("Continuing to the next file...")
        except FileNotFoundError:
            print(f"Error: Could not find the script at {script_path}. Make sure it's in the same directory.")
            break

    print("-" * 50)
    print("Batch processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch run snow cover prediction on a directory of GeoTIFFs.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input GeoTIFFs to define ROIs and dates.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output prediction GeoTIFFs.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pt file.')
    parser.add_argument('--credentials_path', type=str, required=True, help='Path to your GEE service account credentials JSON file.')
    
    parser.add_argument('--no_cloud_mask', action='store_true', help='Use this flag to DISABLE cloud masking. (Masking is enabled by default)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for GPU inference.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers for data loading.')

    args = parser.parse_args()
    main(args)