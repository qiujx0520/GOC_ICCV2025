import torch
import os

# Define directory paths
block_pth_dir = '~/GOC/DiT/block_pth'
output_dir = '~/GOC/DiT/averaged_pth'
# Ensure the save directory exists
os.makedirs(output_dir, exist_ok=True)  

# Get the paths of all .pth files
pth_files = [os.path.join(block_pth_dir, f) for f in os.listdir(block_pth_dir) if f.endswith('.pth')]

# Check if there are any files
if not pth_files:
    raise FileNotFoundError(f"No .pth files found in {block_pth_dir}")

# Load the first file to get the number of blocks
sample_outputs = torch.load(pth_files[0])
# Get the number of blocks
num_blocks = len(sample_outputs)  
# Assume each block has the same number of steps
step_count = len(sample_outputs[0])  

# Initialize a dictionary to store the average results
averaged_block_outputs = {block_idx: [] for block_idx in range(num_blocks)}

# Iterate through each block
for block_idx in range(num_blocks):
    # Iterate through each step
    for step_idx in range(step_count):
        print(block_idx, step_idx)
        # Used to store the step outputs of all .pth files
        step_outputs = []  
        
        # Iterate through each file and collect the step outputs at the same position
        for pth_file in pth_files:
            block_outputs = torch.load(pth_file)
            # Assume we take the first tensor
            step_output = block_outputs[block_idx][step_idx][0]  
            step_outputs.append(step_output)
        
        # Calculate the average of the outputs of all files at this step
        avg_output = torch.mean(torch.stack(step_outputs), dim=0)
        
        # Save the calculated average to the new block_outputs
        averaged_block_outputs[block_idx].append([avg_output])

# Save the averaged results as a new .pth file
average_save_path = os.path.join(output_dir, 'averaged_block_outputs.pth')
torch.save(averaged_block_outputs, average_save_path)
print(f"Averaged block outputs have been saved to: {average_save_path}")