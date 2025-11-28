import os
import sys
import numpy as np
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory to path to access Example module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import *

from Example.Example import params_init, data_generation
import config

# Import specific functions from ODE Parser
args = config.parse_args()
torch.manual_seed(args.SEED)
np.random.seed(args.SEED)

# Set device
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    device = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU instead")

#===========================Path part==============================================
print("\n"+ "="*60)
print("\n[INFO] Setting up the path...")

# Check if CUDA is available and set base path accordingly
# Use args.model if available, otherwise fall back to args.params_name
model_name = getattr(args, 'model', None) or getattr(args, 'params_name', 'default')
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    base_path = os.path.join(config.DIR_PROJECT, 'Results', 'gpu_folder', model_name)
    print(f"Using GPU folder: {base_path}")
else:
    base_path = os.path.join(config.DIR_PROJECT, 'Results', 'cpu_folder', model_name)
    print(f"Using CPU folder: {base_path}")

# Set up model path and save directory
model_PATH = Path(base_path)
save_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.TRAIN_SIZE}')
os.makedirs(save_dir, exist_ok=True)
print(f'[INFO] The save directory is set up successfully: {save_dir}')
print("="*60)
#=================================================================================
# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Train to learn stochastic part in noise level and num samples")
print("2. Skip Training and generate the prediction results")

print("="*60)

# choice = '1'
while True:
# choice = '1' #
    choice = input("\nChoose option (1 or 2 ):").strip()
    if choice in ['1','2']:
        break
    else:
        print("Please enter '1' or '2'.")

if choice == '1':
    # Option 1: Train everything in second stage
    print("\n[INFO] Training everything in second stage...")
    
    # Add noise level selection
    print("\n" + "="*60)
    print(f"NOISE LEVEL SELECTION for {args.NOISE_LEVEL}")
    print("="*60)
   
    # Set up save directory for second stage
    second_stage_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.TRAIN_SIZE}')
    os.makedirs(second_stage_dir, exist_ok=True)
    print(f'[INFO] Using second stage directory: {second_stage_dir}')
   
    # Check for data file - should be in the noise level directory
    data_file_path = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'simulation_results_noise_{args.NOISE_LEVEL}.npz')
    if not os.path.exists(data_file_path):
        raise RuntimeError(f'[ERROR] Data file not found: {data_file_path}. You should run 1stage_deterministic.py first')
    else:
        print(f'[SUCCESS] Data file found: {data_file_path}. Now you can train the following.')
    
    data = np.load(data_file_path)
    current_state_full = data['current_state']
    next_state_full = data['next_state']
    dimension = current_state_full.shape[1]
    scaler = np.ones(dimension)*args.DIFF_SCALE
    train_size = args.TRAIN_SIZE

    current_state_train_np = current_state_full[:train_size]
    next_state_train_np = next_state_full[:train_size]
    
    current_state_train = torch.from_numpy(current_state_train_np).float().to(device)
    next_state_train = torch.from_numpy(next_state_train_np).float().to(device)
    
    dt = params_init(case_name=model_name)['Dt']  # Get dt from params
    
    
    def learned_model_wrapper(x):
        """Wrapper function for the learned FEX model."""
        return FEX_model_learned(x, 
                                 model_name=model_name,
                                 params_name=model_name,
                                 noise_level=args.NOISE_LEVEL,
                                 device=str(device),
                                 base_path=base_path)
    
    def learned_model_with_force_wrapper(x):
        """Wrapper function for the learned FEX model with force term."""
        # FEX_with_force is not implemented yet, use regular model
        return learned_model_wrapper(x)
    
    if dimension == 1:
        residual = generate_euler_maruyama_residue(func=learned_model_wrapper, current_state=current_state_train, next_state=next_state_train, dt=dt)
        print(f'[INFO] Residual shape: {residual.shape}')
        # For 1D case, convert residual to numpy if needed
        if isinstance(residual, torch.Tensor):
            residual_np = residual.cpu().numpy()
        else:
            residual_np = residual
        # Reshape to 3D format (size, dim, time_steps) for generate_second_step
        # For 1D case: residual is (N, 1) or (N,), need to make it (N, 1, 1)
        if residual_np.ndim == 1:
            residual_np = residual_np[:, np.newaxis]  # (N,) -> (N, 1)
        # Add time dimension: (N, 1) -> (N, 1, 1)
        residuals_for_step = residual_np[:, :, np.newaxis]
        print(f'[INFO] Reshaped residuals for generate_second_step: {residuals_for_step.shape}')
    else:
        residuals, u_current_reshaped, residual_cov_time = generate_euler_maruyama_residue(func=learned_model_wrapper, current_state=current_state_train, next_state=next_state_train, dt=dt)
        print(f'[INFO] Residuals shape: {residuals.shape}')
        # For multi-D case, use residuals directly (already 3D: MC_samples, dim, time_steps)
        residuals_for_step = residuals

 #===================================================================================
    if not os.path.exists(os.path.join(second_stage_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(second_stage_dir,'ZT_Solution.npy')):
        ODE_Solution,ZT_Solution = generate_second_step(
            current_state_train, residuals_for_step, scaler, dt, train_size, device,
            num_time_points=101, time_dependent=False  # Only process 100 time points
        )
        print(f'[INFO] the ODE solution shape is: {ODE_Solution.shape}')
        mean_value, std_value = generate_mean_and_std(ODE_Solution)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        np.save(os.path.join(second_stage_dir, "ODE_Solution.npy"), ODE_Solution)
        np.save(os.path.join(second_stage_dir, "ZT_Solution.npy"), ZT_Solution)
    else:
        print('[INFO] the ODE solution has already been generated, skip the generation process.')
        ODE_Solution = np.load(os.path.join(second_stage_dir, "ODE_Solution.npy"))
        mean_value, std_value = generate_mean_and_std(ODE_Solution)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        ZT_Solution = np.load(os.path.join(second_stage_dir, "ZT_Solution.npy"))
