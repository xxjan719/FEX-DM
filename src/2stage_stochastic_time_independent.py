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

    # Try to load training data from npz file, otherwise use first train_size samples
    train_data_path = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'train_data_{args.TRAIN_SIZE}.npz')
    if os.path.exists(train_data_path):
        print(f'[INFO] Loading training data from: {train_data_path}')
        train_data = np.load(train_data_path)
        current_state_train_np = train_data['current_state']
        next_state_train_np = train_data['next_state']
        print(f'[INFO] Loaded training data shapes: current_state={current_state_train_np.shape}, next_state={next_state_train_np.shape}')
    else:
        print(f'[INFO] Training data file not found at {train_data_path}, using first {train_size} samples')
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
        residual = generate_euler_maruyama_residue(func=learned_model_wrapper, current_state=current_state_full, next_state=next_state_full, dt=dt)
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
        residuals_for_step = residual_np

    else:
        residuals, u_current_reshaped, residual_cov_time = generate_euler_maruyama_residue(func=learned_model_wrapper, current_state=current_state_full, next_state=next_state_full, dt=dt)
        print(f'[INFO] Residuals shape: {residuals.shape}')
        # For multi-D case, use residuals directly (already 3D: MC_samples, dim, time_steps)
        residuals_for_step = residuals

 #===================================================================================
    if not os.path.exists(os.path.join(second_stage_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(second_stage_dir,'ZT_Solution.npy')):
        ODE_Solution,ZT_Solution = generate_second_step(
            current_state_full, residuals_for_step, scaler, dt, train_size, device,
            num_time_points=101, time_dependent=False,  # Only process 100 time points
            current_state_train=current_state_train_np  # Pass training data from npz
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
    
    # Save training data files
    print("[INFO] Checking training data files...")
    
    data_inf_path = os.path.join(second_stage_dir, 'data_inf.pt')
    if not os.path.exists(data_inf_path):
        save_parameters(ZT_Solution, ODE_Solution, second_stage_dir, args, device)
        # Load the saved data after saving
        data_inf = torch.load(data_inf_path)
    else:
        print('[INFO] the data_inf.pt file has already been generated, skip the generation process.')
        data_inf = torch.load(data_inf_path)
    
    # Load variables from data_inf (works for both branches)
    ZT_Train_new = data_inf['ZT_Train_new']
    ODE_Train_new = data_inf['ODE_Train_new']
    ZT_Train_mean = data_inf['ZT_Train_mean']
    ZT_Train_std = data_inf['ZT_Train_std']
    ODE_Train_mean = data_inf['ODE_Train_mean']
    ODE_Train_std = data_inf['ODE_Train_std']
    
    # Split into train and validation sets
    NTrain = int(ZT_Train_new.shape[0] * 0.8)
    NValid = int(ZT_Train_new.shape[0] * 0.2)
    
    ZT_Train_new_normal = ZT_Train_new[NTrain:]
    ODE_Train_new_normal = ODE_Train_new[NTrain:]
    
    ZT_Train_new_valid = ZT_Train_new[NValid:]
    ODE_Train_new_valid = ODE_Train_new[NValid:]
    print(f'[INFO] the ZT_Train_new_normal shape is: {ZT_Train_new_normal.shape}')
    print(f'[INFO] the ODE_Train_new_normal shape is: {ODE_Train_new_normal.shape}')
    print(f'[INFO] the ZT_Train_new_valid shape is: {ZT_Train_new_valid.shape}')
    print(f'[INFO] the ODE_Train_new_valid shape is: {ODE_Train_new_valid.shape}')
    if not os.path.exists(os.path.join(second_stage_dir,'FNET.pth')):
        print('[INFO] the FNET.pth file has not been generated, start the training process.')
        FNET = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
        FNET_optim = torch.optim.Adam(FNET.parameters(), lr=args.NN_SOLVER_LR, weight_decay=1e-6)
        FNET.zero_grad()
        criterion = torch.nn.MSELoss()
        n_iteration = args.NN_SOLVER_EPOCHS
        best_valid_err = 5.0  # Initialize best validation error

        for epoch in range(n_iteration):
            FNET_optim.zero_grad()
            pred = FNET(ZT_Train_new_normal.reshape((ZT_Train_new_normal.shape[0],1)))
            loss = criterion(pred,ODE_Train_new_normal)
            loss.backward()
            FNET_optim.step()
        
            # Compute validation loss
            pred_valid = FNET(ZT_Train_new_valid.reshape((ZT_Train_new_valid.shape[0],1)))
            loss_valid = criterion(pred_valid,ODE_Train_new_valid)
            if loss_valid < best_valid_err:
                FNET.update_best()
                best_valid_err = loss_valid
        
            if epoch % 100 == 0:
                print(f'epoch is {epoch+1}; loss is {loss}; valid loss is {loss_valid}')
        FNET.final_update()
        FNET_path = os.path.join(second_stage_dir,'FNET.pth')
        torch.save(FNET.state_dict(),FNET_path)
    else:
        print('[INFO] the FNET.pth file has already been generated, skip the training process.')
    print("\n")
    print('[SUCCESS] training process finished.')
    print("="*60)
    print("The choice 1 is finished. You may need to run the choice 2 to get the prediction results.")
    print("="*60)
    exit()
elif choice == '2':
    pass