import os
import sys
import numpy as np
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory to path to access Example module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from utils import *

# Import from Example module - import the file directly
import importlib.util
example_file = os.path.join(project_root, 'Example', 'Example.py')
spec = importlib.util.spec_from_file_location("Example", example_file)
Example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Example)
params_init = Example.params_init
data_generation = Example.data_generation
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


# Set up save directory for second stage
second_stage_FEX_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.TRAIN_SIZE}')
All_stage_TF_CDM_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'All_stage_TF_CDM_{args.TRAIN_SIZE}')
os.makedirs(second_stage_FEX_dir, exist_ok=True)
os.makedirs(All_stage_TF_CDM_dir, exist_ok=True)
print(f'[INFO] Using second stage directory: {second_stage_FEX_dir}')
print(f'[INFO] Using All stage TF-CDM directory: {All_stage_TF_CDM_dir}')

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
    scaler_TF_CDM = np.ones(dimension)*10.0
    
    def learned_model_wrapper(x):
        """Wrapper function for the learned FEX model."""
        return FEX_model_learned(x, 
                                 model_name=model_name,
                                 noise_level=args.NOISE_LEVEL,
                                 device=str(device),
                                 base_path=base_path)
    
    def learned_model_with_force_wrapper(x):
        """Wrapper function for the learned FEX model with force term."""
        # FEX_with_force is not implemented yet, use regular model
        return learned_model_wrapper(x)
    
    if dimension == 1:
        residual_FEX = generate_euler_maruyama_residue(func=learned_model_wrapper, current_state=current_state_full, next_state=next_state_full, dt=dt)
        print(f'[INFO] Residual shape: {residual_FEX.shape}')
        # For 1D case, convert residual to numpy if needed
        if isinstance(residual_FEX, torch.Tensor):
            residual_FEX_np = residual_FEX.cpu().numpy()
        else:
            residual_FEX_np = residual_FEX
        # Reshape to 3D format (size, dim, time_steps) for generate_second_step
        # For 1D case: residual is (N, 1) or (N,), need to make it (N, 1, 1)
        if residual_FEX_np.ndim == 1:
            residual_FEX_np = residual_FEX_np[:, np.newaxis]  # (N,) -> (N, 1)
        residuals_FEX_for_step = residual_FEX_np
        residuals_TF_CDM_for_step = (next_state_full - current_state_full)

    else:
        residuals_FEX, u_current_reshaped, residual_cov_time = generate_euler_maruyama_residue(func=learned_model_wrapper, current_state=current_state_full, next_state=next_state_full, dt=dt)
        print(f'[INFO] Residuals shape: {residuals_FEX.shape}')
        # For multi-D case, use residuals directly (already 3D: MC_samples, dim, time_steps)
        residuals_FEX_for_step = residuals_FEX
        
        residuals_TF_CDM_for_step = (next_state_full - current_state_full)
    
 #===================================================================================
    # Compute short_indx once and reuse for both FEX-DM and TF-CDM
    # Since both use the same current_state_full and current_state_train_np, short_indx will be identical
    short_indx_shared = None
    need_short_indx = (not os.path.exists(os.path.join(second_stage_FEX_dir,'ODE_Solution.npy')) or 
                      not os.path.exists(os.path.join(All_stage_TF_CDM_dir,'ODE_Solution.npy')))
    
    if need_short_indx:
        print('[INFO] Computing shared short_indx for both FEX-DM and TF-CDM...')
        from utils.ODEParser import process_chunk, process_chunk_faiss_cpu
        short_size = 2048
        it_size_x0train = train_size
        it_n_index = train_size // it_size_x0train
        
        if torch.cuda.is_available():
            short_indx_shared = process_chunk(it_n_index, it_size_x0train, short_size, 
                                             current_state_full, current_state_train_np, 
                                             train_size, current_state_full.shape[1])
        else:
            short_indx_shared = process_chunk_faiss_cpu(it_n_index, it_size_x0train, short_size, 
                                                        current_state_full, current_state_train_np, 
                                                        train_size, current_state_full.shape[1])
        print(f'[INFO] Shared short_indx computed, shape: {short_indx_shared.shape}')
    
    # check FEX ODE solution and ZT solution
    if not os.path.exists(os.path.join(second_stage_FEX_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(second_stage_FEX_dir,'ZT_Solution.npy')):
        ODE_Solution_FEX,ZT_Solution_FEX = generate_second_step(
            current_state_full, residuals_FEX_for_step, scaler, dt, train_size, device,
            num_time_points=101, time_dependent=False,  # Only process 100 time points
            current_state_train=current_state_train_np,  # Pass training data from npz
            short_indx=short_indx_shared  # Use shared short_indx
        )
        print(f'[INFO] the ODE solution shape is: {ODE_Solution_FEX.shape}')
        mean_value, std_value = generate_mean_and_std(ODE_Solution_FEX)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        np.save(os.path.join(second_stage_FEX_dir, "ODE_Solution.npy"), ODE_Solution_FEX)
        np.save(os.path.join(second_stage_FEX_dir, "ZT_Solution.npy"), ZT_Solution_FEX)
    else:
        print('[INFO] the ODE solution has already been generated, skip the generation process.')
        ODE_Solution_FEX = np.load(os.path.join(second_stage_FEX_dir, "ODE_Solution.npy"))
        mean_value, std_value = generate_mean_and_std(ODE_Solution_FEX)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        ZT_Solution_FEX = np.load(os.path.join(second_stage_FEX_dir, "ZT_Solution.npy"))
    
    # check TF-CDM ODE solution and ZT solution
    if not os.path.exists(os.path.join(All_stage_TF_CDM_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(All_stage_TF_CDM_dir,'ZT_Solution.npy')):
        ODE_Solution_TF_CDM,ZT_Solution_TF_CDM = generate_second_step(
            current_state_full, residuals_TF_CDM_for_step, scaler_TF_CDM, dt, train_size, device,
            num_time_points=101, time_dependent=False,  # Only process 100 time points
            current_state_train=current_state_train_np,  # Pass training data from npz
            short_indx=short_indx_shared  # Use shared short_indx
        )
        print(f'[INFO] the ODE solution shape is: {ODE_Solution_TF_CDM.shape}')
        mean_value, std_value = generate_mean_and_std(ODE_Solution_TF_CDM)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        np.save(os.path.join(All_stage_TF_CDM_dir, "ODE_Solution.npy"), ODE_Solution_TF_CDM)
        np.save(os.path.join(All_stage_TF_CDM_dir, "ZT_Solution.npy"), ZT_Solution_TF_CDM)
    else:
        print('[INFO] the ODE solution has already been generated, skip the generation process.')
        ODE_Solution_TF_CDM = np.load(os.path.join(All_stage_TF_CDM_dir, "ODE_Solution.npy"))
        mean_value, std_value = generate_mean_and_std(ODE_Solution_TF_CDM)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        ZT_Solution_TF_CDM = np.load(os.path.join(All_stage_TF_CDM_dir, "ZT_Solution.npy"))




    # Save training data files
    print("[INFO] Checking training data files...")
    
    data_inf_path_FEX = os.path.join(second_stage_FEX_dir, 'data_inf.pt')
    if not os.path.exists(data_inf_path_FEX):
        save_parameters(ZT_Solution_FEX, ODE_Solution_FEX, second_stage_FEX_dir, args, device)
        # Load the saved data after saving
        data_inf_FEX = torch.load(data_inf_path_FEX)
    else:
        print('[INFO] the data_inf.pt file has already been generated, skip the generation process.')
        data_inf_FEX = torch.load(data_inf_path_FEX)
    

    data_inf_path_TF_CDM = os.path.join(All_stage_TF_CDM_dir, 'data_inf.pt')
    ZT_Solution_TF_CDM = np.hstack((current_state_train, ZT_Solution_TF_CDM))
    if not os.path.exists(data_inf_path_TF_CDM):
        save_parameters(ZT_Solution_TF_CDM, ODE_Solution_TF_CDM, All_stage_TF_CDM_dir, args, device)
        # Load the saved data after saving
        data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM)
    else:
        print('[INFO] the data_inf.pt file has already been generated, skip the generation process.')
        data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM)




    # Load variables from data_inf (works for both branches)
    ZT_Train_new_FEX = data_inf_FEX['ZT_Train_new']
    ODE_Train_new_FEX = data_inf_FEX['ODE_Train_new']
    ZT_Train_mean_FEX = data_inf_FEX['ZT_Train_mean']
    ZT_Train_std_FEX = data_inf_FEX['ZT_Train_std']
    ODE_Train_mean_FEX = data_inf_FEX['ODE_Train_mean']
    ODE_Train_std_FEX = data_inf_FEX['ODE_Train_std']

    ZT_Train_new_TF_CDM = data_inf_TF_CDM['ZT_Train_new']
    ODE_Train_new_TF_CDM = data_inf_TF_CDM['ODE_Train_new']
    ZT_Train_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean']
    ZT_Train_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std']
    ODE_Train_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean']
    ODE_Train_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std']
    
    # Split into train and validation sets
    NTrain_FEX = int(ZT_Train_new_FEX.shape[0] * 0.8)
    NValid_FEX = int(ZT_Train_new_FEX.shape[0] * 0.2)
    
    NTrain_TF_CDM = int(ZT_Train_new_TF_CDM.shape[0] * 0.8)
    NValid_TF_CDM = int(ZT_Train_new_TF_CDM.shape[0] * 0.2)

    ZT_Train_new_normal_FEX = ZT_Train_new_FEX[NTrain_FEX:]
    ODE_Train_new_normal_FEX = ODE_Train_new_FEX[NTrain_FEX:]
    
    ZT_Train_new_valid_FEX = ZT_Train_new_FEX[NValid_FEX:]
    ODE_Train_new_valid_FEX = ODE_Train_new_FEX[NValid_FEX:]
    print(f'[INFO] the ZT_Train_new_normal_FEX shape is: {ZT_Train_new_normal_FEX.shape}')
    print(f'[INFO] the ODE_Train_new_normal_FEX shape is: {ODE_Train_new_normal_FEX.shape}')
    print(f'[INFO] the ZT_Train_new_valid_FEX shape is: {ZT_Train_new_valid_FEX.shape}')
    print(f'[INFO] the ODE_Train_new_valid_FEX shape is: {ODE_Train_new_valid_FEX.shape}')
    
    ZT_Train_new_normal_TF_CDM = ZT_Train_new_TF_CDM[NTrain_TF_CDM:]
    ODE_Train_new_normal_TF_CDM = ODE_Train_new_TF_CDM[NTrain_TF_CDM:]
    
    ZT_Train_new_valid_TF_CDM = ZT_Train_new_TF_CDM[NValid_TF_CDM:]
    ODE_Train_new_valid_TF_CDM = ODE_Train_new_TF_CDM[NValid_TF_CDM:]
    print(f'[INFO] the ZT_Train_new_normal_TF_CDM shape is: {ZT_Train_new_normal_TF_CDM.shape}')
    print(f'[INFO] the ODE_Train_new_normal_TF_CDM shape is: {ODE_Train_new_normal_TF_CDM.shape}')
    print(f'[INFO] the ZT_Train_new_valid_TF_CDM shape is: {ZT_Train_new_valid_TF_CDM.shape}')
    print(f'[INFO] the ODE_Train_new_valid_TF_CDM shape is: {ODE_Train_new_valid_TF_CDM.shape}')
    if not os.path.exists(os.path.join(second_stage_FEX_dir,'FNET.pth')) or not os.path.exists(os.path.join(All_stage_TF_CDM_dir,'FNET.pth')):
        print('[INFO] the FNET.pth file has not been generated, start the training process.')
        FNET_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
        FNET_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
        FNET_optim_FEX = torch.optim.Adam(FNET_FEX.parameters(), lr=args.NN_SOLVER_LR, weight_decay=1e-6)
        FNET_optim_TF_CDM = torch.optim.Adam(FNET_TF_CDM.parameters(), lr=args.NN_SOLVER_LR, weight_decay=1e-6)
        FNET_FEX.zero_grad()
        FNET_TF_CDM.zero_grad()
        criterion = torch.nn.MSELoss()
        criterion_TF_CDM = torch.nn.MSELoss()
        n_iteration = args.NN_SOLVER_EPOCHS
        best_valid_err = 5.0  # Initialize best validation error
        
        # Print table header
        print("\n" + "="*80)
        print(f"{'Model':<10} {'Epoch':<8} {'Train Loss':<15} {'Valid Loss':<15}")
        print("="*80)

        for epoch in range(n_iteration):
            FNET_optim_FEX.zero_grad()
            pred_FEX = FNET_FEX(ZT_Train_new_normal_FEX.reshape((ZT_Train_new_normal_FEX.shape[0],1)))
            loss_FEX = criterion(pred_FEX,ODE_Train_new_normal_FEX)
            loss_FEX.backward()
            FNET_optim_FEX.step()
            
            FNET_optim_TF_CDM.zero_grad()
            pred_TF_CDM = FNET_TF_CDM(ZT_Train_new_normal_TF_CDM)
            loss_TF_CDM = criterion_TF_CDM(pred_TF_CDM,ODE_Train_new_normal_TF_CDM)
            loss_TF_CDM.backward()
            FNET_optim_TF_CDM.step()
             
            # Compute validation loss
            pred_valid_FEX = FNET_FEX(ZT_Train_new_valid_FEX.reshape((ZT_Train_new_valid_FEX.shape[0],1)))
            loss_valid_FEX = criterion(pred_valid_FEX,ODE_Train_new_valid_FEX)
            
            pred_valid_TF_CDM = FNET_TF_CDM(ZT_Train_new_valid_TF_CDM)
            loss_valid_TF_CDM = criterion_TF_CDM(pred_valid_TF_CDM,ODE_Train_new_valid_TF_CDM)
            

            if loss_valid_FEX < best_valid_err:
                FNET_FEX.update_best()
                best_valid_err = loss_valid_FEX
            if loss_valid_TF_CDM < best_valid_err:
                FNET_TF_CDM.update_best()
                best_valid_err = loss_valid_TF_CDM
        
            if epoch % 100 == 0:
                print(f"{'FEX-DM':<10} {epoch+1:<8} {loss_FEX.item():<15.6f} {loss_valid_FEX.item():<15.6f}")
                print(f"{'TF-CDM':<10} {epoch+1:<8} {loss_TF_CDM.item():<15.6f} {loss_valid_TF_CDM.item():<15.6f}")
                print("-"*80)
        FNET_FEX.final_update()
        FNET_TF_CDM.final_update()
        FNET_path_FEX = os.path.join(second_stage_FEX_dir,'FNET.pth')
        torch.save(FNET_FEX.state_dict(),FNET_path_FEX)
        FNET_path_TF_CDM = os.path.join(All_stage_TF_CDM_dir,'FNET.pth')
        torch.save(FNET_TF_CDM.state_dict(),FNET_path_TF_CDM)
    else:
        print('[INFO] the FNET.pth file has already been generated, skip the training process.')
    print("\n")
    print('[SUCCESS] training process finished.')
    print("="*60)
    print("The choice 1 is finished. You may need to run the choice 2 to get the prediction results.")
    print("="*60)
    exit()
elif choice == '2':

    print("DRAWING PLOTS...")
    print("="*60)
    print("1. Trajectory Comparison")
    print("2. Drift and Diffusion")
    print("3. Conditional Distribution")
    print("="*60)
    while True:
        plot_choice = input("Choose the plot to draw (1, 2, 3):").strip()
        if plot_choice in ['1', '2', '3']:
            break
        else:
            print("Please enter '1', '2', or '3'.")
    
    if plot_choice == '1':
        plot_trajectory_comparison_simulation(
            second_stage_dir_FEX=second_stage_FEX_dir,
            All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
            model_name=model_name,
            noise_level=args.NOISE_LEVEL,
            device=device
        )
    elif plot_choice == '2':
        plot_drift_and_diffusion(
            second_stage_dir_FEX=second_stage_FEX_dir,
            All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
            model_name=model_name,
            noise_level=args.NOISE_LEVEL,
            device=device
        )
    elif plot_choice == '3':
        plot_conditional_distribution(
            second_stage_dir_FEX=second_stage_FEX_dir,
            All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
            model_name=model_name,
            noise_level=args.NOISE_LEVEL,
            device=device
        )
    else:
        print("Please enter '1', '2', or '3'.")