import os
import sys
import numpy as np
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory to path to access Example module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.nn.functional as F
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

# ==================== Helper Functions ====================
def run_time_dependent_trajectory_simulation(
    model_name, params, models_dict, scaler, dimension, 
    NPATH, TIME_AMOUNT, dt, total_time_steps, device, base_path, 
    noise_level, initial_values=None
):
    """
    Run time-dependent trajectory simulation and return results for plotting.
    
    Args:
        model_name: Name of the model (e.g., 'Trigonometric1d')
        params: Model parameters dictionary
        models_dict: Dictionary of time-dependent models
        scaler: Scaling factor for stochastic updates
        dimension: Dimension of the system
        NPATH: Number of paths for simulation
        TIME_AMOUNT: Total time for simulation
        dt: Time step size
        total_time_steps: Total number of time steps
        device: Device to use
        base_path: Base path for loading FEX model
        noise_level: Noise level
        initial_values: List of initial values (default: [-3, 0.6, 3] for Trigonometric1d)
    
    Returns:
        results_dict: Dictionary containing simulation results for each initial value
    """
    from utils.helper import predict_time_dependent_stochastic
    
    # Choose initial values based on model
    if initial_values is None:
        if model_name == 'Trigonometric1d':
            initial_values = [-3, 0.6, 3]
        elif model_name == 'DoubleWell1d':
            initial_values = [-5, 1.5, 5]
        elif model_name == 'EXP1d':
            initial_values = [-2, 1.5, 2]  # EXP1d initial values
        elif model_name == 'OU1d':
            initial_values = [-6, 1.5, 6]
        elif model_name == 'MM1d':
            initial_values = [-0.5, 0.6, 1.5]  # MM1d initial values
        else:
            initial_values = [-5, 1.5, 5]  # Default fallback
    
    # Initialize simulation arrays
    num_steps = int(TIME_AMOUNT / dt)
    
    print(f"\n[INFO] Running simulations for initial values: {initial_values}")
    print("="*80)
    
    # Store results for all initial values
    results_dict = {}
    
    # Run simulation for each initial value
    for initial_value in initial_values:
        print(f"\n[INFO] Running simulation for initial value: {initial_value}")
        
        # Initialize arrays for both ground truth and FEX prediction
        u_all_ground_truth = np.zeros((NPATH, dimension, num_steps + 1), dtype=np.float32)
        u_pred_all_FEX = np.zeros((NPATH, dimension, num_steps + 1), dtype=np.float32)
        
        # Use 'value' type for initial conditions
        initial_state_1d = initial_value * np.ones(NPATH)
        
        # Reshape to (NPATH, dimension)
        if dimension == 1:
            initial_state = initial_state_1d[:, np.newaxis]
        else:
            # For multi-dimensional, repeat the same initial condition across dimensions
            initial_state = np.repeat(initial_state_1d[:, np.newaxis], dimension, axis=1)
        print(f'[INFO] the initial state shape is: {initial_state.shape}')
        # Verify shape matches
        assert initial_state.shape == (NPATH, dimension), \
            f"Initial state shape {initial_state.shape} doesn't match expected ({NPATH}, {dimension})"
        
        # Initialize both ground truth and prediction with same initial state
        u_all_ground_truth[:, :, 0] = initial_state.copy()
        u_pred_all_FEX[:, :, 0] = initial_state.copy()
        current_state_ground_truth = initial_state.copy()
        current_pred_state_FEX = initial_state.copy()
        
        # Store results for this initial value
        results_dict[initial_value] = {
            'u_all_ground_truth': u_all_ground_truth,
            'u_pred_all_FEX': u_pred_all_FEX,
            'current_state_ground_truth': current_state_ground_truth,
            'current_pred_state_FEX': current_pred_state_FEX
        }
    
    # Extract domain_folder from base_path
    domain_folder = None
    if base_path:
        path_parts = base_path.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                break
    
    # FEX model wrapper (simplified)
    def learned_model_wrapper(x):
        return FEX_model_learned(x, 
                               model_name=model_name,
                               noise_level=noise_level,
                               device=str(device),
                               base_path=base_path,
                               domain_folder=domain_folder)
    
    print(f"\n[INFO] Starting simulation: {num_steps} steps, {NPATH} paths, dt={dt}")
    print("="*80)
    # Run simulation for each initial value
    for initial_value in initial_values:
        print(f"\n[INFO] Starting simulation for initial value {initial_value}: {num_steps} steps, {NPATH} paths, dt={dt}")
        print("="*80)
        
        # Get arrays for this initial value
        u_all_ground_truth = results_dict[initial_value]['u_all_ground_truth']
        u_pred_all_FEX = results_dict[initial_value]['u_pred_all_FEX']
        current_state_ground_truth = results_dict[initial_value]['current_state_ground_truth']
        current_pred_state_FEX = results_dict[initial_value]['current_pred_state_FEX']
        
        # Ground truth SDE parameters
        if model_name == 'Trigonometric1d':
            k = 1  # frequency parameter
            sig = params['sig'] * noise_level
        
        # Simulation loop
        for idx in range(1, num_steps + 1):
            # Generate Wiener increments (use same for both ground truth and prediction for fair comparison)
            Winc = np.random.randn(NPATH, dimension)
            Winc_tensor = torch.tensor(Winc, dtype=torch.float32).to(device)
            dW = np.sqrt(dt) * Winc
            
            # Current time
            current_time = (idx - 1) * dt
            
            # ========== GROUND TRUTH SIMULATION ==========
            if model_name == 'Trigonometric1d':
                # Trigonometric SDE: dX_t = sin(2*k*pi*X_t)dt + sig*cos(2*k*pi*t)*dW_t
                drift = np.sin(2 * k * np.pi * current_state_ground_truth[:, 0])
                diffusion = sig * np.cos(2 * k * np.pi * current_time)  # Diffusion: function of time, not X_t
                next_state_ground_truth = current_state_ground_truth.copy()
                next_state_ground_truth[:, 0] = current_state_ground_truth[:, 0] + drift * dt + diffusion * dW[:, 0]
            elif model_name == 'OL2d':
                # OL2d: 2D potential-based SDE
                # V(x,y) = 2.5*(x^2-1)^2 + 5*y^2
                # dVdx = [10*x*(x^2-1), 10*y]
                # drift = -dVdx/gamma = [-10*x*(x^2-1), -10*y] = [-10*x^3 + 10*x, -10*y]
                # Dimension 1: drift = -10*x1^3 + 10*x1 = 10*x1 - 10*x1^3
                # Dimension 2: drift = -10*x2
                sig_base = params['sig'] * noise_level
                gamma = np.ones(2)  # Friction coefficients
                
                # Compute drift for each dimension
                x1 = current_state_ground_truth[:, 0]
                x2 = current_state_ground_truth[:, 1] if dimension >= 2 else np.zeros_like(x1)
                
                drift1 = -10 * x1 * (x1**2 - 1)  # -10*x1^3 + 10*x1
                drift2 = -10 * x2  # -10*x2
                
                # Diffusion: Sigma * dW
                Sigma = sig_base * np.eye(dimension)
                diffusion = np.dot(dW, Sigma.T)  # (NPATH, dimension)
                
                next_state_ground_truth = current_state_ground_truth.copy()
                next_state_ground_truth[:, 0] = current_state_ground_truth[:, 0] + drift1 * dt + diffusion[:, 0]
                if dimension >= 2:
                    next_state_ground_truth[:, 1] = current_state_ground_truth[:, 1] + drift2 * dt + diffusion[:, 1]
            elif model_name == 'MM1d':
                # MM1d: dX_t = (tanh(X_t) - 0.5*X_t)dt + sig*dB_t
                sig = params['sig'] * noise_level
                x1 = current_state_ground_truth[:, 0]
                # Drift: tanh(x1) - 0.5*x1
                drift = np.tanh(x1) - 0.5 * x1
                # Diffusion: sig (constant)
                diffusion = sig * dW[:, 0]
                next_state_ground_truth = current_state_ground_truth.copy()
                next_state_ground_truth[:, 0] = current_state_ground_truth[:, 0] + drift * dt + diffusion
        
         
            u_all_ground_truth[:, :, idx] = next_state_ground_truth
            current_state_ground_truth = next_state_ground_truth
            
            # ========== FEX PREDICTION ==========
            # FEX deterministic update
            current_tensor = torch.tensor(current_pred_state_FEX, dtype=torch.float32).to(device)
            with torch.no_grad():
                FEX_update = learned_model_wrapper(current_tensor).cpu().numpy()
            
            det_update = FEX_update * dt
            
            # Stochastic update using time-dependent model
            t_idx = min(idx - 1, total_time_steps - 1)  # Use time step index
            
            stoch_update_FEX = predict_time_dependent_stochastic(
                Winc_tensor, t_idx, models_dict, device=str(device)
            )
            
            if stoch_update_FEX is None:
               raise ValueError(f"[ERROR] Stochastic update is None for time step {t_idx}, you should run choice 1 first to train models.")
            
            # Apply scaler to stochastic update (same as time-independent case)
            stoch_update_FEX = stoch_update_FEX / scaler
            
            # Update prediction
            next_pred_state_FEX = current_pred_state_FEX + det_update + stoch_update_FEX
            u_pred_all_FEX[:, :, idx] = next_pred_state_FEX
            
            # Update state
            current_pred_state_FEX = next_pred_state_FEX
            
            # Update results dict
            results_dict[initial_value]['current_state_ground_truth'] = current_state_ground_truth
            results_dict[initial_value]['current_pred_state_FEX'] = current_pred_state_FEX
            
            # Print progress
            if idx % 50 == 0:
                mean_error = np.mean(np.abs(np.mean(current_state_ground_truth, axis=0) - np.mean(current_pred_state_FEX, axis=0)))
                print(f"Step {idx}/{num_steps}: Mean error = {mean_error:.6f}")
        
        print(f"\n[INFO] Simulation completed for initial value {initial_value}!")
        print("="*80)
    
    return results_dict, initial_values, num_steps

# Set device
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    device = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU instead")

#============================Load time dependent or not============================
if args.model in ['OU1d', 'DoubleWell1d', 'EXP1d', 'OL2d', 'MM1d']:
    TIME_DEPENDENT = False
elif args.model in ['Trigonometric1d']:
    TIME_DEPENDENT = True
else:
    raise ValueError(f"Model {args.model} is not supported.")


#===========================Path part==============================================
print("\n"+ "="*60)
print("\n[INFO] Setting up the path...")

# Check if CUDA is available and set base path accordingly
# Use args.model if available, otherwise fall back to args.params_name
model_name = getattr(args, 'model', None) or getattr(args, 'params_name', 'default')

# Create domain folder name
domain_folder = f'domain_{args.DOMAIN_START}_{args.DOMAIN_END}'

if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    base_path = os.path.join(config.DIR_PROJECT, 'Results', 'gpu_folder', model_name, domain_folder)
    print(f"Using GPU folder: {base_path}")
else:
    base_path = os.path.join(config.DIR_PROJECT, 'Results', 'cpu_folder', model_name, domain_folder)
    print(f"Using CPU folder: {base_path}")

# Set up model path and save directory
model_PATH = Path(base_path)
save_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.TRAIN_SIZE}')
os.makedirs(save_dir, exist_ok=True)
print(f'[INFO] The save directory is set up successfully: {save_dir}')
print("="*60)


# Set up save directory for second stage (with domain folder)
second_stage_FEX_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.TRAIN_SIZE}')
All_stage_TF_CDM_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'All_stage_TF_CDM_{args.TRAIN_SIZE}')
All_stage_FEX_VAE_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'All_stage_FEX_VAE_{args.TRAIN_SIZE}')
All_stage_FEX_NN_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'All_stage_FEX_NN_{args.TRAIN_SIZE}')
os.makedirs(second_stage_FEX_dir, exist_ok=True)
os.makedirs(All_stage_TF_CDM_dir, exist_ok=True)
os.makedirs(All_stage_FEX_VAE_dir, exist_ok=True)
os.makedirs(All_stage_FEX_NN_dir, exist_ok=True)
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
                                 base_path=base_path,
                                 domain_folder=domain_folder)
    
    
    print(data['dataset'].shape)
    
    # Initialize short_indx_shared (only needed for non-time-dependent cases)
    short_indx_shared = None
    current_state_full_time_dependent = None
    
    if TIME_DEPENDENT == False:
        #===================================================================================
        # Compute short_indx once and reuse for both FEX-DM and TF-CDM
        # Since both use the same current_state_full and current_state_train_np, short_indx will be identical
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

    elif TIME_DEPENDENT == True:
        residuals_FEX, u_current_reshaped, residual_cov_time = generate_euler_maruyama_residue(func=learned_model_wrapper, current_state=current_state_full, next_state=next_state_full, dt=dt, data=data)
        print(f'[INFO] Residuals shape: {residuals_FEX.shape}')
        print(f'[INFO] u_current_reshaped shape: {u_current_reshaped.shape}')
        print(f'[INFO] residual_cov_time shape: {residual_cov_time.shape}')
        residuals_FEX_for_step = residuals_FEX      
        # For time-dependent TF-CDM, compute residuals from dataset
        # dataset shape: (dim, time_steps_plus_1, MC_samples)
        dataset = data['dataset']
        dim, time_steps_plus_1, MC_samples = dataset.shape
        time_steps = time_steps_plus_1 - 1
        # Reshape dataset to (MC_samples, dim, time_steps_plus_1) for easier processing
        dataset_reshaped = np.transpose(dataset, (2, 0, 1))  # (MC_samples, dim, time_steps_plus_1)
        # Compute TF-CDM residuals: next_state - current_state for each time step
        residuals_TF_CDM_for_step = np.zeros((MC_samples, dim, time_steps))
        for t in range(time_steps):
            residuals_TF_CDM_for_step[:, :, t] = dataset_reshaped[:, :, t+1] - dataset_reshaped[:, :, t]
        # For time-dependent, use u_current_reshaped which has shape (MC_samples, dim, time_steps)
        current_state_full_time_dependent = u_current_reshaped
        # Store time_steps_plus_1 for later use in generate_second_step
        time_steps_plus_1_for_ode = time_steps_plus_1
    

    
    # check FEX ODE solution and ZT solution
    if not os.path.exists(os.path.join(second_stage_FEX_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(second_stage_FEX_dir,'ZT_Solution.npy')):
        # Use time-dependent current_state if available, otherwise use regular current_state_full
        current_state_for_ode = current_state_full_time_dependent if TIME_DEPENDENT else current_state_full
        ODE_Solution_FEX,ZT_Solution_FEX = generate_second_step(
            current_state_for_ode, residuals_FEX_for_step, scaler, dt, train_size, device,
            num_time_points=None,  # Use None to process all time steps (should be 100)
            time_dependent=TIME_DEPENDENT,
            current_state_train=current_state_train_np,  # Pass training data from npz
            short_indx=short_indx_shared if not TIME_DEPENDENT else None,  # Use shared short_indx only for time-independent cases
            save_short_indx_dir=second_stage_FEX_dir if TIME_DEPENDENT else None  # Save short_indx for time-dependent cases
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
        print(f'[INFO] the ODE solution shape is: {ODE_Solution_FEX.shape}')
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        ZT_Solution_FEX = np.load(os.path.join(second_stage_FEX_dir, "ZT_Solution.npy"))
        print(f'[INFO] the ZT solution shape is: {ZT_Solution_FEX.shape}')
    
    # check TF-CDM ODE solution and ZT solution
    if not os.path.exists(os.path.join(All_stage_TF_CDM_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(All_stage_TF_CDM_dir,'ZT_Solution.npy')):
        # Use time-dependent current_state if available, otherwise use regular current_state_full
        current_state_for_ode_tf_cdm = current_state_full_time_dependent if TIME_DEPENDENT else current_state_full
        ODE_Solution_TF_CDM,ZT_Solution_TF_CDM = generate_second_step(
            current_state_for_ode_tf_cdm, residuals_TF_CDM_for_step, scaler_TF_CDM, dt, train_size, device,
            num_time_points=None,  # Use None to process all time steps (should be 100)
            time_dependent=TIME_DEPENDENT,
            current_state_train=current_state_train_np,  # Pass training data from npz
            short_indx=short_indx_shared if not TIME_DEPENDENT else None,  # Use shared short_indx only for time-independent cases
            save_short_indx_dir=second_stage_FEX_dir if TIME_DEPENDENT else None  # Load short_indx from FEX-DM directory for time-dependent cases
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
        print(f'[INFO] the ODE solution shape is: {ODE_Solution_TF_CDM.shape}')
        
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        ZT_Solution_TF_CDM = np.load(os.path.join(All_stage_TF_CDM_dir, "ZT_Solution.npy"))
        print(f'[INFO] the ZT solution shape is: {ZT_Solution_TF_CDM.shape}')
    

    if TIME_DEPENDENT == False:
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
    
        # Prepare VAE training data similar to FEX-DM and TF-CDM
        # Calculate residuals_vae_for_step = residuals_FEX_for_step * DIFF_SCALE
        residuals_vae_for_step = residuals_FEX_for_step * args.DIFF_SCALE
    
        # Convert to tensor
        residuals_vae_tensor = torch.tensor(residuals_vae_for_step, dtype=torch.float32).to(device)
    
        # Split into train and validation sets (same pattern as FEX-DM and TF-CDM)
        NTrain_VAE = int(residuals_vae_tensor.shape[0] * 0.8)
        NValid_VAE = int(residuals_vae_tensor.shape[0] * 0.2)
    
        residuals_vae_train = residuals_vae_tensor[NTrain_VAE:]
        residuals_vae_valid = residuals_vae_tensor[NValid_VAE:]
        print(f'[INFO] the residuals_vae_train shape is: {residuals_vae_train.shape}')
        print(f'[INFO] the residuals_vae_valid shape is: {residuals_vae_valid.shape}')
    
        # Define VAE loss function
        def vae_loss(recon_x, x, mu, logvar):
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_div
    
        # Train FEX-DM if FNET.pth doesn't exist
        FNET_path_FEX = os.path.join(second_stage_FEX_dir,'FNET.pth')
        if not os.path.exists(FNET_path_FEX):
            print('[INFO] FEX-DM FNET.pth not found, starting training...')
            FNET_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
            FNET_optim_FEX = torch.optim.Adam(FNET_FEX.parameters(), lr=args.NN_SOLVER_LR, weight_decay=1e-6)
            FNET_FEX.zero_grad()
            criterion = torch.nn.MSELoss()
            n_iteration = args.NN_SOLVER_EPOCHS
            best_valid_err_FEX = float('inf')
        
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
             
                # Compute validation loss
                with torch.no_grad():
                    pred_valid_FEX = FNET_FEX(ZT_Train_new_valid_FEX.reshape((ZT_Train_new_valid_FEX.shape[0],1)))
                    loss_valid_FEX = criterion(pred_valid_FEX,ODE_Train_new_valid_FEX)
            
                if loss_valid_FEX < best_valid_err_FEX:
                    FNET_FEX.update_best()
                    best_valid_err_FEX = loss_valid_FEX
        
                if epoch % 100 == 0:
                    print(f"{'FEX-DM':<10} {epoch+1:<8} {loss_FEX.item():<15.6f} {loss_valid_FEX.item():<15.6f}")
                    print("-"*80)
        
            FNET_FEX.final_update()
            torch.save(FNET_FEX.state_dict(), FNET_path_FEX)
            print(f'[INFO] FEX-DM FNET.pth saved to {FNET_path_FEX}')
        else:
            print('[INFO] FEX-DM FNET.pth already exists, skipping training.')
        
        # Train TF-CDM if FNET.pth doesn't exist
        FNET_path_TF_CDM = os.path.join(All_stage_TF_CDM_dir,'FNET.pth')
        if not os.path.exists(FNET_path_TF_CDM):
            print('[INFO] TF-CDM FNET.pth not found, starting training...')
            FNET_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
            FNET_optim_TF_CDM = torch.optim.Adam(FNET_TF_CDM.parameters(), lr=args.NN_SOLVER_LR, weight_decay=1e-6)
            FNET_TF_CDM.zero_grad()
            criterion_TF_CDM = torch.nn.MSELoss()
            n_iteration = args.NN_SOLVER_EPOCHS
            best_valid_err_TF_CDM = float('inf')
            
            # Print table header
            print("\n" + "="*80)
            print(f"{'Model':<10} {'Epoch':<8} {'Train Loss':<15} {'Valid Loss':<15}")
            print("="*80)

            for epoch in range(n_iteration):
                FNET_optim_TF_CDM.zero_grad()
                pred_TF_CDM = FNET_TF_CDM(ZT_Train_new_normal_TF_CDM)
                loss_TF_CDM = criterion_TF_CDM(pred_TF_CDM,ODE_Train_new_normal_TF_CDM)
                loss_TF_CDM.backward()
                FNET_optim_TF_CDM.step()
                 
                # Compute validation loss
                with torch.no_grad():
                    pred_valid_TF_CDM = FNET_TF_CDM(ZT_Train_new_valid_TF_CDM)
                    loss_valid_TF_CDM = criterion_TF_CDM(pred_valid_TF_CDM,ODE_Train_new_valid_TF_CDM)
                
                if loss_valid_TF_CDM < best_valid_err_TF_CDM:
                    FNET_TF_CDM.update_best()
                    best_valid_err_TF_CDM = loss_valid_TF_CDM
            
                if epoch % 100 == 0:
                    print(f"{'TF-CDM':<10} {epoch+1:<8} {loss_TF_CDM.item():<15.6f} {loss_valid_TF_CDM.item():<15.6f}")
                    print("-"*80)
            
            FNET_TF_CDM.final_update()
            torch.save(FNET_TF_CDM.state_dict(), FNET_path_TF_CDM)
            print(f'[INFO] TF-CDM FNET.pth saved to {FNET_path_TF_CDM}')
        else:
            print('[INFO] TF-CDM FNET.pth already exists, skipping training.')
        
        # Train FEX-VAE
        VAE_path = os.path.join(All_stage_FEX_VAE_dir, 'VAE_FEX.pth')
        if not os.path.exists(VAE_path):
            print('[INFO] Training FEX-VAE model...')
            
            # Use prepared residuals_vae_train and residuals_vae_valid (already normalized and split)
            vae_input_train = residuals_vae_train
            vae_input_valid = residuals_vae_valid
            
            # Initialize VAE model
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            print(f"VAE device: {next(VAE_FEX.parameters()).device}")
            VAE_FEX.zero_grad()
            
            optimizer_vae = torch.optim.Adam(VAE_FEX.parameters(), lr=0.001, weight_decay=1e-6)
            best_valid_err_vae = float('inf')
            n_iter_vae = args.NN_SOLVER_EPOCHS
            
            print("\n" + "="*80)
            print(f"{'Model':<10} {'Epoch':<8} {'Train Loss':<15} {'Valid Loss':<15}")
            print("="*80)
            
            for j in range(n_iter_vae):
                optimizer_vae.zero_grad()
                
                recon_x, mu, logvar = VAE_FEX(vae_input_train)
                loss = vae_loss(recon_x, vae_input_train, mu, logvar)
                loss.backward()
                optimizer_vae.step()
                
                # Validation
                with torch.no_grad():
                    recon_x_valid, mu_valid, logvar_valid = VAE_FEX(vae_input_valid)
                    valid_loss = vae_loss(recon_x_valid, vae_input_valid, mu_valid, logvar_valid)
                
                if valid_loss < best_valid_err_vae:
                    VAE_FEX.update_best()
                    best_valid_err_vae = valid_loss
                
                if j % 100 == 0:
                    print(f"{'FEX-VAE':<10} {j+1:<8} {loss.item():<15.6f} {valid_loss.item():<15.6f}")
                    print("-"*80)
            
            VAE_FEX.final_update()
            torch.save(VAE_FEX.state_dict(), VAE_path)
            print(f'[INFO] VAE_FEX model saved to {VAE_path}')
        else:
            print('[INFO] the VAE_FEX.pth file has already been generated, skip the training process.')
        
        # Train FEX-NN (Covariance Matrix Learning)
        FEX_NN_path = os.path.join(All_stage_FEX_NN_dir, 'FEX_NN.pth')
        if not os.path.exists(FEX_NN_path):
            print('[INFO] Training FEX-NN (moment matching learning) model...')
            from utils.ODEParser import CovarianceNet
            
            # Prepare training data: input is current_state, target is (r_t * r_t^T) / dt
            # Input: current_state_train_np shape: (train_size, dim)
            # Target: (r_t * r_t^T) / dt computed from residuals_FEX_for_step
            
            # Handle residuals shape to match current_state_train_np
            if residuals_FEX_for_step.ndim == 3:
                # Multi-D case: (size, dim, time_steps)
                # Take first time step to match train_size
                if residuals_FEX_for_step.shape[0] == train_size:
                    residuals_flat = residuals_FEX_for_step[:, :, 0]  # (train_size, dim)
                else:
                    # Take first train_size samples from first time step
                    residuals_flat = residuals_FEX_for_step[:train_size, :, 0]  # (train_size, dim)
            else:
                # 1D case: (size, 1) or (size,)
                if residuals_FEX_for_step.ndim == 1:
                    residuals_FEX_for_step = residuals_FEX_for_step[:, np.newaxis]
                residuals_flat = residuals_FEX_for_step[:train_size]  # (train_size, 1)
            
            # Ensure residuals_flat matches current_state_train_np shape
            if residuals_flat.shape[0] != current_state_train_np.shape[0]:
                residuals_flat = residuals_flat[:current_state_train_np.shape[0]]
            
            dim = current_state_train_np.shape[1]
            
            # Compute target: (r_t * r_t^T) / dt for each sample
            # For 1D: target is r_t^2 / dt (scalar)
            # For multi-D: target is outer product r_t * r_t^T / dt (matrix, flattened to vector)
            if dim == 1:
                # 1D case: target is (r_t^2) / dt, shape (N, 1)
                target_cov = (residuals_flat ** 2) / dt
                output_dim = 1
            else:
                # Multi-D case: compute outer product r_t * r_t^T for each sample
                # Shape: (N, dim, dim) -> flatten to (N, dim*dim)
                N = residuals_flat.shape[0]
                target_cov = np.zeros((N, dim * dim))
                for i in range(N):
                    r_t = residuals_flat[i:i+1, :]  # (1, dim)
                    r_outer = np.outer(r_t, r_t)  # (dim, dim)
                    target_cov[i, :] = (r_outer / dt).flatten()
                output_dim = dim * dim
            
            print(f'[INFO] FEX-NN training data shape: input {current_state_train_np.shape}, target {target_cov.shape}')
            
            # Convert to tensors
            # Input: current_state_train_tensor (current state)
            # Target: target_cov (covariance matrix from residuals)
            current_state_train_tensor = torch.tensor(current_state_train_np, dtype=torch.float32).to(device)
            target_cov_tensor = torch.tensor(target_cov, dtype=torch.float32).to(device)
            
            # Split into train and validation (80/20)
            N_total = current_state_train_tensor.shape[0]
            NTrain_NN = int(N_total * 0.8)
            NValid_NN = N_total - NTrain_NN
            
            # Input: current_state, Target: target_cov
            x_train_NN = current_state_train_tensor[:NTrain_NN]
            target_train_NN = target_cov_tensor[:NTrain_NN]
            x_valid_NN = current_state_train_tensor[NTrain_NN:]
            target_valid_NN = target_cov_tensor[NTrain_NN:]
            
            # Initialize model
            FEX_NN = CovarianceNet(input_dim=dim, output_dim=output_dim, hid_size=50).to(device)
            FEX_NN.zero_grad()
            optimizer_nn = torch.optim.Adam(FEX_NN.parameters(), lr=args.NN_SOLVER_LR, weight_decay=1e-6)
            criterion_nn = torch.nn.MSELoss()
            
            n_iter_nn = args.NN_SOLVER_EPOCHS
            best_valid_err_nn = float('inf')
            
            print("\n" + "="*80)
            print(f"{'Model':<10} {'Epoch':<8} {'Train Loss':<15} {'Valid Loss':<15}")
            print("="*80)
            
            for j in range(n_iter_nn):
                optimizer_nn.zero_grad()
                
                # Forward pass
                pred_cov = FEX_NN(x_train_NN)
                loss = criterion_nn(pred_cov, target_train_NN)
                loss.backward()
                optimizer_nn.step()
                
                # Validation
                with torch.no_grad():
                    pred_cov_valid = FEX_NN(x_valid_NN)
                    valid_loss = criterion_nn(pred_cov_valid, target_valid_NN)
                
                if valid_loss < best_valid_err_nn:
                    FEX_NN.update_best()
                    best_valid_err_nn = valid_loss
                
                if j % 100 == 0:
                    print(f"{'FEX-NN':<10} {j+1:<8} {loss.item():<15.6f} {valid_loss.item():<15.6f}")
                    print("-"*80)
            
            FEX_NN.final_update()
            torch.save(FEX_NN.state_dict(), FEX_NN_path)
            
            print(f'[INFO] FEX_NN model saved to {FEX_NN_path}')
        else:
            print('[INFO] the COV_NN.pth file has already been generated, skip the training process.')
        
        print("\n")
        print('[SUCCESS] training process finished.')
        print("="*60)
        print("The choice 1 is finished. You may need to run the choice 2 to get the prediction results.")
        print("="*60)
        exit()
    else:
        # Time-dependent case
        from utils.helper import train_FN_time_dependent, save_parameters, VAE
        import torch.nn.functional as F
        
        # Train FEX-DM (time-dependent models)
        train_FN_time_dependent(ODE_Solution_FEX, ZT_Solution_FEX, dimension, device, 
                               args.NN_SOLVER_LR, args.NN_SOLVER_EPOCHS, best_valid_err=5.0, 
                               save_dir=second_stage_FEX_dir, num_time_points=None, time_range=None, dt=dt)
        
        # Train TF-CDM (time-dependent, separate model for each time step)
        print("\n[INFO] Training TF-CDM models for time-dependent case...")
        from utils.helper import train_TF_CDM_time_dependent
        
        # Ensure current_state_full_time_dependent matches ODE_Solution_TF_CDM shape
        # current_state_full_time_dependent: (MC_samples, dim, time_steps)
        # ODE_Solution_TF_CDM: (size, dim, time_steps)
        # They should have the same size and time_steps
        size_tf_cdm, dim_tf_cdm, time_steps_tf_cdm = ODE_Solution_TF_CDM.shape
        MC_samples_state, dim_state, time_steps_state = current_state_full_time_dependent.shape
        
        # Use the smaller size to ensure compatibility
        min_size = min(size_tf_cdm, MC_samples_state)
        min_time_steps = min(time_steps_tf_cdm, time_steps_state)
        
        # Slice to match dimensions
        ODE_Solution_TF_CDM_aligned = ODE_Solution_TF_CDM[:min_size, :, :min_time_steps]
        ZT_Solution_TF_CDM_aligned = ZT_Solution_TF_CDM[:min_size, :, :min_time_steps]
        current_state_aligned = current_state_full_time_dependent[:min_size, :, :min_time_steps]
        
        train_TF_CDM_time_dependent(
            ODE_Solution_TF_CDM_aligned,
            ZT_Solution_TF_CDM_aligned,
            current_state_aligned,
            dimension,
            device,
            args.NN_SOLVER_LR,
            args.NN_SOLVER_EPOCHS,
            best_valid_err=5.0,
            save_dir=All_stage_TF_CDM_dir,
            num_time_points=None,
            time_range=None,
            dt=dt
        )
        
        # Train FEX-VAE (time-dependent, separate model for each time step)
        print("\n[INFO] Training FEX-VAE models for time-dependent case...")
        from utils.helper import train_VAE_time_dependent
        
        # Prepare residuals for VAE: scale by DIFF_SCALE
        # residuals_FEX_for_step has shape (MC_samples, dim, time_steps) from line 378
        residuals_vae_for_step = residuals_FEX_for_step * args.DIFF_SCALE
        
        # Ensure residuals_vae_for_step is 3D: (size, dim, time_steps)
        # If it's already 3D, use it directly
        if residuals_vae_for_step.ndim == 3:
            # Use the first size samples to match training size
            size_actual = min(residuals_vae_for_step.shape[0], train_size)
            residuals_vae_aligned = residuals_vae_for_step[:size_actual, :, :]
        else:
            raise ValueError(f"Expected 3D residuals for VAE training, got shape {residuals_vae_for_step.shape}")
        
        train_VAE_time_dependent(
            residuals_vae_aligned,
            dimension,
            device,
            learning_rate=0.001,
            n_iter=args.NN_SOLVER_EPOCHS,
            save_dir=All_stage_FEX_VAE_dir,
            num_time_points=None,
            time_range=None,
            dt=dt
        )
        
        # Train FEX-NN (time-dependent, separate model for each time step)
        print("\n[INFO] Training FEX-NN models for time-dependent case...")
        
        # Prepare residuals and current_state for FEX-NN training
        # residuals_FEX_for_step has shape (MC_samples, dim, time_steps) from line 378
        # current_state_full_time_dependent has shape (MC_samples, dim, time_steps) from line 391
        
        # Ensure shapes match
        size_residuals, dim_residuals, time_steps_residuals = residuals_FEX_for_step.shape
        size_state, dim_state, time_steps_state = current_state_full_time_dependent.shape
        
        # Use the smaller size to ensure compatibility
        min_size = min(size_residuals, size_state)
        min_time_steps = min(time_steps_residuals, time_steps_state)
        
        # Slice to match dimensions
        residuals_nn_aligned = residuals_FEX_for_step[:min_size, :, :min_time_steps]
        current_state_nn_aligned = current_state_full_time_dependent[:min_size, :, :min_time_steps]
        
        train_FEX_NN_time_dependent(
            residuals_nn_aligned,
            current_state_nn_aligned,
            dimension,
            device,
            learning_rate=args.NN_SOLVER_LR,
            n_iter=args.NN_SOLVER_EPOCHS,
            save_dir=All_stage_FEX_NN_dir,
            num_time_points=None,
            time_range=None,
            dt=dt
        )
        
        print("\n")
        print('[SUCCESS] Training process finished for time-dependent case.')
        print("="*60)
        print("The choice 1 is finished. You may need to run the choice 2 to get the prediction results.")
        print("="*60)
        exit()
        





elif choice == '2':
    print("\n[INFO] Generating prediction results and drawing plots...")
    
    if TIME_DEPENDENT == False:
        # Time-independent case - use existing plotting functions
        print("DRAWING PLOTS...")
        print("="*60)
        print("1. Trajectory Comparison")
        print("2. Drift and Diffusion")
        print("3. Conditional Distribution")
        print("4. Drift and Diffusion with Error Plots")
        print("5. Trajectory Error Estimation")
        print("6. Conditional Distribution with Errors")
        print("="*60)
        while True:
            plot_choice = input("Choose the plot to draw (1, 2, 3, 4, 5, 6):").strip()
            if plot_choice in ['1', '2', '3', '4', '5', '6']:
                break
            else:
                print("Please enter '1', '2', '3', '4', '5', or '6'.")
        
        if plot_choice == '1':
            plot_trajectory_comparison_simulation(
                second_stage_dir_FEX=second_stage_FEX_dir,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                model_name=model_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
                seed=args.SEED
            )
        elif plot_choice == '2':
            plot_drift_and_diffusion(
                second_stage_dir_FEX=second_stage_FEX_dir,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                model_name=model_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
                seed=args.SEED
            )
        elif plot_choice == '3':
            plot_conditional_distribution(
                second_stage_dir_FEX=second_stage_FEX_dir,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                model_name=model_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
                seed=args.SEED
            )
        elif plot_choice == '4':
            plot_drift_and_diffusion_with_errors(
                second_stage_dir_FEX=second_stage_FEX_dir,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                model_name=model_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
                seed=args.SEED
            )
        elif plot_choice == '5':
            plot_trajectory_error_estimation(
                second_stage_dir_FEX=second_stage_FEX_dir,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                model_name=model_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
                seed=args.SEED
            )
        elif plot_choice == '6':
            plot_conditional_distribution_with_errors(
                second_stage_dir_FEX=second_stage_FEX_dir,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                model_name=model_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
                seed=args.SEED
            )
            # For DoubleWell1d, also generate the timeseries plot
            if model_name == 'DoubleWell1d':
                from utils.plot import plot_conditional_distribution_doublewell_timeseries
                plot_conditional_distribution_doublewell_timeseries(
                    second_stage_dir_FEX=second_stage_FEX_dir,
                    All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                    All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                    All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                    model_name=model_name,
                    noise_level=args.NOISE_LEVEL,
                    device=device,
                    initial_value=1.5,
                    times_to_plot=[5, 30, 100],
                    save_dir=None,  # Will use default from second_stage_dir_FEX
                    figsize=(16, 5),
                    dpi=300,
                    seed=args.SEED
                )
    else:
        # Time-dependent case - plot selection menu
        print("DRAWING PLOTS...")
        print("="*60)
        print("1. Trajectory Error Estimation")
        print("2. Drift and Diffusion")
        print("3. Conditional Distribution (t=0)")
        print("="*60)
        while True:
            plot_choice = input("Choose the plot to draw (1, 2, or 3):").strip()
            if plot_choice in ['1', '2', '3']:
                break
            else:
                print("Please enter '1', '2', or '3'.")
        
        # Load data and parameters (needed for both plot choices)
        data_file_path = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'simulation_results_noise_{args.NOISE_LEVEL}.npz')
        if not os.path.exists(data_file_path):
            raise RuntimeError(f'[ERROR] Data file not found: {data_file_path}. You should run 1stage_deterministic.py first')
        
        data = np.load(data_file_path)
        current_state_full = data['current_state']
        dimension = current_state_full.shape[1]
        
        params = params_init(case_name=model_name)
        dt = params['Dt']
        TIME_AMOUNT = params.get('TIME_AMOUNT', 1.0)
        NPATH = params.get('NPATH', 5000)
        
        # Define scaler (same as in choice 1)
        scaler = np.ones(dimension) * args.DIFF_SCALE
        scaler_TF_CDM = np.ones(dimension) * 10.0
        
        # Load time-dependent models
        from utils.helper import load_time_dependent_models, predict_time_dependent_stochastic
        
        print("\n[INFO] Loading neural network models...")
        
        models_dict = load_time_dependent_models(second_stage_FEX_dir, dimension, device=str(device))
        
        if not models_dict:
            print("[ERROR] No time-dependent models found. Please run choice 1 first to train models.")
            exit(1)
        
        # Get total time steps from models
        total_time_steps = max(models_dict.keys()) + 1
        print(f"[INFO] Total time steps: {total_time_steps}")
        
        # Create save directory for plots
        plot_save_dir = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'plots_time_dependent_{args.TRAIN_SIZE}')
        os.makedirs(plot_save_dir, exist_ok=True)
        print(f"[INFO] Saving plots to: {plot_save_dir}")
        
        if plot_choice == '1':
            # Plot 1: Trajectory Error Estimation
            print("\n[INFO] Generating trajectory error estimation plots...")
            
            # Run simulation
            results_dict, initial_values, num_steps = run_time_dependent_trajectory_simulation(
                model_name=model_name,
                params=params,
                models_dict=models_dict,
                scaler=scaler,
                dimension=dimension,
                NPATH=NPATH,
                TIME_AMOUNT=TIME_AMOUNT,
                dt=dt,
                total_time_steps=total_time_steps,
                device=device,
                base_path=base_path,
                noise_level=args.NOISE_LEVEL
            )
            
            # Plot trajectory error estimation
            from utils.plot import plot_time_dependent_trajectory_error
            
            saved_paths = plot_time_dependent_trajectory_error(
                results_dict=results_dict,
                initial_values=initial_values,
                num_steps=num_steps,
                dt=dt,
                dimension=dimension,
                save_dir=plot_save_dir,
                model_name=model_name,
                models_dict=models_dict,
                scaler=scaler,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                scaler_TF_CDM=scaler_TF_CDM,
                base_path=base_path,
                noise_level=args.NOISE_LEVEL,
                device=device,
                figsize=(18, 12),
                dpi=300
            )
            
        elif plot_choice == '2':
            # Plot 2: Drift and Diffusion
            print("\n[INFO] Generating drift and diffusion plots...")
            from utils.plot import plot_drift_and_diffusion_time_dependent
            
            drift_diff_path = plot_drift_and_diffusion_time_dependent(
                second_stage_dir_FEX=second_stage_FEX_dir,
                models_dict=models_dict,
                scaler=scaler,
                model_name=model_name,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                scaler_TF_CDM=scaler_TF_CDM,
                noise_level=args.NOISE_LEVEL,
                device=device,
                base_path=base_path,
                Npath=5000,
                N_x0=500,
                x_min=-5 if model_name == 'Trigonometric1d' else -6,
                x_max=5 if model_name == 'Trigonometric1d' else 6,
                time_steps_to_plot=None,  # Will use all available time steps
                save_dir=plot_save_dir,
                figsize=(18, 12),
                dpi=300,
                seed=args.SEED
            )
            
        elif plot_choice == '3':
            # Plot 3: Conditional Distribution at t=0
            print("\n[INFO] Generating conditional distribution plots at t=0...")
            from utils.plot import plot_conditional_distribution_time_dependent
            
            # Set initial values based on model
            if model_name == 'Trigonometric1d':
                initial_values = [-3, 0.6, 3]
            elif model_name == 'DoubleWell1d':
                initial_values = [-5, 1.5, 5]
            else:
                initial_values = [-5, 1.5, 5]  # Default fallback
            
            cond_dist_path = plot_conditional_distribution_time_dependent(
                second_stage_dir_FEX=second_stage_FEX_dir,
                models_dict=models_dict,
                scaler=scaler,
                model_name=model_name,
                All_stage_dir_TF_CDM=All_stage_TF_CDM_dir,
                All_stage_dir_FEX_VAE=All_stage_FEX_VAE_dir,
                All_stage_dir_FEX_NN=All_stage_FEX_NN_dir,
                scaler_TF_CDM=scaler_TF_CDM,
                noise_level=args.NOISE_LEVEL,
                device=device,
                base_path=base_path,
                initial_values=initial_values,
                Npath=500000,
                save_dir=plot_save_dir,
                figsize=(18, 6),
                dpi=300,
                seed=args.SEED
            )
        
        # print(f"\n[SUCCESS] All plots saved to {plot_save_dir}!")
        # print("[SUCCESS] Time-dependent prediction and plotting completed!")