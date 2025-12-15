import os
import sys
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch
import faiss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import faiss

def cond_alpha(t,dt): # in the training paper: it should be related to  b(\tau) in formula (3.1)
    return 1-t+dt

def cond_sigma2(t,dt):
    return t+dt

def f(t,dt):
    alpha_t = cond_alpha(t,dt)
    f_t = -1.0/(alpha_t)
    return f_t

def g2(t,dt):
    dsigma2_dt = 1.0
    f_t = f(t,dt)
    sigma2_t = cond_sigma2(t,dt)
    g2 = dsigma2_dt - 2*f_t*sigma2_t
    return g2
def g(t,dt):
    return (g2(t,dt))**0.5


def ODE_solver(zt,x_sample,z_sample,x0_test,
               ODESOLVER_TIME_STEPS:int=2000):
    t_vec = torch.linspace(1.0,0.0,ODESOLVER_TIME_STEPS+1)
    log_weight_likelihood = -1.0* torch.sum( (x0_test[:,None,:]-x_sample)**2/2 , axis = 2, keepdims= False)
    weight_likelihood =torch.exp(log_weight_likelihood)
    for j in range(ODESOLVER_TIME_STEPS): 
        if j% 100 == 0:
            print(f'this is {j} times / overall {ODESOLVER_TIME_STEPS} times')
        t = t_vec[j+1]
        dt = t_vec[j] - t_vec[j+1]
        #print()
        score_gauss = -1.0*(zt[:,None,:]-cond_alpha(t,dt)*z_sample)/cond_sigma2(t,dt)

        log_weight_gauss= -1.0* torch.sum( (zt[:,None,:]-cond_alpha(t,dt)*z_sample)**2/(2*cond_sigma2(t,dt)) , axis =2, keepdims= False)
        weight_temp = torch.exp( log_weight_gauss )
        weight_temp = weight_temp*weight_likelihood
        weight = weight_temp/ torch.sum(weight_temp,axis=1, keepdims=True)
        score = torch.sum(score_gauss*weight[:,:,None],axis=1, keepdims= False)  
        ## score is followed by the formula 3.11
        
        zt= zt - (f(t,dt)*zt-0.5*g2(t, dt)*score) *dt
    return zt

class FN_Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim, self.hid_size)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.output = nn.Linear(self.hid_size, self.output_dim)
        

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

    def update_best(self):
        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):
        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias


class CovarianceNet(nn.Module):
    """
    Neural network to learn covariance matrix Σ_θ(x_t) from current state x_t.
    For dimension d, outputs a d×d covariance matrix (flattened to d² vector).
    """
    
    def __init__(self, input_dim, output_dim, hid_size):
        super(CovarianceNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # dimension * dimension for full matrix
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim, self.hid_size)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.output = nn.Linear(self.hid_size, self.output_dim)
        
        # Store best weights for model checkpointing
        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x
    
    def update_best(self):
        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def final_update(self):
        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias

def process_chunk_faiss_cpu(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim, batch_size=256, sample_batch_size=100):
    """
    A function to perform vector similarity search with large `x_sample` processed in batches.

    Parameters:
    - it_n_index: Number of iterations for chunks.
    - it_size_x0train: Size of each chunk.
    - short_size: Number of nearest neighbors to find.
    - x_sample: Vectors to search against (reference vectors).
    - x0_train: Input vectors to be searched (query vectors).
    - train_size: Total number of query vectors.
    - batch_size: Number of query vectors processed at a time to prevent memory overflow.
    - sample_batch_size: Number of reference vectors (`x_sample`) processed at a time.
    
    Returns:
    - x0_train_index_initial: Indices of the nearest neighbors for each query vector.
    """
    # Ensure x_sample and x0_train are PyTorch tensors for GPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_sample = torch.tensor(x_sample, dtype=torch.float32, device=device)
    x0_train = torch.tensor(x0_train, dtype=torch.float32, device=device)

    # Prepare the output array
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)

    for jj in range(it_n_index):
        print(f'This is {jj} time')
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        print(f'start_idx is {start_idx}; end_idx is {end_idx}')

        # Extract chunk of query vectors
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Process query vectors in smaller batches to avoid memory overflow
        for batch_start in range(0, x0_train_chunk.size(0), batch_size):
            batch_end = min(batch_start + batch_size, x0_train_chunk.size(0))
            batch = x0_train_chunk[batch_start:batch_end]

            # Prepare temporary storage for distances and indices
            batch_distances = []
            batch_indices = []

            # Process `x_sample` in smaller batches
            for sample_start in range(0, x_sample.size(0), sample_batch_size):
                # print('this is first batch size', sample_start)
                sample_end = min(sample_start + sample_batch_size, x_sample.size(0))
                sample_batch = x_sample[sample_start:sample_end]

                # Compute pairwise distances between the query batch and `x_sample` batch
                distances = torch.cdist(batch, sample_batch, p=2)

                # Track distances and adjust indices for the chunk
                batch_distances.append(distances)
                batch_indices.append(
                    torch.arange(sample_start, sample_end, device=device).unsqueeze(0).repeat(batch.size(0), 1)
                )

            # Concatenate distances and indices across all `x_sample` batches
            batch_distances = torch.cat(batch_distances, dim=1)
            batch_indices = torch.cat(batch_indices, dim=1)

            # Get the `short_size` nearest neighbors
            _, topk_indices = torch.topk(batch_distances, k=short_size, largest=False, dim=1)

            # Map global indices
            topk_global_indices = torch.gather(batch_indices, 1, topk_indices)

            # Store results in the output array
            x0_train_index_initial[start_idx + batch_start:start_idx + batch_end, :] = topk_global_indices.cpu().numpy()

        if jj % 500 == 0:
            print('Find index iteration', jj, it_size_x0train)

    return x0_train_index_initial
        


def process_chunk(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim, batch_size=10000):
    """
    Optimized GPU version of process_chunk using FAISS.
    
    Args:
        batch_size: Number of query vectors to process at once (default: 10000)
    """
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)
    
    # Initialize GPU resources
    gpu = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(x_dim)  # Create a FAISS index for exact searches
    gpu_index = faiss.index_cpu_to_gpu(gpu, 0, index)
    
    # Convert to float32 numpy arrays for FAISS
    if isinstance(x_sample, torch.Tensor):
        x_sample_np = x_sample.cpu().numpy().astype('float32')
    else:
        x_sample_np = np.ascontiguousarray(x_sample.astype('float32'))
    
    if isinstance(x0_train, torch.Tensor):
        x0_train_np = x0_train.cpu().numpy().astype('float32')
    else:
        x0_train_np = np.ascontiguousarray(x0_train.astype('float32'))
    
    # Add all x_sample vectors to the index at once (more efficient)
    print(f'[INFO] Adding {len(x_sample_np)} vectors to FAISS index...')
    gpu_index.add(x_sample_np)
    print(f'[INFO] Index ready. Searching for nearest neighbors...')
    
    # Process queries in batches for better performance
    total_processed = 0
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train_np[start_idx:end_idx]
        chunk_size = end_idx - start_idx
        
        # Process chunk in batches if it's large
        if chunk_size > batch_size:
            for batch_start in range(0, chunk_size, batch_size):
                batch_end = min(batch_start + batch_size, chunk_size)
                batch = x0_train_chunk[batch_start:batch_end]
                
                # Perform the search
                _, index_initial = gpu_index.search(batch, short_size)
                x0_train_index_initial[start_idx + batch_start:start_idx + batch_end, :] = index_initial
                total_processed += batch_end - batch_start
                
                if total_processed % 10000 == 0:
                    print(f'[INFO] Processed {total_processed}/{train_size} queries...')
        else:
            # Process entire chunk at once if it's small
            _, index_initial = gpu_index.search(x0_train_chunk, short_size)
            x0_train_index_initial[start_idx:end_idx, :] = index_initial
            total_processed += chunk_size
            
            if jj % max(1, it_n_index // 10) == 0 or jj == it_n_index - 1:
                print(f'[INFO] Processed chunk {jj+1}/{it_n_index} ({total_processed}/{train_size} queries)')
    
    print(f'[INFO] Completed nearest neighbor search for {train_size} queries')
    
    # Cleanup resources
    del gpu_index
    del index
    del gpu
    
    return x0_train_index_initial

def select_time_points(total_time_steps: int, dt: float, num_points: int = 100):
    """
    Select a subset of time points with regular spacing.
    
    Args:
        total_time_steps (int): Total number of time steps available
        dt (float): Time step size
        num_points (int): Number of points to select
        
    Returns:
        tuple: (selected_indices, selected_times)
    """
    # If we want all time steps, return them directly
    if num_points >= total_time_steps:
        selected_indices = np.arange(total_time_steps)
        selected_times = selected_indices * dt
        return selected_indices, selected_times
    
    # Calculate total simulation time
    # Use total_time_steps * dt to include all time points from 0 to (total_time_steps - 1)
    # The last time point should be at (total_time_steps - 1) * dt
    total_time = (total_time_steps - 1) * dt
    
    # Create regularly spaced time points
    # Use num_points to get evenly spaced points including the last one
    selected_times = np.linspace(0, total_time, num_points)
    
    # Convert times to indices
    selected_indices = np.round(selected_times / dt).astype(int)
    
    # Ensure indices are within bounds and include the last time point
    selected_indices = np.clip(selected_indices, 0, total_time_steps - 1)
    
    # Ensure the last time point (index total_time_steps - 1) is included
    if selected_indices[-1] != total_time_steps - 1:
        selected_indices[-1] = total_time_steps - 1
        selected_times[-1] = (total_time_steps - 1) * dt
    
    # Remove duplicates while preserving order
    unique_indices = []
    unique_times = []
    for idx, time in zip(selected_indices, selected_times):
        if idx not in unique_indices:
            unique_indices.append(idx)
            unique_times.append(time)
    
    return np.array(unique_indices), np.array(unique_times)

def generate_euler_maruyama_residue(func, current_state, next_state, dt, data=None):
    """
    Generate Euler-Maruyama residuals from current_state and next_state.
    
    Args:
        func: Function that takes current state and returns derivative/force
        current_state: Current state array/tensor
                       - 1D case: (N, 1) or (N,)
                       - Multi-D case: (MC_samples, dim, time_steps) or (MC_samples, dim)
        next_state: Next state array/tensor with same shape as current_state
        dt: Time step size
    
    Returns:
        For 1D: residual (scaled residuals)
        For multi-D: residuals, u_current_reshaped, residual_cov_time
    """
    import numpy as np
    
    # Convert to numpy if needed
    if isinstance(current_state, torch.Tensor):
        current_state_np = current_state.cpu().numpy()
    else:
        current_state_np = current_state
        
    if isinstance(next_state, torch.Tensor):
        next_state_np = next_state.cpu().numpy()
    else:
        next_state_np = next_state
    
    # Determine if multi-dimensional case based on shape
    # Multi-D case: shape is (MC_samples, dim, time_steps) - 3D array with time dimension
    # 1D case: shape is (N, 1) or (N,) - 2D or 1D array
    # For multi-D with time: current_state[:, :, t] is state at time t, next_state[:, :, t] is state at time t+1
    if data is not None:
        dataset = data['dataset']
        # Multi-dimensional case with time dimension: dataset shape is (dim, time_steps_plus_1, MC_samples)
        dim, time_steps_plus_1, MC_samples = dataset.shape
        time_steps = time_steps_plus_1 - 1

        # Filter out trajectories with NaN values
        print(f"[INFO] Dataset shape: {dataset.shape}")
        print(f"[INFO] Checking for NaN values in trajectories...")
        
        # Find trajectories that contain any NaN values
        # dataset is (dim, time_steps_plus_1, MC_samples), so we check along axes (0, 1) to get (MC_samples,)
        nan_trajectories = np.any(np.isnan(dataset), axis=(0, 1))
        valid_trajectories = ~nan_trajectories
        
        print(f"[INFO] Found {np.sum(nan_trajectories)} trajectories with NaN values")
        print(f"[INFO] Using {np.sum(valid_trajectories)} valid trajectories")
        
        if np.sum(valid_trajectories) == 0:
            raise RuntimeError("No valid trajectories found! All trajectories contain NaN values.")
        
        # Filter the dataset to only include valid trajectories
        # dataset is (dim, time_steps_plus_1, MC_samples), so we filter along axis 2
        dataset = dataset[:, :, valid_trajectories]
        MC_samples = dataset.shape[2]
        print(f"[INFO] Filtered dataset shape: {dataset.shape}")
 
        # Initialize output arrays
        residuals = np.zeros((MC_samples, dim, time_steps))
        u_current_reshaped = np.zeros((MC_samples, dim, time_steps))
        
        # Process each time step individually to avoid memory issues
        for t in range(time_steps):
            if t % 10 == 0:
                print(f'Processing time step {t}/{time_steps}')
            
            # Extract current and next states for this time step
            # dataset is (dim, time_steps_plus_1, MC_samples)
            # dataset[:, t, :] gives (dim, MC_samples), transpose to (MC_samples, dim)
            u_current = dataset[:, t, :].T     # (MC_samples, dim)
            u_next = dataset[:, t+1, :].T       # (MC_samples, dim)
            
            # Store current state for output
            u_current_reshaped[:, :, t] = u_current
            
            # Euler prediction
            # Check if this is the learned_model_with_force_wrapper (for periodic_cascade)
            if isinstance(current_state, torch.Tensor):
                func_input = torch.from_numpy(u_current).float()
            else:
                func_input = u_current
            
            func_output = func(func_input)
            
            # Convert func_output to numpy
            if isinstance(func_output, torch.Tensor):
                func_output_np = func_output.cpu().numpy()
            else:
                func_output_np = func_output
            
            u_euler_pred = u_current + dt * func_output_np
            
            # Calculate residuals for this time step
            residuals[:, :, t] = u_next - u_euler_pred
        
        # Calculate residual covariance for each time step
        residual_cov_time = np.zeros((time_steps, dim))
        
        for t in range(time_steps):
            # Calculate standard deviations for each dimension
            for d in range(dim):
                std_d = np.std(residuals[:, d, t])
                # Calculate residual covariance
                residual_cov_time[t, d] = std_d / np.sqrt(dt)
            
            if t % 100 == 0:
                print(f"Time {t}: {residual_cov_time[t, :]}")
        
        print("Residual covariance shape:", residual_cov_time.shape)
        print("First time step covariance:", residual_cov_time[0, :])
        
        return residuals, u_current_reshaped, residual_cov_time
    
    else:        
        # Calculate residuals: (next_state - current_state - func(current_state) * dt)
        # For OU5d: dx = Bx dt + Σ dW
        # So: Σ dW = (next_state - current_state) - Bx dt
        # If learned drift f(x) ≈ Bx, then: residual ≈ Σ dW
        
        # Convert to torch tensor for func if needed (func might expect torch tensors)
        if isinstance(current_state, torch.Tensor):
            func_input = current_state
        else:
            func_input = torch.from_numpy(current_state_np).float()
        
        func_output = func(func_input)
        
        # Convert func_output to numpy
        if isinstance(func_output, torch.Tensor):
            func_output_np = func_output.cpu().numpy()
        else:
            func_output_np = func_output
        
        # Calculate residuals: residual = (next_state - current_state) - learned_drift * dt
        # This should equal Σ*dW if learned_drift = Bx (perfect learning)
        residual = (next_state_np - current_state_np - func_output_np * dt)
        
        # Debug: Check residual calculation for dimension 1 and dimension 5
        dim = current_state_np.shape[1]
        if dim == 1 or dim == 5:
            try:
                import sys
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                sys.path.append(os.path.join(project_root, 'Example'))
                from Example import params_init
                
                if dim == 5:  # OU5d
                    model_params = params_init(case_name='OU5d')
                    B = model_params['B']  # True drift matrix (5x5)
                    Sigma_base = model_params['Sigma']  # Base diffusion matrix (5x5) - without noise_level scaling
                    
                    # Compute true drift: B @ x for each sample
                    # current_state_np is (N, 5), B is (5, 5)
                    # For each sample i: true_drift[i] = B @ current_state_np[i]
                    true_drift = np.dot(current_state_np, B.T)  # (N, 5)
                    
                    # True residual: (next_state - current_state) - true_drift * dt
                    true_residual = (next_state_np - current_state_np - true_drift * dt)
                    
                    # Compute true residual std (this reflects the actual Sigma used in data generation)
                    true_residual_std = np.std(true_residual, axis=0)
                    
                    # Infer noise_level from true residual std
                    # Expected: true_residual_std = diag(Sigma_base * noise_level) * sqrt(dt)
                    # So: noise_level = true_residual_std / (diag(Sigma_base) * sqrt(dt))
                    Sigma_base_diag = np.diag(Sigma_base)
                    inferred_noise_level = true_residual_std / (Sigma_base_diag * np.sqrt(dt))
                    # Use mean of inferred noise_level (should be same for all dimensions if data is consistent)
                    noise_level_used = np.mean(inferred_noise_level)
                    Sigma_actual = Sigma_base * noise_level_used
                    
                    # Compare learned vs true residual
                    print(f"[DEBUG OU5d] Learned residual std: {np.std(residual, axis=0)}")
                    print(f"[DEBUG OU5d] True residual std: {true_residual_std}")
                    print(f"[DEBUG OU5d] Inferred noise_level from data: {noise_level_used:.4f}")
                    print(f"[DEBUG OU5d] Expected std (Sigma_base_diag * noise_level * sqrt(dt)): {np.diag(Sigma_actual) * np.sqrt(dt)}")
                    print(f"[DEBUG OU5d] Base expected std (Sigma_base_diag * sqrt(dt), noise_level=1.0): {Sigma_base_diag * np.sqrt(dt)}")
                    print(f"[DEBUG OU5d] Drift error std: {np.std(func_output_np - true_drift, axis=0)}")
                    
                elif dim == 1:  # 1D models (OU1d, DoubleWell1d, EXP1d, MM1d, etc.)
                    # Try to detect which 1D model by checking residual std
                    # For OU1d: sig = 0.3, so expected std = 0.3 * sqrt(dt)
                    # Try OU1d first (most common)
                    try:
                        model_params = params_init(case_name='OU1d')
                        th = model_params['th']  # theta
                        mu = model_params['mu']  # mu
                        sig = model_params['sig']  # sigma
                        
                        # Compute true drift: th * (mu - x)
                        # current_state_np is (N, 1) or (N,)
                        if current_state_np.ndim == 1:
                            x = current_state_np
                        else:
                            x = current_state_np[:, 0]
                        true_drift = th * (mu - x)  # (N,)
                        if current_state_np.ndim == 2:
                            true_drift = true_drift[:, np.newaxis]  # (N, 1)
                        
                        # True residual: (next_state - current_state) - true_drift * dt
                        true_residual = (next_state_np - current_state_np - true_drift * dt)
                        
                    except:
                        print(f"[DEBUG 1D] Could not compute true residual for 1D model")
                
            except Exception as e:
                print(f"[DEBUG] Could not compute true residual: {e}")
        
        return residual

def generate_second_step(current_state:np.ndarray,
                          residuals:np.ndarray,
                          scaler:np.ndarray,
                          dt:float,
                          train_size:int=10000,
                          device:str='cpu',
                          ODESOLVER_TIME_STEPS:int=2000,
                          num_time_points:int=None,
                          time_dependent: bool = False,
                          current_state_train:np.ndarray=None,
                          short_indx:np.ndarray=None,
                          save_short_indx_dir:str=None):
    """
    Generate second step ODE solution using residuals.
    
    Args:
        current_state: Current state array
                      - Time-independent: (size, dim)
                      - Time-dependent: (size, dim, time_steps)
        residuals: Residuals array
                   - Time-independent: (size, dim)
                   - Time-dependent: (size, dim, time_steps)
        scaler: Scaling factor array (dim,)
        dt: Time step size
        train_size: Number of training samples
        device: Device string ('cpu' or 'cuda')
        ODESOLVER_TIME_STEPS: Number of ODE solver time steps
        num_time_points: Number of time points to process (None = all)
        time_dependent: Whether the problem is time-dependent
    
    Returns:
        ODE_Solution: ODE solution array
        ZT_Solution: ZT solution array
    """
    odeslover_time_steps = ODESOLVER_TIME_STEPS
    size = int(residuals.shape[0])
    train_size = min(train_size, size)
    
    # Short index:
    short_size = 2048
    # Only OU5d uses it_size_x0train = 400, other models use train_size
    # Detect OU5d by checking if dimension is 5 (OU5d is the only 5D model)
    dim = int(residuals.shape[1]) if not time_dependent else int(residuals.shape[1])
    if dim == 5:
        it_size_x0train = 1000
    else:
        it_size_x0train = train_size
    it_n_index = train_size // it_size_x0train
    
    # Batch processing parameters
    it_size = min(train_size, 60000)
    it_n = int(train_size / it_size)
    
    if not time_dependent:
        # Time-independent case: residuals shape is (size, dim)
        dim = int(residuals.shape[1])
        
        # Initialize output arrays (2D)
        ODE_Solution = np.zeros((train_size, dim))
        ZT_Solution = np.random.randn(train_size, dim)
        
        # Debug: Show scaler values
        print(f"Scaler values: {scaler}")
        print(f"Using train_size: {train_size} out of total size: {size}")
        print(f"Dimension: {dim}")
        
        # For time-independent, current_state should be (size, dim)
        current_state_sample = current_state  # (size, dim)
        # Use provided current_state_train if available, otherwise use first train_size samples
        if current_state_train is not None:
            current_state_train = current_state_train  # Use provided training data
        else:
            current_state_train = current_state[:train_size]  # (train_size, dim)
        
        # Find nearest neighbors - use provided short_indx if available, otherwise compute it
        if short_indx is not None:
            print('[INFO] Using provided short_indx, shape:', short_indx.shape)
            current_state_short = current_state_sample[short_indx]  # (train_size, short_size, dim)
        else:
            print('[INFO] Computing short_indx...')
            if torch.cuda.is_available():
                short_indx = process_chunk(it_n_index, it_size_x0train, short_size, 
                                            current_state_sample, current_state_train, 
                                            train_size, current_state.shape[1])
                print('short indx is', short_indx.shape)
                current_state_short = current_state_sample[short_indx]  # (train_size, short_size, dim)
            else:
                short_indx = process_chunk_faiss_cpu(it_n_index, it_size_x0train, short_size, 
                                                    current_state_sample, current_state_train, 
                                                    train_size, current_state.shape[1])
                print('short indx is', short_indx.shape)
                current_state_short = current_state_sample[short_indx]  # (train_size, short_size, dim)
        
        # Scale residuals
        scaled_residuals = residuals * scaler  # (size, dim)
        z_short = scaled_residuals[short_indx]  # (train_size, short_size, dim)
        # Debug: Show scaled residual std (from full dataset)
        actual_scaled_std_full = np.std(scaled_residuals, axis=0)
        print(f"Scaled residual std (full dataset): {actual_scaled_std_full}")
        
        # Also compute from training data only
        scaled_residuals_train = scaled_residuals[:train_size]  # (train_size, dim)
        actual_scaled_std_train = np.std(scaled_residuals_train, axis=0)
        print(f"Scaled residual std (training data only): {actual_scaled_std_train}")
        
        # For OU5d: compute expected scaled residual std based on true Sigma matrix
        # Expected: Sigma_diag * sqrt(dt) * DIFF_SCALE
        # Check if this is OU5d by checking dimension and trying to import params_init
        if dim == 5:
            try:
                import sys
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                sys.path.append(os.path.join(project_root, 'Example'))
                from Example import params_init
                model_params = params_init(case_name='OU5d')
                Sigma_diag = np.diag(model_params['Sigma'])  # [0.8, 0.6, 0.9, 0.4, 0.5]
                expected_residual_std = Sigma_diag * np.sqrt(dt)  # Should be [0.08, 0.06, 0.09, 0.04, 0.05]
                expected_scaled_std = expected_residual_std * scaler[0]  # Should be [8, 6, 9, 4, 5] if DIFF_SCALE=100
                print(f"Expected scaled residual std (Sigma_diag * sqrt(dt) * DIFF_SCALE): {expected_scaled_std}")
                print(f"Difference (training - expected): {actual_scaled_std_train - expected_scaled_std}")
                print(f"Difference (full - expected): {actual_scaled_std_full - expected_scaled_std}")
            except Exception as e:
                print(f"[INFO] Could not compute expected values: {e}")
        
        # Process in mini-batches
        for jj in range(it_n):
            start_idx = jj * it_size
            end_idx = min((jj + 1) * it_size, size)
            print(f'start_idx is {start_idx}; end_idx is {end_idx}')
            
            # Extract mini-batch
            it_residuals = scaled_residuals[start_idx:end_idx]  # (batch_size, dim)
            
            # Generate random noise for this batch
            z_T = ZT_Solution[start_idx:end_idx, :]  # (batch_size, dim)
            
            # Convert to tensors
            it_zt = torch.tensor(z_T, dtype=torch.float32).to(device)
            it_x0 = torch.tensor(current_state_sample[start_idx:end_idx], dtype=torch.float32).to(device)
            x_mini_batch = current_state_short[start_idx:end_idx]  # (batch_size, short_size, dim)
            z_mini_batch = z_short[start_idx:end_idx]  # (batch_size, short_size, dim)
            x_mini_batch_tensor = torch.tensor(x_mini_batch, dtype=torch.float32).to(device)
            z_mini_batch_tensor = torch.tensor(z_mini_batch, dtype=torch.float32).to(device)
            
            # Call ODE solver for this mini-batch
            y_temp = ODE_solver(it_zt, x_mini_batch_tensor, z_mini_batch_tensor, it_x0, odeslover_time_steps)
            
            # Store results
            ODE_Solution[start_idx:end_idx, :] = y_temp.cpu().detach().numpy()
            if jj % 100 == 0:
                print(f'this is {jj+1} times which has already done.')
        print(f'Time-independent case completed.')
    
    else:
        # Time-dependent case: residuals shape is (size, dim, time_steps)
        total_time_steps = residuals.shape[2]
        dim = int(residuals.shape[1])
        
        # Select time points to process
        if num_time_points is not None and num_time_points < total_time_steps:
            # Only use select_time_points if we want fewer than all time steps
            selected_indices, selected_times = select_time_points(
                total_time_steps, dt, num_time_points
            )
            print(f"Processing {len(selected_indices)} time points out of {total_time_steps} total")
            print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
            time_indices = selected_indices
            time_step = len(selected_indices)
        else:
            # Use all time steps if num_time_points is None or >= total_time_steps
            time_indices = range(total_time_steps)
            selected_times = np.arange(total_time_steps) * dt
            time_step = total_time_steps
            print(f"Processing all {total_time_steps} time points")
            print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        
        # Initialize output arrays (3D)
        ODE_Solution = np.zeros((size, dim, time_step))
        ZT_Solution = np.zeros((size, dim, time_step))
        
        # Debug: Show scaler values
        print(f"Scaler values: {scaler}")
        print(f"Using train_size: {train_size} out of total size: {size}")
        print(f"Dimension: {dim}, Time steps: {time_step}")
        
        for t_idx, t in enumerate(time_indices):
            print('-'.center(100, '-'))
            print(f'this is {t_idx+1} times / overall {time_step} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
            
            # Print residual std for each dimension
            residual_std = [np.std(residuals[:, d, t]) / np.sqrt(dt) for d in range(dim)]
            print(f"Residual std (scaled by sqrt(dt)): {residual_std}")
            print('-'.center(100, '-'))
            
            # Extract current state for this time step
            # current_state shape: (size, dim, time_steps)
            current_state_sample = current_state[:, :, t]  # (size, dim)
            current_state_train = current_state[:train_size, :, t]  # (train_size, dim)
            
            # Find nearest neighbors - check if saved short_indx exists, otherwise compute it
            short_indx_file = None
            if save_short_indx_dir is not None:
                short_indx_dir = os.path.join(save_short_indx_dir, 'short_indx_time')
                os.makedirs(short_indx_dir, exist_ok=True)
                short_indx_file = os.path.join(short_indx_dir, f'indx_{t}.npy')
            
            if short_indx_file is not None and os.path.exists(short_indx_file):
                print(f'[INFO] Loading short_indx for time step {t} from {short_indx_file}')
                short_indx = np.load(short_indx_file)
                print(f'[INFO] Loaded short_indx shape: {short_indx.shape}')
            else:
                print(f'[INFO] Computing short_indx for time step {t}...')
                short_indx = process_chunk_faiss_cpu(it_n_index, it_size_x0train, short_size, 
                                                    current_state_sample, current_state_train, 
                                                    train_size, current_state.shape[1])
                print(f'[INFO] Computed short_indx shape: {short_indx.shape}')
                # Save short_indx if save directory is provided
                if short_indx_file is not None:
                    np.save(short_indx_file, short_indx)
                    print(f'[INFO] Saved short_indx for time step {t} to {short_indx_file}')
            
            current_state_short = current_state_sample[short_indx]  # (train_size, short_size, dim)
            
            # Scale residuals for this time step
            scaled_residuals = residuals[:, :, t] * scaler  # (size, dim)
            z_short = scaled_residuals[short_indx]  # (train_size, short_size, dim)
            
            # Initialize ZT for this time step
            ZT_Solution[:, :, t_idx] = np.random.randn(size, dim)
            
            # Debug: Show scaled residual std
            print(f"Scaled residual std at t={t}: {np.std(scaled_residuals, axis=0)}")
            
            # Process in mini-batches
            for jj in range(it_n):
                start_idx = jj * it_size
                end_idx = min((jj + 1) * it_size, size)
                print(f'start_idx is {start_idx}; end_idx is {end_idx}')
                
                # Extract mini-batch
                it_residuals = scaled_residuals[start_idx:end_idx]  # (batch_size, dim)
                
                # Generate random noise for this batch
                z_T = ZT_Solution[start_idx:end_idx, :, t_idx]  # (batch_size, dim)
                
                # Convert to tensors
                it_zt = torch.tensor(z_T, dtype=torch.float32).to(device)
                it_x0 = torch.tensor(current_state_sample[start_idx:end_idx], dtype=torch.float32).to(device)
                
                # Get corresponding short indices for this batch
                batch_short_indx = short_indx[start_idx:end_idx]  # (batch_size, short_size)
                x_mini_batch = torch.tensor(current_state_sample[batch_short_indx], dtype=torch.float32).to(device)
                z_mini_batch = torch.tensor(scaled_residuals[batch_short_indx], dtype=torch.float32).to(device)
                
                # Call ODE solver for this mini-batch
                y_temp = ODE_solver(it_zt, x_mini_batch, z_mini_batch, it_x0, odeslover_time_steps)
                
                # Store results
                ODE_Solution[start_idx:end_idx, :, t_idx] = y_temp.cpu().detach().numpy()
            
            print(f'this is {t_idx+1} times which has already done.')
    
    return ODE_Solution, ZT_Solution

def generate_mean_and_std(ODE_Solution:np.ndarray):
    """
    Generate mean and standard deviation from ODE solution.
    
    Args:
        ODE_Solution: ODE solution array
                     - Time-independent: (size, dim) - 2D
                     - Time-dependent: (size, dim, time_steps) - 3D
    
    Returns:
        mean_value: Mean values array
                   - Time-independent: (1, dim)
                   - Time-dependent: (time_steps, dim)
        std_value: Standard deviation values array
                  - Time-independent: (1, dim)
                  - Time-dependent: (time_steps, dim)
    """
    if ODE_Solution.ndim == 2:
        # Time-independent case: (size, dim)
        size, dim = ODE_Solution.shape
        mean_value = np.zeros((1, dim))
        std_value = np.zeros((1, dim))
        
        for d in range(dim):
            dim_data = ODE_Solution[:, d]
            mean_value[0, d] = np.mean(dim_data)
            std_value[0, d] = np.std(dim_data)
    elif ODE_Solution.ndim == 3:
        # Time-dependent case: (size, dim, time_steps)
        size, dim, time_steps = ODE_Solution.shape
        mean_value = np.zeros((time_steps, dim))
        std_value = np.zeros((time_steps, dim))
        
        for t in range(time_steps):
            for d in range(dim):
                dim_data = ODE_Solution[:, d, t]
                mean_value[t, d] = np.mean(dim_data)
                std_value[t, d] = np.std(dim_data)
    else:
        raise ValueError(f"ODE_Solution must be 2D or 3D, got {ODE_Solution.ndim}D")
    
    return mean_value, std_value
