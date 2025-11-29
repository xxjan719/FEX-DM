"""
Plotting utilities for FEX-DM
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde
from .ODEParser import FN_Net
from .FEX import FEX_model_learned

# Import params_init from Example
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'Example'))
try:
    from Example import params_init
except ImportError:
    # Fallback if import fails
    params_init = None


def plot_training_data_histogram(current_state_train, 
                                 save_path, 
                                 model_name='OU1d',
                                 train_size=None,
                                 noise_level=None,
                                 bins=50,
                                 figsize=(8, 6),
                                 dpi=300):
    """
    Plot and save histogram of training data distribution.
    
    Args:
        current_state_train: numpy array of current state values for training
        save_path: directory path where the figure should be saved
        model_name: name of the model (default: 'OU1d')
        train_size: number of training samples (for title)
        noise_level: noise level used (for title and filename)
        bins: number of bins for histogram (default: 50)
        figsize: figure size tuple (default: (8, 6))
        dpi: resolution for saved figure (default: 300)
    
    Returns:
        str: path to the saved figure file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate filename
    if noise_level is not None:
        filename = f'{model_name}_training_data_histogram_noise_{noise_level}.pdf'
    else:
        filename = f'{model_name}_training_data_histogram.pdf'
    
    histogram_path = os.path.join(save_path, filename)
    
    # Create figure
    plt.figure(figsize=figsize)
    plt.hist(current_state_train.flatten(), bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Current State Value')
    plt.ylabel('Frequency')
    
    # Create title
    title_parts = [f'{model_name} Training Data Distribution']
    if train_size is not None:
        title_parts.append(f'Samples: {train_size}')
    if noise_level is not None:
        title_parts.append(f'Noise Level: {noise_level}')
    plt.title(', '.join(title_parts))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(histogram_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Histogram saved to: {histogram_path}")
    
    
    # return histogram_path


def plot_trajectory_comparison_simulation(second_stage_dir_FEX,
                                All_stage_dir_TF_CDM,
                                model_name='OU1d',
                                noise_level=1.0,
                                device='cpu',
                                initial_values=None,
                                sde_params=None,
                                save_dir=None,
                                figsize=(18, 6),
                                dpi=300):
    """
    Plot comparison between FEX-DM and TF-CDM (optional) models for SDE simulation.
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        second_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        model_name: Model name (e.g., 'OU1d')
        params_name: Params name (defaults to model_name)
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        initial_values: List of initial values to test (default: [-6, 1.5, 6])
        sde_params: Dictionary with SDE parameters (mu, sigma, theta, sde_T, sde_dt)
                    If None, uses default OU1d parameters
        save_dir: Directory to save the figure (default: second_stage_dir_FEX)
        figsize: Figure size tuple (default: (18, 6))
        dpi: Resolution for saved figure (default: 300)
    
    Returns:
        str: Path to the saved figure file
    """
    
    # Load SDE parameters from model parameters
    if sde_params is None:
        # Load parameters from model_name
        model_params = params_init(case_name=model_name)
        sde_params = {
            'mu': model_params['mu'],
            'sigma': model_params['sig'],  # Note: 'sig' in params, 'sigma' in sde_params
            'theta': model_params['th'],    # Note: 'th' in params, 'theta' in sde_params
            'sde_T': model_params['T'],
            'sde_dt': model_params['Dt']
        }
    
    if initial_values is None:
        initial_values = [-6, 1.5, 6]
    
    if save_dir is None:
        # Create plot folder in the same parent directory as second_stage_dir_FEX
        parent_dir = os.path.dirname(second_stage_dir_FEX)
        save_dir = os.path.join(parent_dir, 'plot')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Model styles
    model_styles = {
        "FEX-DM": {
            "color": "orange",
            "fill": "red",
            "linestyle": "-"
        }
    }
    
    if All_stage_dir_TF_CDM is not None:
        model_styles["TF-CDM"] = {
            "color": "steelblue",
            "fill": "blue",
            "linestyle": "-"
        }
    
    # Load FEX-DM model and parameters
    print("[INFO] Loading FEX-DM model and parameters...")
    data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
    if not os.path.exists(data_inf_path_FEX):
        raise FileNotFoundError(f"FEX-DM data_inf.pt not found at {data_inf_path_FEX}")

    data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
    
    xTrain_mean_FEX = data_inf_FEX['ZT_Train_mean'].to(device)
    xTrain_std_FEX = data_inf_FEX['ZT_Train_std'].to(device)
    yTrain_mean_FEX = data_inf_FEX['ODE_Train_mean'].to(device)
    yTrain_std_FEX = data_inf_FEX['ODE_Train_std'].to(device)
    diff_scale_FEX = data_inf_FEX['diff_scale']
    
    # Load FEX-DM FN_Net model
    FNET_path_FEX = os.path.join(second_stage_dir_FEX, 'FNET.pth')
    if not os.path.exists(FNET_path_FEX):
        raise FileNotFoundError(f"FEX-DM FNET.pth not found at {FNET_path_FEX}")

    # Determine dimension from data
    dimension = data_inf_FEX['ZT_Train_new'].shape[1]
    FN_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
    FN_FEX.load_state_dict(torch.load(FNET_path_FEX, map_location=device))
    FN_FEX.eval()
    
    # Load TF-CDM model and parameters if provided
    FN_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = None

    if All_stage_dir_TF_CDM is not None:
        print("[INFO] Loading TF-CDM model and parameters...")
        data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
        if not os.path.exists(data_inf_path_TF_CDM):
            print(f"[WARNING] TF-CDM data_inf.pt not found at {data_inf_path_TF_CDM}, skipping TF-CDM")
        else:
            data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
            xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
            xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
            yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
            yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
            diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
            
            # Load TF-CDM FN_Net model
            FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
            if os.path.exists(FNET_path_TF_CDM):
                FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                FN_TF_CDM.eval()
            else:
                print(f"[WARNING] TF-CDM FNET.pth not found at {FNET_path_TF_CDM}, skipping TF-CDM")
                model_styles.pop("TF-CDM", None)
                FN_TF_CDM = None
    
    # Create FEX function wrapper
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device)
    
    # Extract SDE parameters
    mu = sde_params['mu']
    sigma = sde_params['sigma']
    theta = sde_params['theta']
    sde_T = sde_params['sde_T']
    sde_dt = sde_params['sde_dt']
    
    x_dim = dimension
    ode_time_steps = int(sde_T / sde_dt)
    Npath = 500000
    
    def run_simulation(true_init, ax, title):
        """Run the simulation for a given initial value, comparing both FEX-DM and TF-CDM."""
        ode_mean_pred = {model: np.zeros(ode_time_steps) for model in model_styles}
        ode_std_pred = {model: np.zeros(ode_time_steps) for model in model_styles}
        x_pred_new_dict = {model: torch.clone((true_init * torch.ones(Npath, x_dim)).to(device)) 
                          for model in model_styles}
        
        ode_mean_true = np.zeros(ode_time_steps)
        ode_std_true = np.zeros(ode_time_steps)
        ode_path_true = true_init * np.ones((Npath, x_dim))
        
        for jj in range(ode_time_steps):
            # Generate the same random noise z for both models to ensure fair comparison
            z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
            
            for model in model_styles:
                # Skip TF-CDM for x0 = -6 and x0 = 6
                if model == "TF-CDM" and (abs(true_init - (-6)) < 0.01 or abs(true_init - 6) < 0.01):
                    continue  # Skip TF-CDM for these initial values
                
                x_pred_new = x_pred_new_dict[model]
                
                if model == "FEX-DM":
                    with torch.no_grad():
                        prediction = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
                        prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                
                elif model == "TF-CDM" and FN_TF_CDM is not None:
                    with torch.no_grad():
                        prediction = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                        prediction = (prediction / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                
                ode_mean_pred[model][jj] = np.mean(prediction)
                ode_std_pred[model][jj] = np.std(prediction)
                
                # Update each model's state separately
                x_pred_new_dict[model] = torch.tensor(prediction).to(device, dtype=torch.float32)
            
            # True trajectory evolution
            ode_path_true = ode_path_true + theta * (mu - ode_path_true) * sde_dt + \
                           sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
            ode_mean_true[jj] = np.mean(ode_path_true)
            ode_std_true[jj] = np.std(ode_path_true)
        
        # Time axis
        tmesh = np.linspace(sde_dt, ode_time_steps * sde_dt, ode_time_steps)
        
        # Plot
        ax.plot(tmesh, ode_mean_true, linewidth=4, label="Mean of ground truth", 
                color='black', linestyle=':')
        
        for model, style in model_styles.items():
            # Skip TF-CDM plotting for x0 = -6 and x0 = 6
            if model == "TF-CDM" and (abs(true_init - (-6)) < 0.01 or abs(true_init - 6) < 0.01):
                continue  # Skip TF-CDM plotting for these initial values
            
            ax.plot(
                tmesh,
                ode_mean_pred[model],
                label=f"Pred Mean ({model})",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2
            )
            ax.fill_between(
                tmesh,
                ode_mean_pred[model] - ode_std_pred[model],
                ode_mean_pred[model] + ode_std_pred[model],
                color=style["fill"],
                alpha=0.2
            )
        
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel('Value', fontsize=24)
        ax.set_title(f'{title} $x_0$ = {true_init:.2f}', fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
    
    # Create figure
    fig, axes = plt.subplots(1, len(initial_values), figsize=figsize)
    if len(initial_values) == 1:
        axes = [axes]
    
    for col, x0 in enumerate(initial_values):
        run_simulation(x0, axes[col], "")
    
    # Create legend
    legend_handles = [
        plt.Line2D([0], [0], color='black', linestyle=':', linewidth=4, label='Mean of ground truth'),
        plt.Line2D([0], [0], color='orange', linestyle='-', linewidth=3, label='Pred Mean (FEX-DM)')
    ]
    
    if "TF-CDM" in model_styles:
        legend_handles.append(
            plt.Line2D([0], [0], color='steelblue', linestyle='-', linewidth=3, label='Pred Mean (TF-CDM)')
        )
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=len(legend_handles), fontsize=18)
    
    # Save and show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, 'final_image_1row_3plots.pdf')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    # Verify the file was created
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    else:
        print(f"[WARNING] File was not created at: {save_path}")
    
    
    
    return save_path


def plot_drift_and_diffusion(second_stage_dir_FEX,
                             All_stage_dir_TF_CDM=None,
                             model_name='OU1d',
                             noise_level=1.0,
                             device='cpu',
                             Npath=500000,
                             N_x0=500,
                             x_min=-6,
                             x_max=6,
                             save_dir=None,
                             figsize=(15, 6),
                             dpi=300):
    """
    Plot drift (μ(x)) and diffusion (σ(x)) coefficients for FEX-DM and TF-CDM models.
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        model_name: Model name (e.g., 'OU1d')
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        Npath: Number of paths for Monte Carlo simulation (default: 500000)
        N_x0: Number of initial values in grid (default: 500)
        x_min: Minimum initial value (default: -6)
        x_max: Maximum initial value (default: 6)
        save_dir: Directory to save the figure (default: parent of second_stage_dir_FEX/plot)
        figsize: Figure size tuple (default: (15, 6))
        dpi: Resolution for saved figure (default: 300)
    
    Returns:
        str: Path to the saved figure file
    """
    # Load SDE parameters from model parameters
    if params_init is not None:
        model_params = params_init(case_name=model_name)
        sigma_base = model_params['sig']
        sde_params = {
            'mu': model_params['mu'],
            'sigma': sigma_base * noise_level,
            'theta': model_params['th'],
            'sde_T': model_params['T'],
            'sde_dt': model_params['Dt']
        }
    else:
        raise ValueError("params_init is not available")
    
    mu = sde_params['mu']
    sigma = sde_params['sigma']
    theta = sde_params['theta']
    sde_dt = sde_params['sde_dt']
    
    if save_dir is None:
        parent_dir = os.path.dirname(second_stage_dir_FEX)
        save_dir = os.path.join(parent_dir, 'plot')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load FEX-DM model and parameters
    print("[INFO] Loading FEX-DM model and parameters...")
    data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
    if not os.path.exists(data_inf_path_FEX):
        raise FileNotFoundError(f"FEX-DM data_inf.pt not found at {data_inf_path_FEX}")
    
    data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
    xTrain_mean_FEX = data_inf_FEX['ZT_Train_mean'].to(device)
    xTrain_std_FEX = data_inf_FEX['ZT_Train_std'].to(device)
    yTrain_mean_FEX = data_inf_FEX['ODE_Train_mean'].to(device)
    yTrain_std_FEX = data_inf_FEX['ODE_Train_std'].to(device)
    diff_scale_FEX = data_inf_FEX['diff_scale']
    
    # Load FEX-DM FN_Net model
    FNET_path_FEX = os.path.join(second_stage_dir_FEX, 'FNET.pth')
    if not os.path.exists(FNET_path_FEX):
        raise FileNotFoundError(f"FEX-DM FNET.pth not found at {FNET_path_FEX}")
    
    dimension = data_inf_FEX['ZT_Train_new'].shape[1]
    FN_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
    FN_FEX.load_state_dict(torch.load(FNET_path_FEX, map_location=device))
    FN_FEX.eval()
    
    # Load TF-CDM model and parameters if provided
    FN_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = None
    
    if All_stage_dir_TF_CDM is not None:
        print("[INFO] Loading TF-CDM model and parameters...")
        data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
        if not os.path.exists(data_inf_path_TF_CDM):
            print(f"[WARNING] TF-CDM data_inf.pt not found at {data_inf_path_TF_CDM}, skipping TF-CDM")
        else:
            data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
            xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
            xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
            yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
            yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
            diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
            
            # Load TF-CDM FN_Net model
            FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
            if os.path.exists(FNET_path_TF_CDM):
                FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                FN_TF_CDM.eval()
            else:
                print(f"[WARNING] TF-CDM FNET.pth not found at {FNET_path_TF_CDM}, skipping TF-CDM")
                FN_TF_CDM = None
    
    # Create FEX function wrapper
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device)
    
    x_dim = dimension
    x0_grid = np.linspace(x_min, x_max, N_x0)
    
    # Initialize arrays for drift and diffusion
    bx_pred_FEX = np.zeros(N_x0)
    sigmax_pred_FEX = np.zeros(N_x0)
    bx_pred_TF_CDM = np.zeros(N_x0)
    sigmax_pred_TF_CDM = np.zeros(N_x0)
    
    # True drift and diffusion (for OU process: dX = theta*(mu - X)dt + sigma*dB)
    bx_true = theta * (mu - x0_grid)  # Drift: theta*(mu - x)
    sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sigma
    
    print(f"[INFO] Computing drift and diffusion for {N_x0} initial values...")
    for jj in range(N_x0):
        if jj % 50 == 0:
            print(f"[INFO] Processing {jj+1}/{N_x0}...")
        
        true_init = x0_grid[jj]
        x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
        
        # Generate the same random noise for both models
        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
        
        # FEX-DM Prediction
        with torch.no_grad():
            prediction_FEX = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
            prediction_FEX = (prediction_FEX / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
        
        # Compute drift: mean((prediction - x0) / dt)
        bx_pred_FEX[jj] = np.mean((prediction_FEX - true_init) / sde_dt)
        # Compute diffusion: std((prediction - x0 - bx*dt)) * sqrt(1/dt)
        sigmax_pred_FEX[jj] = np.std((prediction_FEX - true_init - bx_pred_FEX[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
        
        # TF-CDM Prediction
        if FN_TF_CDM is not None:
            with torch.no_grad():
                prediction_TF_CDM = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                prediction_TF_CDM = (prediction_TF_CDM / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
            
            # Compute drift and diffusion
            bx_pred_TF_CDM[jj] = np.mean((prediction_TF_CDM - true_init) / sde_dt)
            sigmax_pred_TF_CDM[jj] = np.std((prediction_TF_CDM - true_init - bx_pred_TF_CDM[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
    
    # Calculate relative errors in training domain (0 to 2.5)
    training_mask = (x0_grid >= 0) & (x0_grid <= 2.5)
    bx_true_training = bx_true[training_mask]
    sigmax_true_training = sigmax_true[training_mask]
    bx_pred_FEX_training = bx_pred_FEX[training_mask]
    sigmax_pred_FEX_training = sigmax_pred_FEX[training_mask]
    
    # Relative error for FEX-DM: |pred - true| / |true|
    # Use absolute value of true to avoid division by zero issues
    bx_rel_error_FEX = np.mean(np.abs(bx_pred_FEX_training - bx_true_training) / (np.abs(bx_true_training) + 1e-10))
    sigmax_rel_error_FEX = np.mean(np.abs(sigmax_pred_FEX_training - sigmax_true_training) / (np.abs(sigmax_true_training) + 1e-10))
    
    # Print relative error table
    print("\n" + "="*80)
    print("RELATIVE ERROR TABLE (Training Domain: x ∈ [0, 2.5])")
    print("="*80)
    print(f"{'Model':<15} {'Drift Error':<20} {'Diffusion Error':<20}")
    print("-"*80)
    print(f"{'FEX-DM':<15} {bx_rel_error_FEX:<20.6e} {sigmax_rel_error_FEX:<20.6e}")
    
    if FN_TF_CDM is not None:
        bx_pred_TF_CDM_training = bx_pred_TF_CDM[training_mask]
        sigmax_pred_TF_CDM_training = sigmax_pred_TF_CDM[training_mask]
        bx_rel_error_TF_CDM = np.mean(np.abs(bx_pred_TF_CDM_training - bx_true_training) / (np.abs(bx_true_training) + 1e-10))
        sigmax_rel_error_TF_CDM = np.mean(np.abs(sigmax_pred_TF_CDM_training - sigmax_true_training) / (np.abs(sigmax_true_training) + 1e-10))
        print(f"{'TF-CDM':<15} {bx_rel_error_TF_CDM:<20.6e} {sigmax_rel_error_TF_CDM:<20.6e}")
    
    print("="*80 + "\n")
    
    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.4)
    
    # Color & Style Setup
    colors = {'FEX-DM': 'orange', 'TF-CDM': 'steelblue', 'Ground-Truth': 'black'}
    linestyles = {'FEX-DM': '-', 'TF-CDM': '--', 'Ground-Truth': ':'}
    markers = {'FEX-DM': 'o', 'TF-CDM': 's'}
    
    # Drift Plot (μ(x))
    ax[0].plot(x0_grid, bx_pred_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
               color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5)
    
    # Plot TF-CDM only within training domain (0 to 2.5)
    if FN_TF_CDM is not None:
        training_mask = (x0_grid >= 0) & (x0_grid <= 2.5)
        x0_training = x0_grid[training_mask]
        bx_pred_TF_CDM_training = bx_pred_TF_CDM[training_mask]
        ax[0].plot(x0_training, bx_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                   color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2)
    
    ax[0].plot(x0_grid, bx_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
               color=colors['Ground-Truth'], linewidth=2)
    
    ax[0].axvspan(0, 2.5, color='gray', alpha=0.2, label="Training Domain")
    ax[0].axvline(0, color='gray', linestyle='--', linewidth=2)
    ax[0].axvline(2.5, color='gray', linestyle='--', linewidth=2)
    
    ax[0].set_xlabel('$x$', fontsize=30)
    ax[0].set_ylabel('$\\hat{\\mu}(x)$', fontsize=30)
    ax[0].tick_params(axis='both', labelsize=25)
    ax[0].set_xticks([-6, 0, 6, 2.5])
    
    # Diffusion Plot (σ(x))
    ax[1].plot(x0_grid, sigmax_pred_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
               color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5)
    
    # Plot TF-CDM only within training domain (0 to 2.5)
    if FN_TF_CDM is not None:
        training_mask = (x0_grid >= 0) & (x0_grid <= 2.5)
        x0_training = x0_grid[training_mask]
        sigmax_pred_TF_CDM_training = sigmax_pred_TF_CDM[training_mask]
        ax[1].plot(x0_training, sigmax_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                   color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2)
    
    ax[1].plot(x0_grid, sigmax_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
               color=colors['Ground-Truth'], linewidth=2)
    
    ax[1].axvspan(0, 2.5, color='gray', alpha=0.2, label="Training Domain")
    ax[1].axvline(0, color='gray', linestyle='--', linewidth=2)
    ax[1].axvline(2.5, color='gray', linestyle='--', linewidth=2)
    
    ax[1].set_xlabel('$x$', fontsize=30)
    ax[1].set_ylabel('$\\hat{\\sigma}(x)$', fontsize=30)
    ax[1].tick_params(axis='both', labelsize=25)
    ax[1].set_xticks([-6, 0, 6, 2.5])
    ax[1].set_ylim([0.1, 0.45])
    
    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=22, frameon=True, 
               ncol=4, bbox_to_anchor=(0.5, 1.05))
    
    # Save and show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, 'drift_and_diffusion_improved.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    else:
        print(f"[WARNING] File was not created at: {save_path}")
    
    
    return save_path


def plot_conditional_distribution(second_stage_dir_FEX,
                                  All_stage_dir_TF_CDM=None,
                                  model_name='OU1d',
                                  noise_level=1.0,
                                  device='cpu',
                                  initial_values=None,
                                  sde_params=None,
                                  save_dir=None,
                                  figsize=(18, 6),
                                  dpi=300):
    """
    Plot conditional distribution comparison between FEX-DM and TF-CDM (optional) models.
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        model_name: Model name (e.g., 'OU1d')
        noise_level: Noise level (default: 1.0)
        device: Device to use ('cpu' or 'cuda')
        initial_values: List of initial values to plot (default: [-6, 1.5, 6])
        sde_params: Dictionary of SDE parameters (mu, sigma, theta). If None, will try to load from params_init
        save_dir: Directory to save the plot (default: plot folder in parent of second_stage_dir_FEX)
        figsize: Figure size tuple (default: (18, 6))
        dpi: Resolution for saved figure (default: 300)
    
    Returns:
        str: Path to the saved figure file
    """
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Set default initial values
    if initial_values is None:
        initial_values = [-6, 1.5, 6]
    
    # Set default save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(second_stage_dir_FEX), 'plot')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load SDE parameters
    if sde_params is None:
        if params_init is not None:
            try:
                model_params = params_init(model_name=model_name, noise_level=noise_level)
                mu = model_params.get('mu', 1.2)
                sigma = model_params.get('sigma', 0.3)
                theta = model_params.get('theta', 1.0)
            except Exception as e:
                print(f"[WARNING] Could not load params from params_init: {e}")
                print("[INFO] Using default SDE parameters")
                mu = 1.2
                sigma = 0.3
                theta = 1.0
        else:
            print("[INFO] params_init not available, using default SDE parameters")
            mu = 1.2
            sigma = 0.3
            theta = 1.0
    else:
        mu = sde_params.get('mu', 1.2)
        sigma = sde_params.get('sigma', 0.3)
        theta = sde_params.get('theta', 1.0)
    
    # Load FEX-DM model and parameters
    print("[INFO] Loading FEX-DM model and parameters...")
    data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
    if not os.path.exists(data_inf_path_FEX):
        raise FileNotFoundError(f"FEX-DM data_inf.pt not found at {data_inf_path_FEX}")
    
    data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
    xTrain_mean_FEX = data_inf_FEX['ZT_Train_mean'].to(device)
    xTrain_std_FEX = data_inf_FEX['ZT_Train_std'].to(device)
    yTrain_mean_FEX = data_inf_FEX['ODE_Train_mean'].to(device)
    yTrain_std_FEX = data_inf_FEX['ODE_Train_std'].to(device)
    diff_scale_FEX = data_inf_FEX['diff_scale']
    
    # Load FEX-DM FN_Net model
    FNET_path_FEX = os.path.join(second_stage_dir_FEX, 'FNET.pth')
    if not os.path.exists(FNET_path_FEX):
        raise FileNotFoundError(f"FEX-DM FNET.pth not found at {FNET_path_FEX}")
    
    dimension = data_inf_FEX['ZT_Train_new'].shape[1]
    FN_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
    FN_FEX.load_state_dict(torch.load(FNET_path_FEX, map_location=device))
    FN_FEX.eval()
    
    # Load TF-CDM model and parameters if provided
    FN_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = None
    
    if All_stage_dir_TF_CDM is not None:
        print("[INFO] Loading TF-CDM model and parameters...")
        data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
        if not os.path.exists(data_inf_path_TF_CDM):
            print(f"[WARNING] TF-CDM data_inf.pt not found at {data_inf_path_TF_CDM}, skipping TF-CDM")
        else:
            data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
            xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
            xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
            yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
            yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
            diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
            
            # Load TF-CDM FN_Net model
            FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
            if os.path.exists(FNET_path_TF_CDM):
                FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                FN_TF_CDM.eval()
            else:
                print(f"[WARNING] TF-CDM FNET.pth not found at {FNET_path_TF_CDM}, skipping TF-CDM")
                FN_TF_CDM = None
    
    # Create FEX function wrapper
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device)
    
    # Define fixed colors for each model
    model_colors = {
        "FEX-DM": "orange",
        "TF-CDM": "blue"
    }
    
    # SDE parameters
    sde_dt = 0.01
    x_dim = dimension
    Npath = 500000
    
    def plot_conditional_distribution_single(true_init, ax):
        """
        Plot conditional distribution for a given initial value, with both FEX-DM and TF-CDM in the same subplot.
        """
        x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
        
        # True Samples
        ode_path_true = true_init * np.ones((Npath, x_dim))
        true_samples = ode_path_true + theta * (mu - ode_path_true) * sde_dt + sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
        
        # Define Plotting Range
        x_min, x_max = np.min(true_samples) - 0.05, np.max(true_samples) + 0.05
        x_vals = np.linspace(x_min, x_max, 200)
        
        # Compute KDE for True Distribution
        kde = gaussian_kde(true_samples.T)
        pdf_vals = kde(x_vals)
        ax.plot(x_vals, pdf_vals, color='black', linewidth=1.8, linestyle='dashed', label="Ground Truth")
        
        # Model Predictions
        for model in ["FEX-DM", "TF-CDM"]:
            if model == "FEX-DM":
                with torch.no_grad():
                    z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                    prediction = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
                    prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
            elif model == "TF-CDM":
                # Skip TF-CDM for x0 = -6 and x0 = 6
                if abs(true_init - (-6)) < 0.01 or abs(true_init - 6) < 0.01:
                    continue  # Skip TF-CDM for these initial values
                if FN_TF_CDM is not None:
                    with torch.no_grad():
                        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                        prediction = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                        prediction = (prediction / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                else:
                    continue  # Skip if TF-CDM model not available
            
            # Plot Histogram of Learned Distribution
            ax.hist(prediction, bins=50, density=True, alpha=0.5, color=model_colors[model], 
                    histtype='stepfilled', edgecolor=model_colors[model], label=f"{model}")
        
        # Plot Settings
        ax.set_xlabel('$x$', fontsize=22)
        ax.set_ylabel('pdf', fontsize=22)
        ax.set_title(f'$x_0$ = {true_init:.2f}', fontsize=24)
        ax.set_xlim([x_min, x_max])
        ax.tick_params(axis='both', labelsize=22)
    
    # Create 1×3 Subplot Grid (One row, three subplots)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for col, x0 in enumerate(initial_values):
        plot_conditional_distribution_single(x0, axes[col])
    
    # Manually Add Legend with Fixed Colors
    legend_handles = [
        plt.Line2D([0], [0], color=model_colors["FEX-DM"], linewidth=6, label="FEX-DM"),
        plt.Line2D([0], [0], color=model_colors["TF-CDM"], linewidth=6, label="TF-CDM"),
        plt.Line2D([0], [0], color="black", linestyle="dashed", linewidth=2, label="Ground Truth")
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=3, fontsize=16, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(save_dir, 'conditional_distribution_1row_3plots.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    else:
        print(f"[WARNING] File was not created at: {save_path}")
    
    return save_path
