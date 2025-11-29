"""
Plotting utilities for FEX-DM
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
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
            for model in model_styles:
                x_pred_new = x_pred_new_dict[model]
                
                if model == "FEX-DM":
                    with torch.no_grad():
                        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                        prediction = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
                        prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                
                elif model == "TF-CDM" and FN_TF_CDM is not None:
                    with torch.no_grad():
                        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
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
