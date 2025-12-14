"""
Plotting utilities for FEX-DM
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde
from .ODEParser import FN_Net, CovarianceNet
from .FEX import FEX_model_learned
from .helper import VAE

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
                                 dpi=300,
                                 dataset_full=None):
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
        dataset_full: full dataset for OL2d visualization (optional)
    
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
    
    if model_name == 'OL2d' and dataset_full is not None:
        # For OL2d, create 2D visualization
        # dataset_full shape: (2, Nt+1, N_data)
        # Extract final positions
        final_positions_x = dataset_full[0, -1, :]
        final_positions_y = dataset_full[1, -1, :]
        
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(10, 4))
        
        # Upper plot: Potential function contour
        ax1 = plt.subplot(1, 2, 1)
        x_range = np.linspace(-1.5, 1.5, 400)
        y_range = np.linspace(-1, 1, 400)
        X, Y = np.meshgrid(x_range, y_range)
        Z = 0.5 * (X**2 - 1)**2 + 5 * Y**2
        contour1 = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
        fig.colorbar(contour1, ax=ax1)
        ax1.set_title('Potential Function $V(x, y)$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Lower plot: Density of particle positions
        histogram, xedges, yedges = np.histogram2d(final_positions_x, final_positions_y, bins=50, range=[[-1.5, 1.5], [-1, 1]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax2 = plt.subplot(1, 2, 2)
        contour2 = ax2.contourf(histogram.T, extent=extent, levels=20, cmap='viridis')
        fig.colorbar(contour2, ax=ax2)
        ax2.set_title('Density of Particle Positions')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 2D visualization saved to: {histogram_path}")
        
        # Also create 1D histogram for x_sample[:,0]
        # Use the same reshaping as the user's code: x_sample = (data_sample[:sde_Nt,:,:]).transpose(0,2,1).reshape((N_sample_path*sde_Nt,x_dim))
        # dataset_full shape: (2, Nt+1, N_data) = (n_dim, Nt+1, N_data)
        # Convert to (Nt+1, n_dim, N_data) format
        data_sample_format = dataset_full.transpose(1, 0, 2)  # (Nt+1, 2, N_data)
        # Take first Nt time steps: (Nt, 2, N_data)
        data_sample_format = data_sample_format[:dataset_full.shape[1]-1, :, :]  # (Nt, 2, N_data)
        # Transpose (0, 2, 1): (Nt, N_data, 2)
        data_sample_format = data_sample_format.transpose(0, 2, 1)  # (Nt, N_data, 2)
        # Reshape: (Nt*N_data, 2)
        x_sample = data_sample_format.reshape(-1, 2)  # (Nt*N_data, 2)
        # Extract dimension 1: (Nt*N_data,)
        x_sample_dim1 = x_sample[:, 0]
        
        histogram_1d_path = os.path.join(save_path, f'{model_name}_training_data_histogram_x1_noise_{noise_level}.pdf' if noise_level is not None else f'{model_name}_training_data_histogram_x1.pdf')
        plt.figure(figsize=(8, 6))
        plt.hist(x_sample_dim1, bins=100, density=True, edgecolor='black', alpha=0.7)
        plt.xlabel('x (Dimension 1)')
        plt.ylabel('Density')
        title_parts = [f'{model_name} Training Data Distribution - Dimension 1']
        if train_size is not None:
            title_parts.append(f'Samples: {train_size}')
        if noise_level is not None:
            title_parts.append(f'Noise Level: {noise_level}')
        plt.title(', '.join(title_parts))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(histogram_1d_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 1D histogram (x1) saved to: {histogram_1d_path}")
    else:
        # For 1D models, create regular histogram
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
        plt.close()
        print(f"[INFO] Histogram saved to: {histogram_path}")
    
    
    # return histogram_path


def plot_trajectory_comparison_simulation(second_stage_dir_FEX,
                                All_stage_dir_TF_CDM=None,
                                All_stage_dir_FEX_VAE=None,
                                All_stage_dir_FEX_NN=None,
                                model_name='OU1d',
                                noise_level=1.0,
                                device='cpu',
                                initial_values=None,
                                sde_params=None,
                                save_dir=None,
                                figsize=(18, 6),
                                dpi=300,
                                seed=42):
    """
    Plot comparison between FEX-DM, TF-CDM (optional), and FEX-VAE (optional) models for SDE simulation.
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        All_stage_dir_FEX_VAE: Optional directory path for FEX-VAE second stage results
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
        # For DoubleWell1d, we don't have mu, so set defaults
        if model_name == 'DoubleWell1d':
            sde_params = {
                'mu': 0.0,  # Not used for DoubleWell1d (drift is x - x^3)
                'sigma': model_params['sig'] * noise_level,
                'theta': model_params.get('th', 1.0),  # May exist but not used for drift
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
        else:
            sde_params = {
                'mu': model_params.get('mu', 1.2),  # Default for OU1d
                'sigma': model_params['sig'] * noise_level,
                'theta': model_params.get('th', 1.0),
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
    
    if initial_values is None:
        # Set model-specific default initial values
        if model_name == 'OU1d':
            initial_values = [-6, 1.5, 6]
        elif model_name == 'Trigonometric1d':
            initial_values = [-3, 0.6, 3]
        elif model_name == 'DoubleWell1d':
            initial_values = [-5, 1.5, 5]
        elif model_name == 'EXP1d':
            initial_values = [-2, 1.5, 2]
        elif model_name == 'MM1d':
            initial_values = [-0.5, 0.6, 1.5]  # MM1d initial values
        else:
            initial_values = [-6, 1.5, 6]  # Default fallback
    
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
    
    if All_stage_dir_FEX_VAE is not None:
        model_styles["FEX-VAE"] = {
            "color": "green",
            "fill": "green",
            "linestyle": "-"
        }
    
    if All_stage_dir_FEX_NN is not None:
        model_styles["FEX-NN"] = {
            "color": "purple",
            "fill": "purple",
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
    
    # Load FEX-VAE model if provided
    VAE_FEX = None

    if All_stage_dir_FEX_VAE is not None:
        print("[INFO] Loading FEX-VAE model...")
        # Load FEX-VAE VAE model
        VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
        if os.path.exists(VAE_path):
            from utils.helper import VAE
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
            VAE_FEX.eval()
        else:
            print(f"[WARNING] FEX-VAE VAE_FEX.pth not found at {VAE_path}, skipping FEX-VAE")
            model_styles.pop("FEX-VAE", None)
            VAE_FEX = None
    
    # Load FEX-NN model if provided
    FEX_NN = None

    if All_stage_dir_FEX_NN is not None:
        print("[INFO] Loading FEX-NN model...")
        # Load FEX-NN model
        FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
        if os.path.exists(FEX_NN_path):
            from utils.ODEParser import CovarianceNet
            # Get dimension from FEX-DM data
            output_dim_nn = dimension * dimension if dimension > 1 else 1
            FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
            FEX_NN.eval()
        else:
            print(f"[WARNING] FEX-NN FEX_NN.pth not found at {FEX_NN_path}, skipping FEX-NN")
            model_styles.pop("FEX-NN", None)
            FEX_NN = None
    
    # Extract domain folder from second_stage_dir_FEX path
    domain_folder = None
    if second_stage_dir_FEX:
        path_parts = second_stage_dir_FEX.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                break
    
    # Construct base_path for FEX_model_learned to find final_expressions.txt
    base_path = os.path.dirname(os.path.dirname(second_stage_dir_FEX))
    
    # Create FEX function wrapper
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device,
                                  domain_folder=domain_folder, base_path=base_path)
    
    # Extract SDE parameters
    mu = sde_params['mu']
    sigma = sde_params['sigma']
    theta = sde_params['theta']
    sde_T = sde_params['sde_T']
    sde_dt = sde_params['sde_dt']
    
    x_dim = dimension
    ode_time_steps = int(sde_T / sde_dt)
    Npath = 500000
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
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
                
                elif model == "FEX-VAE" and VAE_FEX is not None:
                    with torch.no_grad():
                        # Use the same z as FEX-DM, decode with VAE, then apply formula
                        prediction = VAE_FEX.decoder(z)
                        prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                
                elif model == "FEX-NN" and FEX_NN is not None:
                    with torch.no_grad():
                        # Predict covariance matrix from current state
                        cov_pred = FEX_NN(x_pred_new)  # (Npath, dim*dim) or (Npath, 1) for 1D
                        if dimension == 1:
                            # 1D case: cov_pred is (Npath, 1), use as variance
                            std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8))  # (Npath, 1)
                            prediction = (x_pred_new + FEX(x_pred_new) * sde_dt + std_pred * z * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                        else:
                            # Multi-D case: reshape to (Npath, dim, dim) and sample
                            cov_matrix = cov_pred.reshape(Npath, dimension, dimension)  # (Npath, dim, dim)
                            # Ensure positive semi-definite by adding small identity
                            cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                            # Sample from multivariate normal
                            try:
                                from torch.distributions import MultivariateNormal
                                dist = MultivariateNormal(torch.zeros(dimension, device=device), cov_matrix)
                                noise = dist.sample()  # (Npath, dim)
                            except:
                                # Fallback: use Cholesky decomposition
                                L = torch.linalg.cholesky(cov_matrix)  # (Npath, dim, dim)
                                noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)  # (Npath, dim)
                            prediction = (x_pred_new + FEX(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                
                ode_mean_pred[model][jj] = np.mean(prediction)
                ode_std_pred[model][jj] = np.std(prediction)
                
                # Update each model's state separately
                x_pred_new_dict[model] = torch.tensor(prediction).to(device, dtype=torch.float32)
            
            # True trajectory evolution - drift depends on model
            if model_name == 'DoubleWell1d':
                # Double Well: dX = (X - X^3)dt + sig*dB
                drift_true = ode_path_true - ode_path_true**3  # Drift: x - x^3
            else:
                # OU1d: dX = theta*(mu - X)dt + sigma*dB
                drift_true = theta * (mu - ode_path_true)
            ode_path_true = ode_path_true + drift_true * sde_dt + \
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
    
    if "FEX-VAE" in model_styles:
        legend_handles.append(
            plt.Line2D([0], [0], color='green', linestyle='-', linewidth=3, label='Pred Mean (FEX-VAE)')
        )
    if "FEX-NN" in model_styles:
        legend_handles.append(
            plt.Line2D([0], [0], color='purple', linestyle='-', linewidth=3, label='Pred Mean (FEX-NN)')
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
                             All_stage_dir_FEX_VAE=None,
                             All_stage_dir_FEX_NN=None,
                             model_name='OU1d',
                             noise_level=1.0,
                             device='cpu',
                             Npath=500000,
                             N_x0=500,
                             x_min=-6,
                             x_max=6,
                             save_dir=None,
                             figsize=(15, 6),
                             dpi=300,
                             seed=42):
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
        # For DoubleWell1d, we don't have mu, so set defaults
        if model_name == 'DoubleWell1d':
            sde_params = {
                'mu': 0.0,  # Not used for DoubleWell1d (drift is x - x^3)
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', 1.0),  # May exist but not used for drift
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
        elif model_name == 'EXP1d':
            # EXP1d: dX = th * X * dt + sig * Exp(1) * sqrt(dt)
            sde_params = {
                'mu': 0.0,  # Not used for EXP1d
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', -2.0),  # th is the drift coefficient for EXP1d
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
        else:
            sde_params = {
                'mu': model_params.get('mu', 1.2),  # Default for OU1d
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', 1.0),
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
    
    # Load FEX-VAE model if provided
    VAE_FEX = None

    if All_stage_dir_FEX_VAE is not None:
        print("[INFO] Loading FEX-VAE model...")
        # Load FEX-VAE VAE model
        VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
        if os.path.exists(VAE_path):
            from utils.helper import VAE
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
            VAE_FEX.eval()
        else:
            print(f"[WARNING] FEX-VAE VAE_FEX.pth not found at {VAE_path}, skipping FEX-VAE")
            VAE_FEX = None
    
    # Load FEX-NN model if provided
    FEX_NN = None

    if All_stage_dir_FEX_NN is not None:
        print("[INFO] Loading FEX-NN model...")
        # Load FEX-NN model
        FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
        if os.path.exists(FEX_NN_path):
            from utils.ODEParser import CovarianceNet
            output_dim_nn = dimension * dimension if dimension > 1 else 1
            FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
            FEX_NN.eval()
        else:
            print(f"[WARNING] FEX-NN FEX_NN.pth not found at {FEX_NN_path}, skipping FEX-NN")
            FEX_NN = None
    
    # Extract domain folder from second_stage_dir_FEX path
    domain_folder = None
    domain_start = 0.0
    domain_end = 2.5
    if second_stage_dir_FEX:
        path_parts = second_stage_dir_FEX.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                # Parse domain_start and domain_end from domain_folder (e.g., "domain_3.0_6.0")
                try:
                    parts = part.replace('domain_', '').split('_')
                    if len(parts) >= 2:
                        domain_start = float(parts[0])
                        domain_end = float(parts[1])
                except:
                    # If parsing fails, use defaults
                    pass
                break
    
    # Adjust x_min and x_max to extend by domain width on each side
    domain_width = domain_end - domain_start
    x_min = domain_start - domain_width
    x_max = domain_end + domain_width
    
    # Construct base_path for FEX_model_learned to find final_expressions.txt
    base_path = os.path.dirname(os.path.dirname(second_stage_dir_FEX))
    
    # Create FEX function wrapper
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device,
                                  domain_folder=domain_folder, base_path=base_path)
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    x_dim = dimension
    x0_grid = np.linspace(x_min, x_max, N_x0)
    
    # Initialize arrays for drift and diffusion
    bx_pred_FEX = np.zeros(N_x0)
    sigmax_pred_FEX = np.zeros(N_x0)
    bx_pred_TF_CDM = np.zeros(N_x0)
    sigmax_pred_TF_CDM = np.zeros(N_x0)
    bx_pred_VAE = np.zeros(N_x0)
    sigmax_pred_VAE = np.zeros(N_x0)
    bx_pred_NN = np.zeros(N_x0)
    sigmax_pred_NN = np.zeros(N_x0)
    
    # True drift and diffusion based on model
    if model_name == 'OU1d':
        # OU process: dX = theta*(mu - X)dt + sigma*dB
        bx_true = theta * (mu - x0_grid)  # Drift: theta*(mu - x)
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sigma
    elif model_name == 'DoubleWell1d':
        # Double Well: dX = (X - X^3)dt + sig*dB
        # Drift: μ(x) = x - x^3
        bx_true = x0_grid - x0_grid**3
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    elif model_name == 'EXP1d':
        # EXP1d: dX = th * X * dt + sig * Exp(1) * sqrt(dt)
        # Drift: th * x (where th = -2.0)
        bx_true = theta * x0_grid  # Drift: th * x
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    elif model_name == 'OL2d':
        # OL2d: 2D potential-based SDE
        # V(x,y) = 2.5*(x^2-1)^2 + 5*y^2
        # dVdx = [10*x*(x^2-1), 10*y]
        # drift = -dVdx/gamma = [-10*x*(x^2-1), -10*y] = [-10*x^3 + 10*x, -10*y]
        # For dimension 1: drift = -10*x1^3 + 10*x1 = 10*x1 - 10*x1^3
        # For dimension 2: drift = -10*x2
        # Note: This function is for 1D plotting, so we compute drift for dimension 1
        bx_true = -10 * x0_grid**3 + 10 * x0_grid  # Drift: -10*x^3 + 10*x
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    elif model_name == 'MM1d':
        # MM1d: dX_t = (tanh(X_t) - 0.5*X_t)dt + sig*dB_t
        bx_true = np.tanh(x0_grid) - 0.5 * x0_grid  # Drift: tanh(x) - 0.5*x
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    else:
        # Default to OU1d
        bx_true = theta * (mu - x0_grid)
        sigmax_true = sigma * np.ones(N_x0)
    
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
        
        # FEX-VAE Prediction
        if VAE_FEX is not None:
            with torch.no_grad():
                # Use the same z as FEX-DM, decode with VAE, then apply formula
                prediction_VAE = VAE_FEX.decoder(z)
                prediction_VAE = (prediction_VAE / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
            
            # Compute drift and diffusion
            bx_pred_VAE[jj] = np.mean((prediction_VAE - true_init) / sde_dt)
            sigmax_pred_VAE[jj] = np.std((prediction_VAE - true_init - bx_pred_VAE[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
        
        # FEX-NN Prediction
        if FEX_NN is not None:
            with torch.no_grad():
                # Predict covariance matrix from current state
                cov_pred = FEX_NN(x_pred_new)  # (Npath, dim*dim) or (Npath, 1) for 1D
                if dimension == 1:
                    # 1D case: cov_pred is (Npath, 1), use as variance
                    std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)  # (Npath,)
                    prediction_NN = (x_pred_new.squeeze(-1) + FEX(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                    prediction_NN = prediction_NN[:, np.newaxis]  # (Npath, 1)
                else:
                    # Multi-D case: reshape to (Npath, dim, dim) and sample
                    cov_matrix = cov_pred.reshape(Npath, dimension, dimension)  # (Npath, dim, dim)
                    # Ensure positive semi-definite by adding small identity
                    cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                    # Sample from multivariate normal using Cholesky
                    try:
                        L = torch.linalg.cholesky(cov_matrix)  # (Npath, dim, dim)
                        noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)  # (Npath, dim)
                    except:
                        # Fallback: use diagonal only
                        noise = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=1, dim2=2), min=1e-8)) * z  # (Npath, dim)
                    prediction_NN = (x_pred_new + FEX(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
            
            # Compute drift and diffusion
            bx_pred_NN[jj] = np.mean((prediction_NN - true_init) / sde_dt)
            sigmax_pred_NN[jj] = np.std((prediction_NN - true_init - bx_pred_NN[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
    
    # Calculate errors in training domain
    training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
    bx_true_training = bx_true[training_mask]
    sigmax_true_training = sigmax_true[training_mask]
    bx_pred_FEX_training = bx_pred_FEX[training_mask]
    sigmax_pred_FEX_training = sigmax_pred_FEX[training_mask]
    
    # Drift error: max absolute error (since drift can be near zero, relative error doesn't make sense)
    bx_error_FEX = np.max(np.abs(bx_pred_FEX_training - bx_true_training))
    
    # Diffusion error: max absolute error
    sigmax_error_FEX = np.max(np.abs(sigmax_pred_FEX_training - sigmax_true_training))
    
    # Print error table
    print("\n" + "="*80)
    print(f"ERROR TABLE (Training Domain: x ∈ [{domain_start}, {domain_end}])")
    print("="*80)
    print(f"{'Model':<15} {'Drift Error (Max Abs)':<25} {'Diffusion Error (Max Abs)':<25}")
    print("-"*80)
    print(f"{'FEX-DM':<15} {bx_error_FEX:<25.6e} {sigmax_error_FEX:<25.6e}")
    
    if FN_TF_CDM is not None:
        bx_pred_TF_CDM_training = bx_pred_TF_CDM[training_mask]
        sigmax_pred_TF_CDM_training = sigmax_pred_TF_CDM[training_mask]
        # Drift error: max absolute error
        bx_error_TF_CDM = np.max(np.abs(bx_pred_TF_CDM_training - bx_true_training))
        # Diffusion error: max absolute error
        sigmax_error_TF_CDM = np.max(np.abs(sigmax_pred_TF_CDM_training - sigmax_true_training))
        print(f"{'TF-CDM':<15} {bx_error_TF_CDM:<25.6e} {sigmax_error_TF_CDM:<25.6e}")
    
    if VAE_FEX is not None:
        bx_pred_VAE_training = bx_pred_VAE[training_mask]
        sigmax_pred_VAE_training = sigmax_pred_VAE[training_mask]
        # Drift error: max absolute error
        bx_error_VAE = np.max(np.abs(bx_pred_VAE_training - bx_true_training))
        # Diffusion error: max absolute error
        sigmax_error_VAE = np.max(np.abs(sigmax_pred_VAE_training - sigmax_true_training))
        print(f"{'FEX-VAE':<15} {bx_error_VAE:<25.6e} {sigmax_error_VAE:<25.6e}")
    
    if FEX_NN is not None:
        bx_pred_NN_training = bx_pred_NN[training_mask]
        sigmax_pred_NN_training = sigmax_pred_NN[training_mask]
        # Drift error: max absolute error
        bx_error_NN = np.max(np.abs(bx_pred_NN_training - bx_true_training))
        # Diffusion error: max absolute error
        sigmax_error_NN = np.max(np.abs(sigmax_pred_NN_training - sigmax_true_training))
        print(f"{'FEX-NN':<15} {bx_error_NN:<25.6e} {sigmax_error_NN:<25.6e}")
    
    print("="*80 + "\n")
    
    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.4)
    
    # Color & Style Setup
    colors = {'FEX-DM': 'orange', 'TF-CDM': 'steelblue', 'FEX-VAE': 'green', 'FEX-NN': 'purple', 'Ground-Truth': 'black'}
    linestyles = {'FEX-DM': '-', 'TF-CDM': '--', 'FEX-VAE': '-', 'FEX-NN': '-', 'Ground-Truth': ':'}
    markers = {'FEX-DM': 'o', 'TF-CDM': 's', 'FEX-VAE': '^', 'FEX-NN': 'v'}
    
    # Drift Plot (μ(x))
    ax[0].plot(x0_grid, bx_pred_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
               color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5)
    
    # Plot TF-CDM only within training domain (0 to 2.5)
    if FN_TF_CDM is not None:
        training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
        x0_training = x0_grid[training_mask]
        bx_pred_TF_CDM_training = bx_pred_TF_CDM[training_mask]
        ax[0].plot(x0_training, bx_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                   color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2)
    
    # Plot FEX-VAE
    if VAE_FEX is not None:
        ax[0].plot(x0_grid, bx_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                   color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5)
    
    # Plot FEX-NN
    if FEX_NN is not None:
        ax[0].plot(x0_grid, bx_pred_NN, label='FEX-NN', linestyle=linestyles['FEX-NN'], 
                   color=colors['FEX-NN'], linewidth=3, marker=markers['FEX-NN'], markersize=5)
    
    ax[0].plot(x0_grid, bx_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
               color=colors['Ground-Truth'], linewidth=2)
    
    ax[0].axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
    ax[0].axvline(domain_start, color='gray', linestyle='--', linewidth=2)
    ax[0].axvline(domain_end, color='gray', linestyle='--', linewidth=2)
    
    ax[0].set_xlabel('$x$', fontsize=30)
    ax[0].set_ylabel('$\\hat{\\mu}(x)$', fontsize=30)
    ax[0].tick_params(axis='both', labelsize=25)
    # Set x-axis ticks: include domain boundaries and some key points
    xticks = [x_min, domain_start, domain_end, x_max]
    ax[0].set_xticks(xticks)
    
    # Diffusion Plot (σ(x))
    ax[1].plot(x0_grid, sigmax_pred_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
               color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5)
    
    # Plot TF-CDM only within training domain
    if FN_TF_CDM is not None:
        training_mask_tf = (x0_grid >= domain_start) & (x0_grid <= domain_end)
        x0_training = x0_grid[training_mask_tf]
        sigmax_pred_TF_CDM_training = sigmax_pred_TF_CDM[training_mask_tf]
        ax[1].plot(x0_training, sigmax_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                   color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2)
    
    # Plot FEX-VAE
    if VAE_FEX is not None:
        ax[1].plot(x0_grid, sigmax_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                   color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5)
    
    # Plot FEX-NN
    if FEX_NN is not None:
        ax[1].plot(x0_grid, sigmax_pred_NN, label='FEX-NN', linestyle=linestyles['FEX-NN'], 
                   color=colors['FEX-NN'], linewidth=3, marker=markers['FEX-NN'], markersize=5)
    
    ax[1].plot(x0_grid, sigmax_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
               color=colors['Ground-Truth'], linewidth=2)
    
    ax[1].axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
    ax[1].axvline(domain_start, color='gray', linestyle='--', linewidth=2)
    ax[1].axvline(domain_end, color='gray', linestyle='--', linewidth=2)
    
    ax[1].set_xlabel('$x$', fontsize=30)
    ax[1].set_ylabel('$\\hat{\\sigma}(x)$', fontsize=30)
    ax[1].tick_params(axis='both', labelsize=25)
    # Set x-axis ticks: include domain boundaries and some key points
    xticks = [x_min, domain_start, domain_end, x_max]
    ax[1].set_xticks(xticks)
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
                                  All_stage_dir_FEX_VAE=None,
                                  All_stage_dir_FEX_NN=None,
                                  model_name='OU1d',
                                  noise_level=1.0,
                                  device='cpu',
                                  initial_values=None,
                                  sde_params=None,
                                  save_dir=None,
                                  figsize=(18, 6),
                                  dpi=300,
                                  seed=42):
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
        # Set model-specific default initial values
        if model_name == 'OU1d':
            initial_values = [-6, 1.5, 6]
        elif model_name == 'Trigonometric1d':
            initial_values = [-3, 0.6, 3]
        elif model_name == 'DoubleWell1d':
            initial_values = [-5, 1.5, 5]
        elif model_name == 'EXP1d':
            initial_values = [-2, 1.5, 2]
        elif model_name == 'MM1d':
            initial_values = [-0.5, 0.6, 1.5]  # MM1d initial values
        else:
            initial_values = [-6, 1.5, 6]  # Default fallback
    
    # Set default save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(second_stage_dir_FEX), 'plot')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load SDE parameters
    if sde_params is None:
        if params_init is not None:
            try:
                model_params = params_init(case_name=model_name)
                sigma_base = model_params['sig']
                # For DoubleWell1d, we don't have mu, so set defaults
                if model_name == 'DoubleWell1d':
                    mu = 0.0  # Not used for DoubleWell1d (drift is x - x^3)
                    sigma = sigma_base * noise_level
                    theta = model_params.get('th', 1.0)  # May exist but not used for drift
                else:
                    mu = model_params.get('mu', 1.2)
                    sigma = sigma_base * noise_level
                    theta = model_params.get('th', 1.0)
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
    
    # Load FEX-VAE model if provided
    VAE_FEX = None

    if All_stage_dir_FEX_VAE is not None:
        print("[INFO] Loading FEX-VAE model...")
        # Load FEX-VAE VAE model
        VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
        if os.path.exists(VAE_path):
            from utils.helper import VAE
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
            VAE_FEX.eval()
        else:
            print(f"[WARNING] FEX-VAE VAE_FEX.pth not found at {VAE_path}, skipping FEX-VAE")
            VAE_FEX = None
    
    # Load FEX-NN model if provided
    FEX_NN = None

    if All_stage_dir_FEX_NN is not None:
        print("[INFO] Loading FEX-NN model...")
        # Load FEX-NN model
        FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
        if os.path.exists(FEX_NN_path):
            from utils.ODEParser import CovarianceNet
            output_dim_nn = dimension * dimension if dimension > 1 else 1
            FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
            FEX_NN.eval()
        else:
            print(f"[WARNING] FEX-NN FEX_NN.pth not found at {FEX_NN_path}, skipping FEX-NN")
            FEX_NN = None
    
    # Extract domain folder from second_stage_dir_FEX path
    domain_folder = None
    if second_stage_dir_FEX:
        path_parts = second_stage_dir_FEX.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                break
    
    # Construct base_path for FEX_model_learned to find final_expressions.txt
    base_path = os.path.dirname(os.path.dirname(second_stage_dir_FEX))
    
    # Create FEX function wrapper
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device,
                                  domain_folder=domain_folder, base_path=base_path)
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Define fixed colors for each model
    model_colors = {
        "FEX-DM": "orange",
        "TF-CDM": "blue",
        "FEX-VAE": "green",
        "FEX-NN": "purple"
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
        
        # True Samples - drift depends on model
        ode_path_true = true_init * np.ones((Npath, x_dim))
        if model_name == 'DoubleWell1d':
            # Double Well: dX = (X - X^3)dt + sig*dB
            drift_true = ode_path_true - ode_path_true**3  # Drift: x - x^3
        else:
            # OU1d: dX = theta*(mu - X)dt + sigma*dB
            drift_true = theta * (mu - ode_path_true)
        true_samples = ode_path_true + drift_true * sde_dt + sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
        
        # Define Plotting Range
        x_min, x_max = np.min(true_samples) - 0.05, np.max(true_samples) + 0.05
        x_vals = np.linspace(x_min, x_max, 200)
        
        # Compute KDE for True Distribution
        kde = gaussian_kde(true_samples.T)
        pdf_vals = kde(x_vals)
        ax.plot(x_vals, pdf_vals, color='black', linewidth=1.8, linestyle='dashed', label="Ground Truth")
        
        # Generate the same random noise z for FEX-DM and FEX-VAE
        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
        
        # Model Predictions
        for model in ["FEX-DM", "TF-CDM", "FEX-VAE", "FEX-NN"]:
            if model == "FEX-DM":
                with torch.no_grad():
                    prediction = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
                    prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
            elif model == "TF-CDM":
                # Skip TF-CDM for x0 = -6 and x0 = 6
                if abs(true_init - (-6)) < 0.01 or abs(true_init - 6) < 0.01:
                    continue  # Skip TF-CDM for these initial values
                if FN_TF_CDM is not None:
                    with torch.no_grad():
                        prediction = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                        prediction = (prediction / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                else:
                    continue  # Skip if TF-CDM model not available
            elif model == "FEX-VAE":
                if VAE_FEX is not None:
                    with torch.no_grad():
                        # Use the same z as FEX-DM, decode with VAE, then apply formula
                        prediction = VAE_FEX.decoder(z)
                        prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                else:
                    continue  # Skip if FEX-VAE model not available
            elif model == "FEX-NN":
                # Skip FEX-NN for x0 = -6 and x0 = 6, only show for middle (x0 = 1.5)
                if abs(true_init - (-6)) < 0.01 or abs(true_init - 6) < 0.01:
                    continue  # Skip FEX-NN for x0 = -6 and x0 = 6
                if FEX_NN is not None:
                    with torch.no_grad():
                        # Predict covariance matrix from current state
                        cov_pred = FEX_NN(x_pred_new)  # (Npath, dim*dim) or (Npath, 1) for 1D
                        if dimension == 1:
                            # 1D case: cov_pred is (Npath, 1), use as variance
                            std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)  # (Npath,)
                            prediction = (x_pred_new.squeeze(-1) + FEX(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                            prediction = prediction[:, np.newaxis]  # (Npath, 1)
                        else:
                            # Multi-D case: reshape to (Npath, dim, dim) and sample
                            cov_matrix = cov_pred.reshape(Npath, dimension, dimension)  # (Npath, dim, dim)
                            # Ensure positive semi-definite by adding small identity
                            cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                            # Sample from multivariate normal using Cholesky
                            try:
                                L = torch.linalg.cholesky(cov_matrix)  # (Npath, dim, dim)
                                noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)  # (Npath, dim)
                            except:
                                # Fallback: use diagonal only
                                noise = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=1, dim2=2), min=1e-8)) * z  # (Npath, dim)
                            prediction = (x_pred_new + FEX(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                else:
                    continue  # Skip if FEX-NN model not available
            
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
        plt.Line2D([0], [0], color=model_colors["FEX-VAE"], linewidth=6, label="FEX-VAE"),
        plt.Line2D([0], [0], color=model_colors["FEX-NN"], linewidth=6, label="FEX-NN"),
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


def plot_drift_and_diffusion_with_errors(second_stage_dir_FEX,
                                         All_stage_dir_TF_CDM=None,
                                         All_stage_dir_FEX_VAE=None,
                                         All_stage_dir_FEX_NN=None,
                                         model_name='OU1d',
                                         noise_level=1.0,
                                         device='cpu',
                                         Npath=500000,
                                         N_x0=500,
                                         N_x0_error=50,  # Smaller subset for error plot
                                         x_min=-6,
                                         x_max=6,
                                         save_dir=None,
                                         figsize=(20, 12),  # Wider to fit legend in one row
                                         dpi=300,
                                         seed=42):
    """
    Plot drift (μ(x)) and diffusion (σ(x)) coefficients with error plots.
    This function extends plot_drift_and_diffusion by adding error visualizations.
    Creates a 2x2 subplot: main plots on top, error plots on bottom.
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        All_stage_dir_FEX_VAE: Optional directory path for FEX-VAE second stage results
        All_stage_dir_FEX_NN: Optional directory path for FEX-NN second stage results
        model_name: Model name (e.g., 'OU1d')
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        Npath: Number of paths for Monte Carlo simulation (default: 500000)
        N_x0: Number of initial values in grid for main plot (default: 500)
        N_x0_error: Number of initial values in grid for error plot (default: 50)
        x_min: Minimum initial value (default: -6)
        x_max: Maximum initial value (default: 6)
        save_dir: Directory to save the figure (default: parent of second_stage_dir_FEX/plot)
        figsize: Figure size tuple (default: (15, 12))
        dpi: Resolution for saved figure (default: 300)
        seed: Random seed for reproducibility
    
    Returns:
        str: Path to the saved figure file
    """
    # Call the original function to get all the data, then we'll compute errors
    # We need to duplicate the computation logic but add error calculation
    
    # Load SDE parameters
    if params_init is not None:
        model_params = params_init(case_name=model_name)
        sigma_base = model_params['sig']
        # For DoubleWell1d, we don't have mu, so set defaults
        if model_name == 'DoubleWell1d':
            sde_params = {
                'mu': 0.0,  # Not used for DoubleWell1d (drift is x - x^3)
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', 1.0),  # May exist but not used for drift
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
        elif model_name == 'EXP1d':
            # EXP1d: dX = th * X * dt + sig * Exp(1) * sqrt(dt)
            sde_params = {
                'mu': 0.0,  # Not used for EXP1d
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', -2.0),  # th is the drift coefficient for EXP1d
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
        else:
            sde_params = {
                'mu': model_params.get('mu', 1.2),  # Default for OU1d
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', 1.0),
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
    
    # Load models (reuse logic from plot_drift_and_diffusion)
    print("[INFO] Loading models for error plot...")
    data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
    if not os.path.exists(data_inf_path_FEX):
        raise FileNotFoundError(f"FEX-DM data_inf.pt not found at {data_inf_path_FEX}")
    
    data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
    xTrain_mean_FEX = data_inf_FEX['ZT_Train_mean'].to(device)
    xTrain_std_FEX = data_inf_FEX['ZT_Train_std'].to(device)
    yTrain_mean_FEX = data_inf_FEX['ODE_Train_mean'].to(device)
    yTrain_std_FEX = data_inf_FEX['ODE_Train_std'].to(device)
    diff_scale_FEX = data_inf_FEX['diff_scale']
    
    dimension = data_inf_FEX['ZT_Train_new'].shape[1]
    FNET_path_FEX = os.path.join(second_stage_dir_FEX, 'FNET.pth')
    if not os.path.exists(FNET_path_FEX):
        raise FileNotFoundError(f"FEX-DM FNET.pth not found at {FNET_path_FEX}")
    
    FN_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
    FN_FEX.load_state_dict(torch.load(FNET_path_FEX, map_location=device))
    FN_FEX.eval()
    
    # Load other models
    FN_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = None
    
    if All_stage_dir_TF_CDM is not None:
        data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
        if os.path.exists(data_inf_path_TF_CDM):
            data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
            xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
            xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
            yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
            yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
            diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
            
            FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
            if os.path.exists(FNET_path_TF_CDM):
                FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                FN_TF_CDM.eval()
    
    VAE_FEX = None
    if All_stage_dir_FEX_VAE is not None:
        VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
        if os.path.exists(VAE_path):
            from utils.helper import VAE
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
            VAE_FEX.eval()
    
    FEX_NN = None
    if All_stage_dir_FEX_NN is not None:
        FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
        if os.path.exists(FEX_NN_path):
            from utils.ODEParser import CovarianceNet
            output_dim_nn = dimension * dimension if dimension > 1 else 1
            FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
            FEX_NN.eval()
    
    # Extract domain folder
    domain_folder = None
    domain_start = 0.0
    domain_end = 2.5
    if second_stage_dir_FEX:
        path_parts = second_stage_dir_FEX.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                # Parse domain_start and domain_end from domain_folder (e.g., "domain_3.0_6.0")
                try:
                    parts = part.replace('domain_', '').split('_')
                    if len(parts) >= 2:
                        domain_start = float(parts[0])
                        domain_end = float(parts[1])
                except:
                    # If parsing fails, use defaults
                    pass
                break
    
    # Adjust x_min and x_max to extend by domain width on each side
    domain_width = domain_end - domain_start
    x_min = domain_start - domain_width
    x_max = domain_end + domain_width
    
    # Construct base_path for FEX_model_learned to find final_expressions.txt
    # base_path should point to the parent directory containing the domain folder
    base_path = os.path.dirname(os.path.dirname(second_stage_dir_FEX))
    
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device,
                                  domain_folder=domain_folder, base_path=base_path)
    
    # Set fixed random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    x_dim = dimension
    
    # First, call the original function to get the main plots data
    # We'll compute a subset for error plots
    x0_grid_error = np.linspace(x_min, x_max, N_x0_error)
    bx_error_FEX = np.zeros(N_x0_error)
    sigmax_error_FEX = np.zeros(N_x0_error)
    bx_error_TF_CDM = np.zeros(N_x0_error)
    sigmax_error_TF_CDM = np.zeros(N_x0_error)
    bx_error_VAE = np.zeros(N_x0_error)
    sigmax_error_VAE = np.zeros(N_x0_error)
    bx_error_NN = np.zeros(N_x0_error)
    sigmax_error_NN = np.zeros(N_x0_error)
    # True drift and diffusion for error plot based on model
    if model_name == 'OU1d':
        # OU process: dX = theta*(mu - X)dt + sigma*dB
        bx_true_error = theta * (mu - x0_grid_error)  # Drift: theta*(mu - x)
        sigmax_true_error = sigma * np.ones(N_x0_error)  # Diffusion: constant sigma
    elif model_name == 'DoubleWell1d':
        # Double Well: dX = (X - X^3)dt + sig*dB
        bx_true_error = x0_grid_error - x0_grid_error**3  # Drift: x - x^3
        sigmax_true_error = sigma * np.ones(N_x0_error)  # Diffusion: constant sig
    elif model_name == 'OL2d':
        # OL2d: 2D potential-based SDE
        # For dimension 1: drift = -10*x1^3 + 10*x1 = 10*x1 - 10*x1^3
        bx_true_error = -10 * x0_grid_error**3 + 10 * x0_grid_error  # Drift: -10*x^3 + 10*x
        sigmax_true_error = sigma * np.ones(N_x0_error)  # Diffusion: constant sig
    elif model_name == 'EXP1d':
        # EXP1d: dX = th * X * dt + sig * Exp(1) * sqrt(dt)
        # Drift: th * x (where th = -2.0)
        bx_true_error = theta * x0_grid_error+sigma/np.sqrt(sde_dt)  # Drift: th * x
        sigmax_true_error = sigma * np.ones(N_x0_error)  # Diffusion: constant sig
    elif model_name == 'MM1d':
        # MM1d: dX_t = (tanh(X_t) - 0.5*X_t)dt + sig*dB_t
        bx_true_error = np.tanh(x0_grid_error) - 0.5 * x0_grid_error  # Drift: tanh(x) - 0.5*x
        sigmax_true_error = sigma * np.ones(N_x0_error)  # Diffusion: constant sig
    else:
        # Default to OU1d
        bx_true_error = theta * (mu - x0_grid_error)
        sigmax_true_error = sigma * np.ones(N_x0_error)
    
    print(f"[INFO] Computing errors for {N_x0_error} initial values...")
    for jj in range(N_x0_error):
        if jj % 10 == 0:
            print(f"[INFO] Processing error {jj+1}/{N_x0_error}...")
        
        true_init = x0_grid_error[jj]
        x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
        
        # FEX-DM
        with torch.no_grad():
            prediction_FEX = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
            prediction_FEX = (prediction_FEX / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
        
        bx_pred = np.mean((prediction_FEX - true_init) / sde_dt)
        sigmax_pred = np.std((prediction_FEX - true_init - bx_pred * sde_dt)) * np.sqrt(1 / sde_dt)
        bx_error_FEX[jj] = np.abs(bx_pred - bx_true_error[jj])
        sigmax_error_FEX[jj] = np.abs(sigmax_pred - sigmax_true_error[jj])
        
        # TF-CDM
        if FN_TF_CDM is not None:
            with torch.no_grad():
                prediction_TF_CDM = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                prediction_TF_CDM = (prediction_TF_CDM / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
            bx_pred = np.mean((prediction_TF_CDM - true_init) / sde_dt)
            sigmax_pred = np.std((prediction_TF_CDM - true_init - bx_pred * sde_dt)) * np.sqrt(1 / sde_dt)
            bx_error_TF_CDM[jj] = np.abs(bx_pred - bx_true_error[jj])
            sigmax_error_TF_CDM[jj] = np.abs(sigmax_pred - sigmax_true_error[jj])
        
        # FEX-VAE
        if VAE_FEX is not None:
            with torch.no_grad():
                prediction_VAE = VAE_FEX.decoder(z)
                prediction_VAE = (prediction_VAE / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
            bx_pred = np.mean((prediction_VAE - true_init) / sde_dt)
            sigmax_pred = np.std((prediction_VAE - true_init - bx_pred * sde_dt)) * np.sqrt(1 / sde_dt)
            bx_error_VAE[jj] = np.abs(bx_pred - bx_true_error[jj])
            sigmax_error_VAE[jj] = np.abs(sigmax_pred - sigmax_true_error[jj])
        
        # FEX-NN
        if FEX_NN is not None:
            with torch.no_grad():
                cov_pred = FEX_NN(x_pred_new)
                if dimension == 1:
                    std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                    prediction_NN = (x_pred_new.squeeze(-1) + FEX(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                    prediction_NN = prediction_NN[:, np.newaxis]
                else:
                    cov_matrix = cov_pred.reshape(Npath, dimension, dimension)
                    cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                    try:
                        L = torch.linalg.cholesky(cov_matrix)
                        noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                    except:
                        noise = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=1, dim2=2), min=1e-8)) * z
                    prediction_NN = (x_pred_new + FEX(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
            bx_pred = np.mean((prediction_NN - true_init) / sde_dt)
            sigmax_pred = np.std((prediction_NN - true_init - bx_pred * sde_dt)) * np.sqrt(1 / sde_dt)
            bx_error_NN[jj] = np.abs(bx_pred - bx_true_error[jj])
            sigmax_error_NN[jj] = np.abs(sigmax_pred - sigmax_true_error[jj])
    
    # Now call the original function to get the main plots, then add error plots
    # We'll create a 2x2 layout
    # First, get data from original function by calling it
    # Actually, let's just create the full plot here with both main and error plots
    
    # Get main plot data by calling plot_drift_and_diffusion internally
    # But we need the data, so let's compute it here too for the main plots
    # Actually, let's just create a combined plot
    
    # Create 2x2 subplot: main plots on top, error plots on bottom
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    
    # Get main plot data - call original function but capture the plot
    # Actually, simpler: just compute main data here too
    x0_grid = np.linspace(x_min, x_max, N_x0)
    bx_pred_FEX = np.zeros(N_x0)
    sigmax_pred_FEX = np.zeros(N_x0)
    bx_pred_TF_CDM = np.zeros(N_x0)
    sigmax_pred_TF_CDM = np.zeros(N_x0)
    bx_pred_VAE = np.zeros(N_x0)
    sigmax_pred_VAE = np.zeros(N_x0)
    bx_pred_NN = np.zeros(N_x0)
    sigmax_pred_NN = np.zeros(N_x0)
    
    # True drift and diffusion based on model
    if model_name == 'OU1d':
        # OU process: dX = theta*(mu - X)dt + sigma*dB
        bx_true = theta * (mu - x0_grid)  # Drift: theta*(mu - x)
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sigma
    elif model_name == 'DoubleWell1d':
        # Double Well: dX = (X - X^3)dt + sig*dB
        bx_true = x0_grid - x0_grid**3  # Drift: x - x^3
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    elif model_name == 'OL2d':
        # OL2d: 2D potential-based SDE
        # For dimension 1: drift = -10*x1^3 + 10*x1 = 10*x1 - 10*x1^3
        bx_true = -10 * x0_grid**3 + 10 * x0_grid  # Drift: -10*x^3 + 10*x
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    elif model_name == 'EXP1d':
        # EXP1d: dX = th * X * dt + sig * Exp(1) * sqrt(dt)
        # Drift: th * x (where th = -2.0)
        bx_true = theta * x0_grid+sigma/np.sqrt(sde_dt) # Drift: th * x
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    elif model_name == 'MM1d':
        # MM1d: dX_t = (tanh(X_t) - 0.5*X_t)dt + sig*dB_t
        bx_true = np.tanh(x0_grid) - 0.5 * x0_grid  # Drift: tanh(x) - 0.5*x
        sigmax_true = sigma * np.ones(N_x0)  # Diffusion: constant sig
    else:
        # Default to OU1d
        bx_true = theta * (mu - x0_grid)
        sigmax_true = sigma * np.ones(N_x0)
    
    print(f"[INFO] Computing drift and diffusion for {N_x0} initial values (main plot)...")
    for jj in range(N_x0):
        if jj % 50 == 0:
            print(f"[INFO] Processing {jj+1}/{N_x0}...")
        
        true_init = x0_grid[jj]
        x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
        
        with torch.no_grad():
            prediction_FEX = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
            prediction_FEX = (prediction_FEX / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
        
        bx_pred_FEX[jj] = np.mean((prediction_FEX - true_init) / sde_dt)
        sigmax_pred_FEX[jj] = np.std((prediction_FEX - true_init - bx_pred_FEX[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
        
        if FN_TF_CDM is not None:
            with torch.no_grad():
                prediction_TF_CDM = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                prediction_TF_CDM = (prediction_TF_CDM / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
            bx_pred_TF_CDM[jj] = np.mean((prediction_TF_CDM - true_init) / sde_dt)
            sigmax_pred_TF_CDM[jj] = np.std((prediction_TF_CDM - true_init - bx_pred_TF_CDM[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
        
        if VAE_FEX is not None:
            with torch.no_grad():
                prediction_VAE = VAE_FEX.decoder(z)
                prediction_VAE = (prediction_VAE / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
            bx_pred_VAE[jj] = np.mean((prediction_VAE - true_init) / sde_dt)
            sigmax_pred_VAE[jj] = np.std((prediction_VAE - true_init - bx_pred_VAE[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
        
        if FEX_NN is not None:
            with torch.no_grad():
                cov_pred = FEX_NN(x_pred_new)
                if dimension == 1:
                    std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                    prediction_NN = (x_pred_new.squeeze(-1) + FEX(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                    prediction_NN = prediction_NN[:, np.newaxis]
                else:
                    cov_matrix = cov_pred.reshape(Npath, dimension, dimension)
                    cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                    try:
                        L = torch.linalg.cholesky(cov_matrix)
                        noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                    except:
                        noise = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=1, dim2=2), min=1e-8)) * z
                    prediction_NN = (x_pred_new + FEX(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
            bx_pred_NN[jj] = np.mean((prediction_NN - true_init) / sde_dt)
            sigmax_pred_NN[jj] = np.std((prediction_NN - true_init - bx_pred_NN[jj] * sde_dt)) * np.sqrt(1 / sde_dt)
    
    colors = {'FEX-DM': 'orange', 'TF-CDM': 'steelblue', 'FEX-VAE': 'green', 'FEX-NN': 'purple', 'Ground-Truth': 'black'}
    linestyles = {'FEX-DM': '-', 'TF-CDM': '--', 'FEX-VAE': '-', 'FEX-NN': '-', 'Ground-Truth': ':'}
    markers = {'FEX-DM': 'o', 'TF-CDM': 's', 'FEX-VAE': '^', 'FEX-NN': 'v'}
    
    # Top row: Main plots (same as original)
    # Drift Plot
    ax = axes[0, 0]
    # Draw FEX-VAE and FEX-NN first (bottom layer)
    if VAE_FEX is not None:
        ax.plot(x0_grid, bx_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
               color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5, zorder=1)
    
    if FEX_NN is not None:
        # Only show FEX-NN in training domain
        training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
        x0_training = x0_grid[training_mask]
        bx_pred_NN_training = bx_pred_NN[training_mask]
        ax.plot(x0_training, bx_pred_NN_training, label='FEX-NN', linestyle=linestyles['FEX-NN'], 
               color=colors['FEX-NN'], linewidth=3, marker=markers['FEX-NN'], markersize=5, zorder=1)
    
    # Draw FEX-DM and TF-CDM on top
    ax.plot(x0_grid, bx_pred_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
           color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5, zorder=3)
    
    if FN_TF_CDM is not None:
        training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
        x0_training = x0_grid[training_mask]
        bx_pred_TF_CDM_training = bx_pred_TF_CDM[training_mask]
        ax.plot(x0_training, bx_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
               color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2, zorder=3)
    
    ax.plot(x0_grid, bx_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
           color=colors['Ground-Truth'], linewidth=2)
    ax.axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
    ax.axvline(domain_start, color='gray', linestyle='--', linewidth=2)
    ax.axvline(domain_end, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$\\hat{\\mu}(x)$', fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    # Set x-axis ticks: include domain boundaries and some key points
    xticks = [x_min, domain_start, domain_end, x_max]
    ax.set_xticks(xticks)
    
    # Diffusion Plot
    ax = axes[0, 1]
    # Draw FEX-VAE and FEX-NN first (bottom layer)
    if VAE_FEX is not None:
        ax.plot(x0_grid, sigmax_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
               color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5, zorder=1)
    
    if FEX_NN is not None:
        # Only show FEX-NN in training domain
        training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
        x0_training = x0_grid[training_mask]
        sigmax_pred_NN_training = sigmax_pred_NN[training_mask]
        ax.plot(x0_training, sigmax_pred_NN_training, label='FEX-NN', linestyle=linestyles['FEX-NN'], 
               color=colors['FEX-NN'], linewidth=3, marker=markers['FEX-NN'], markersize=5, zorder=1)
    
    # Draw FEX-DM and TF-CDM on top
    ax.plot(x0_grid, sigmax_pred_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
           color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5, zorder=3)
    
    if FN_TF_CDM is not None:
        training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
        x0_training = x0_grid[training_mask]
        sigmax_pred_TF_CDM_training = sigmax_pred_TF_CDM[training_mask]
        ax.plot(x0_training, sigmax_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
               color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2, zorder=3)
    
    ax.plot(x0_grid, sigmax_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
           color=colors['Ground-Truth'], linewidth=2)
    ax.axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
    ax.axvline(domain_start, color='gray', linestyle='--', linewidth=2)
    ax.axvline(domain_end, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$\\hat{\\sigma}(x)$', fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    # Set x-axis ticks: include domain boundaries and some key points
    xticks = [x_min, domain_start, domain_end, x_max]
    ax.set_xticks(xticks)
    # Set y-axis range based on model
    if model_name == 'DoubleWell1d':
        ax.set_ylim([0.3, 0.7])
    elif model_name == 'EXP1d':
        ax.set_ylim([0.0, 0.2])
    else:
        ax.set_ylim([0.1, 0.45])
    
    # Bottom row: Error plots
    # Drift Error Plot
    ax = axes[1, 0]
    ax.plot(x0_grid_error, bx_error_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
           color=colors['FEX-DM'], linewidth=2, marker=markers['FEX-DM'], markersize=4)
    
    if FN_TF_CDM is not None:
        training_mask_error = (x0_grid_error >= domain_start) & (x0_grid_error <= domain_end)
        x0_training_error = x0_grid_error[training_mask_error]
        bx_error_TF_CDM_training = bx_error_TF_CDM[training_mask_error]
        ax.plot(x0_training_error, bx_error_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
               color=colors['TF-CDM'], linewidth=2, marker=markers['TF-CDM'], markersize=3)
    
    if VAE_FEX is not None:
        ax.plot(x0_grid_error, bx_error_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
               color=colors['FEX-VAE'], linewidth=2, marker=markers['FEX-VAE'], markersize=4)
    
    if FEX_NN is not None:
        # Only show FEX-NN in training domain (0.0-2.5)
        training_mask_error = (x0_grid_error >= domain_start) & (x0_grid_error <= domain_end)
        x0_training_error = x0_grid_error[training_mask_error]
        bx_error_NN_training = bx_error_NN[training_mask_error]
        ax.plot(x0_training_error, bx_error_NN_training, label='FEX-NN', linestyle=linestyles['FEX-NN'], 
               color=colors['FEX-NN'], linewidth=2, marker=markers['FEX-NN'], markersize=4)
    
    ax.axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
    ax.axvline(domain_start, color='gray', linestyle='--', linewidth=2)
    ax.axvline(domain_end, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$|\\hat{\\mu}(x) - \\mu(x)|$', fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    # Set x-axis ticks: include domain boundaries and some key points
    xticks = [x_min, domain_start, domain_end, x_max]
    ax.set_xticks(xticks)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Diffusion Error Plot
    ax = axes[1, 1]
    ax.plot(x0_grid_error, sigmax_error_FEX, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
           color=colors['FEX-DM'], linewidth=2, marker=markers['FEX-DM'], markersize=4)
    
    if FN_TF_CDM is not None:
        training_mask_error = (x0_grid_error >= domain_start) & (x0_grid_error <= domain_end)
        x0_training_error = x0_grid_error[training_mask_error]
        sigmax_error_TF_CDM_training = sigmax_error_TF_CDM[training_mask_error]
        ax.plot(x0_training_error, sigmax_error_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
               color=colors['TF-CDM'], linewidth=2, marker=markers['TF-CDM'], markersize=3)
    
    if VAE_FEX is not None:
        ax.plot(x0_grid_error, sigmax_error_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
               color=colors['FEX-VAE'], linewidth=2, marker=markers['FEX-VAE'], markersize=4)
    
    if FEX_NN is not None:
        # Only show FEX-NN in training domain (0.0-2.5)
        training_mask_error = (x0_grid_error >= domain_start) & (x0_grid_error <= domain_end)
        x0_training_error = x0_grid_error[training_mask_error]
        sigmax_error_NN_training = sigmax_error_NN[training_mask_error]
        ax.plot(x0_training_error, sigmax_error_NN_training, label='FEX-NN', linestyle=linestyles['FEX-NN'], 
               color=colors['FEX-NN'], linewidth=2, marker=markers['FEX-NN'], markersize=4)
    
    ax.axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
    ax.axvline(domain_start, color='gray', linestyle='--', linewidth=2)
    ax.axvline(domain_end, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$|\\hat{\\sigma}(x) - \\sigma(x)|$', fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    # Set x-axis ticks: include domain boundaries and some key points
    xticks = [x_min, domain_start, domain_end, x_max]
    ax.set_xticks(xticks)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    n_legend_items = len(handles)
    # Use ncol equal to number of items to fit all in one row
    fig.legend(handles, labels, loc='upper center', fontsize=22, frameon=True, 
               ncol=n_legend_items, bbox_to_anchor=(0.5, 1.01))
    
    # Save - give more space at top for legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(save_dir, 'drift_and_diffusion_with_errors.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    
    return save_path


def plot_trajectory_error_estimation(second_stage_dir_FEX,
                                     All_stage_dir_TF_CDM=None,
                                     All_stage_dir_FEX_VAE=None,
                                     All_stage_dir_FEX_NN=None,
                                     model_name='OU1d',
                                     noise_level=1.0,
                                     device='cpu',
                                     initial_values=None,
                                     sde_params=None,
                                     save_dir=None,
                                     figsize=(20, 12),
                                     dpi=300,
                                     seed=42):
    """
    Plot trajectory comparison with error estimation - shows errors (prediction - ground truth) 
    to better distinguish between models.
    
    Creates a 2x3 layout: top row shows mean trajectories, bottom row shows error (pred - true).
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        All_stage_dir_FEX_VAE: Optional directory path for FEX-VAE second stage results
        All_stage_dir_FEX_NN: Optional directory path for FEX-NN second stage results
        model_name: Model name (e.g., 'OU1d')
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        initial_values: List of initial values to test (default: [-6, 1.5, 6])
        sde_params: Dictionary with SDE parameters (mu, sigma, theta, sde_T, sde_dt)
        save_dir: Directory to save the figure
        figsize: Figure size tuple (default: (20, 12))
        dpi: Resolution for saved figure (default: 300)
        seed: Random seed for reproducibility
    
    Returns:
        str: Path to the saved figure file
    """
    # Load SDE parameters from model parameters
    if sde_params is None:
        model_params = params_init(case_name=model_name)
        sigma_base = model_params['sig']
        # For DoubleWell1d, we don't have mu, so set defaults
        if model_name == 'DoubleWell1d':
            sde_params = {
                'mu': 0.0,  # Not used for DoubleWell1d (drift is x - x^3)
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', 1.0),  # May exist but not used for drift
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
        else:
            sde_params = {
                'mu': model_params.get('mu', 1.2),  # Default for OU1d
                'sigma': sigma_base * noise_level,
                'theta': model_params.get('th', 1.0),
                'sde_T': model_params['T'],
                'sde_dt': model_params['Dt']
            }
    
    if initial_values is None:
        # Set model-specific default initial values
        if model_name == 'OU1d':
            initial_values = [-6, 1.5, 6]
        elif model_name == 'Trigonometric1d':
            initial_values = [-3, 0.6, 3]
        elif model_name == 'DoubleWell1d':
            initial_values = [-5, 1.5, 5]
        elif model_name == 'EXP1d':
            initial_values = [-2, 1.5, 2]
        elif model_name == 'MM1d':
            initial_values = [-0.5, 0.6, 1.5]  # MM1d initial values
        else:
            initial_values = [-6, 1.5, 6]  # Default fallback
    
    if save_dir is None:
        parent_dir = os.path.dirname(second_stage_dir_FEX)
        save_dir = os.path.join(parent_dir, 'plot')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Model styles - all solid lines except ground truth (dashed)
    model_styles = {
        "FEX-DM": {"color": "orange", "fill": "orange", "linestyle": "-", "linewidth": 3},
        "TF-CDM": {"color": "steelblue", "fill": "steelblue", "linestyle": "-", "linewidth": 3},
        "FEX-VAE": {"color": "green", "fill": "green", "linestyle": "-", "linewidth": 3},
        "FEX-NN": {"color": "purple", "fill": "purple", "linestyle": "-", "linewidth": 3}
    }
    
    # Load models (same as plot_trajectory_comparison_simulation)
    print("[INFO] Loading models for error estimation...")
    data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
    if not os.path.exists(data_inf_path_FEX):
        raise FileNotFoundError(f"FEX-DM data_inf.pt not found at {data_inf_path_FEX}")
    
    data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
    xTrain_mean_FEX = data_inf_FEX['ZT_Train_mean'].to(device)
    xTrain_std_FEX = data_inf_FEX['ZT_Train_std'].to(device)
    yTrain_mean_FEX = data_inf_FEX['ODE_Train_mean'].to(device)
    yTrain_std_FEX = data_inf_FEX['ODE_Train_std'].to(device)
    diff_scale_FEX = data_inf_FEX['diff_scale']
    
    dimension = data_inf_FEX['ZT_Train_new'].shape[1]
    FNET_path_FEX = os.path.join(second_stage_dir_FEX, 'FNET.pth')
    if not os.path.exists(FNET_path_FEX):
        raise FileNotFoundError(f"FEX-DM FNET.pth not found at {FNET_path_FEX}")
    
    FN_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
    FN_FEX.load_state_dict(torch.load(FNET_path_FEX, map_location=device))
    FN_FEX.eval()
    
    # Load other models
    FN_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = None
    
    if All_stage_dir_TF_CDM is not None:
        data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
        if os.path.exists(data_inf_path_TF_CDM):
            data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
            xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
            xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
            yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
            yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
            diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
            
            FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
            if os.path.exists(FNET_path_TF_CDM):
                FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                FN_TF_CDM.eval()
    
    VAE_FEX = None
    if All_stage_dir_FEX_VAE is not None:
        VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
        if os.path.exists(VAE_path):
            from utils.helper import VAE
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
            VAE_FEX.eval()
    
    FEX_NN = None
    if All_stage_dir_FEX_NN is not None:
        FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
        if os.path.exists(FEX_NN_path):
            from utils.ODEParser import CovarianceNet
            output_dim_nn = dimension * dimension if dimension > 1 else 1
            FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
            FEX_NN.eval()
    
    # Extract domain folder and boundaries
    domain_folder = None
    domain_start = 0.0
    domain_end = 2.5
    if second_stage_dir_FEX:
        path_parts = second_stage_dir_FEX.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                # Parse domain_start and domain_end from domain_folder (e.g., "domain_-2.0_2.0")
                try:
                    parts = part.replace('domain_', '').split('_')
                    if len(parts) >= 2:
                        domain_start = float(parts[0])
                        domain_end = float(parts[1])
                except:
                    # If parsing fails, use defaults
                    pass
                break
    
    # Construct base_path for FEX_model_learned to find final_expressions.txt
    base_path = os.path.dirname(os.path.dirname(second_stage_dir_FEX))
    
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device,
                                  domain_folder=domain_folder, base_path=base_path)
    
    mu = sde_params['mu']
    sigma = sde_params['sigma']
    theta = sde_params['theta']
    sde_T = sde_params['sde_T']
    sde_dt = sde_params['sde_dt']
    
    x_dim = dimension
    ode_time_steps = int(sde_T / sde_dt)
    Npath = 500000
    
    # Set fixed random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Determine which models to plot
    models_to_plot = ["FEX-DM"]
    if FN_TF_CDM is not None:
        models_to_plot.append("TF-CDM")
    if VAE_FEX is not None:
        models_to_plot.append("FEX-VAE")
    if FEX_NN is not None:
        models_to_plot.append("FEX-NN")
    
    def run_simulation_with_error(true_init, ax_mean, ax_error):
        """Run simulation and plot both mean and error."""
        # Determine which models to actually compute for this initial value
        # Skip TF-CDM and FEX-NN if outside training domain
        models_to_compute = []
        for model in models_to_plot:
            if model in ["TF-CDM", "FEX-NN"] and (true_init < domain_start or true_init > domain_end):
                continue  # Skip this model for this initial value
            models_to_compute.append(model)
        
        ode_mean_pred = {model: np.zeros(ode_time_steps) for model in models_to_compute}
        ode_std_pred = {model: np.zeros(ode_time_steps) for model in models_to_compute}
        ode_error_mean = {model: np.zeros(ode_time_steps) for model in models_to_compute}
        ode_error_std = {model: np.zeros(ode_time_steps) for model in models_to_compute}
        
        x_pred_new_dict = {model: torch.clone((true_init * torch.ones(Npath, x_dim)).to(device)) 
                          for model in models_to_compute}
        
        ode_mean_true = np.zeros(ode_time_steps)
        ode_std_true = np.zeros(ode_time_steps)
        ode_path_true = true_init * np.ones((Npath, x_dim))
        
        for jj in range(ode_time_steps):
            z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
            
            for model in models_to_compute:
                
                x_pred_new = x_pred_new_dict[model]
                
                if model == "FEX-DM":
                    with torch.no_grad():
                        prediction = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
                        prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                
                elif model == "TF-CDM" and FN_TF_CDM is not None:
                    with torch.no_grad():
                        prediction = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                        prediction = (prediction / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                
                elif model == "FEX-VAE" and VAE_FEX is not None:
                    with torch.no_grad():
                        prediction = VAE_FEX.decoder(z)
                        prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                
                elif model == "FEX-NN" and FEX_NN is not None:
                    with torch.no_grad():
                        cov_pred = FEX_NN(x_pred_new)
                        if dimension == 1:
                            std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8))
                            prediction = (x_pred_new + FEX(x_pred_new) * sde_dt + std_pred * z * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                        else:
                            cov_matrix = cov_pred.reshape(Npath, dimension, dimension)
                            cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                            try:
                                L = torch.linalg.cholesky(cov_matrix)
                                noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                            except:
                                noise = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=1, dim2=2), min=1e-8)) * z
                            prediction = (x_pred_new + FEX(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                
                ode_mean_pred[model][jj] = np.mean(prediction)
                ode_std_pred[model][jj] = np.std(prediction)
                
                # Calculate error (prediction - ground truth)
                error = prediction - ode_path_true
                ode_error_mean[model][jj] = np.mean(error)
                ode_error_std[model][jj] = np.std(error)
                
                x_pred_new_dict[model] = torch.tensor(prediction).to(device, dtype=torch.float32)
            
            # True trajectory evolution - drift depends on model
            if model_name == 'DoubleWell1d':
                # Double Well: dX = (X - X^3)dt + sig*dB
                drift_true = ode_path_true - ode_path_true**3  # Drift: x - x^3
                ode_path_true = ode_path_true + drift_true * sde_dt + \
                               sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
            elif model_name == 'OL2d':
                # OL2d: 2D potential-based SDE
                # For dimension 1: drift = -10*x1^3 + 10*x1 = 10*x1 - 10*x1^3
                # Note: This is for 1D plotting, so we use dimension 1 drift
                drift_true = -10 * ode_path_true**3 + 10 * ode_path_true  # Drift: -10*x^3 + 10*x
                ode_path_true = ode_path_true + drift_true * sde_dt + \
                               sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
            elif model_name == 'EXP1d':
                # EXP1d: dX = th * X * dt + sig * Exp(1) * sqrt(dt)
                # Use exponential noise instead of normal noise
                ode_path_true = ode_path_true + theta * ode_path_true * sde_dt + \
                               sigma * np.sqrt(sde_dt) * np.random.exponential(scale=1.0, size=(Npath, x_dim))
            else:
                # OU1d: dX = theta*(mu - X)dt + sigma*dB
                drift_true = theta * (mu - ode_path_true)
                ode_path_true = ode_path_true + drift_true * sde_dt + \
                               sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
            ode_mean_true[jj] = np.mean(ode_path_true)
            ode_std_true[jj] = np.std(ode_path_true)
        
        tmesh = np.linspace(sde_dt, ode_time_steps * sde_dt, ode_time_steps)
        
        # Plot mean trajectories (top row)
        ax_mean.plot(tmesh, ode_mean_true, linewidth=4, label="Ground Truth", 
                    color='black', linestyle='--', zorder=10)
        
        # Sort models so FEX-DM is plotted last (on top)
        models_sorted = sorted(models_to_compute, key=lambda x: (x != "FEX-DM", x))
        
        for model in models_sorted:
            
            style = model_styles[model]
            # Set zorder for lines and fill based on model
            # FEX-DM (orange) should be on top, FEX-VAE (green) should be below
            # Higher zorder = drawn on top
            if model == "FEX-DM":
                line_zorder = 6  # FEX-DM line on top
                fill_zorder = 5  # FEX-DM shaded area on top (below line but above others)
            elif model == "FEX-VAE":
                line_zorder = 2
                fill_zorder = 1  # Green shaded area at bottom
            else:
                line_zorder = 3
                fill_zorder = 2  # Other models (TF-CDM, FEX-NN) in middle
            
            ax_mean.plot(tmesh, ode_mean_pred[model], label=model, 
                        color=style["color"], linestyle=style["linestyle"], 
                        linewidth=style["linewidth"], zorder=line_zorder)
            ax_mean.fill_between(tmesh, 
                                ode_mean_pred[model] - ode_std_pred[model],
                                ode_mean_pred[model] + ode_std_pred[model],
                                color=style["fill"], alpha=0.15, zorder=fill_zorder)
        
        ax_mean.set_xlabel('Time', fontsize=20)
        ax_mean.set_ylabel('Mean Value', fontsize=20)
        ax_mean.set_title(f'$x_0$ = {true_init:.2f}', fontsize=22)
        ax_mean.tick_params(axis='both', labelsize=18)
        ax_mean.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Set y-axis limits based on model and initial value
        if model_name == 'DoubleWell1d' and abs(true_init - 1.5) < 0.01:
            ax_mean.set_ylim([0.6, 1.6])
        elif model_name == 'EXP1d' and abs(true_init - 1.5) < 0.01:
            ax_mean.set_ylim([0.1, 1.55])
        elif abs(true_init - 1.5) < 0.01:
            ax_mean.set_ylim([1.1, 1.6])
        
        # Plot errors (bottom row) - no ground truth reference
        ax_error.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        
        for model in models_to_compute:
            
            style = model_styles[model]
            ax_error.plot(tmesh, ode_error_mean[model], label=model, 
                         color=style["color"], linestyle=style["linestyle"], 
                         linewidth=style["linewidth"], zorder=5)
            ax_error.fill_between(tmesh,
                                 ode_error_mean[model] - ode_error_std[model],
                                 ode_error_mean[model] + ode_error_std[model],
                                 color=style["fill"], alpha=0.2, zorder=1)
        
        ax_error.set_xlabel('Time', fontsize=20)
        ax_error.set_ylabel('Error (Pred - True)', fontsize=20)
        ax_error.tick_params(axis='both', labelsize=18)
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, len(initial_values), figsize=figsize)
    if len(initial_values) == 1:
        axes = axes.reshape(2, 1)
    
    for col, x0 in enumerate(initial_values):
        run_simulation_with_error(x0, axes[0, col], axes[1, col])
    
    # Create legend - ground truth dashed, others solid
    legend_handles = [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=4, label='Ground Truth'),
        plt.Line2D([0], [0], color='orange', linestyle='-', linewidth=3, label='FEX-DM')
    ]
    
    if FN_TF_CDM is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color='steelblue', linestyle='-', linewidth=3, label='TF-CDM')
        )
    if VAE_FEX is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color='green', linestyle='-', linewidth=3, label='FEX-VAE')
        )
    if FEX_NN is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color='purple', linestyle='-', linewidth=3, label='FEX-NN')
        )
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.01), 
               ncol=len(legend_handles), fontsize=18, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(save_dir, 'trajectory_error_estimation.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    
    return save_path


def plot_conditional_distribution_with_errors(second_stage_dir_FEX,
                                             All_stage_dir_TF_CDM=None,
                                             All_stage_dir_FEX_VAE=None,
                                             All_stage_dir_FEX_NN=None,
                                             model_name='OU1d',
                                             noise_level=1.0,
                                             device='cpu',
                                             initial_values=None,
                                             sde_params=None,
                                             save_dir=None,
                                             figsize=(18, 12),
                                             dpi=300,
                                             seed=42):
    """
    Plot conditional distribution comparison with error plots (2x3 layout).
    Top row: conditional distributions (PDFs)
    Bottom row: error plots (prediction - ground truth)
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        All_stage_dir_FEX_VAE: Optional directory path for FEX-VAE second stage results
        All_stage_dir_FEX_NN: Optional directory path for FEX-NN second stage results
        model_name: Model name (e.g., 'OU1d')
        noise_level: Noise level (default: 1.0)
        device: Device to use ('cpu' or 'cuda')
        initial_values: List of initial values to plot (default: [-6, 1.5, 6])
        sde_params: Dictionary of SDE parameters (mu, sigma, theta). If None, will try to load from params_init
        save_dir: Directory to save the plot (default: plot folder in parent of second_stage_dir_FEX)
        figsize: Figure size tuple (default: (18, 12))
        dpi: Resolution for saved figure (default: 300)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        str: Path to the saved figure file
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Set default initial values
    if initial_values is None:
        # Set model-specific default initial values
        if model_name == 'OU1d':
            initial_values = [-6, 1.5, 6]
        elif model_name == 'Trigonometric1d':
            initial_values = [-3, 0.6, 3]
        elif model_name == 'DoubleWell1d':
            initial_values = [-5, 1.5, 5]
        elif model_name == 'EXP1d':
            initial_values = [-2, 1.5, 2]
        elif model_name == 'MM1d':
            initial_values = [-0.5, 0.6, 1.5]  # MM1d initial values
        else:
            initial_values = [-6, 1.5, 6]  # Default fallback
    
    # Set default save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(second_stage_dir_FEX), 'plot')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load SDE parameters
    if sde_params is None:
        if params_init is not None:
            try:
                model_params = params_init(case_name=model_name)
                sigma_base = model_params['sig']
                # For DoubleWell1d, we don't have mu, so set defaults
                if model_name == 'DoubleWell1d':
                    mu = 0.0  # Not used for DoubleWell1d (drift is x - x^3)
                    sigma = sigma_base * noise_level
                    theta = model_params.get('th', 1.0)  # May exist but not used for drift
                else:
                    mu = model_params.get('mu', 1.2)
                    sigma = sigma_base * noise_level
                    theta = model_params.get('th', 1.0)
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
    
    # Load FEX model and extract domain boundaries
    domain_folder = None
    domain_start = 0.0
    domain_end = 2.5
    if 'domain_' in second_stage_dir_FEX:
        parts = second_stage_dir_FEX.split('/')
        for part in parts:
            if part.startswith('domain_'):
                domain_folder = part
                # Parse domain_start and domain_end from domain_folder (e.g., "domain_-2.0_2.0")
                try:
                    domain_parts = part.replace('domain_', '').split('_')
                    if len(domain_parts) >= 2:
                        domain_start = float(domain_parts[0])
                        domain_end = float(domain_parts[1])
                except:
                    # If parsing fails, use defaults
                    pass
                break
    
    base_path = os.path.dirname(os.path.dirname(second_stage_dir_FEX))
    
    # Create FEX function wrapper
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device,
                                  domain_folder=domain_folder, base_path=base_path)
    
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
        if os.path.exists(data_inf_path_TF_CDM):
            data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
            xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
            xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
            yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
            yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
            diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
            
            FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
            if os.path.exists(FNET_path_TF_CDM):
                FN_TF_CDM = FN_Net(input_dim=dimension*2, output_dim=dimension, hid_size=50).to(device)
                FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                FN_TF_CDM.eval()
    
    # Load FEX-VAE model if provided
    VAE_FEX = None
    if All_stage_dir_FEX_VAE is not None:
        print("[INFO] Loading FEX-VAE model...")
        VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
        if os.path.exists(VAE_path):
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
            VAE_FEX.eval()
    
    # Load FEX-NN model if provided
    FEX_NN = None
    if All_stage_dir_FEX_NN is not None:
        print("[INFO] Loading FEX-NN model...")
        FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
        if os.path.exists(FEX_NN_path):
            if dimension == 1:
                output_dim = 1  # Variance for 1D
            else:
                output_dim = dimension * dimension  # Flattened covariance matrix
            FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
            FEX_NN.eval()
    
    # Model colors and styles
    model_colors = {
        "FEX-DM": "orange",
        "TF-CDM": "steelblue",
        "FEX-VAE": "green",
        "FEX-NN": "purple"
    }
    
    model_linestyles = {
        "FEX-DM": "-",
        "TF-CDM": "--",
        "FEX-VAE": "-.",
        "FEX-NN": ":"
    }
    
    # Simulation parameters
    x_dim = dimension
    sde_T = 1.0
    sde_dt = 0.01
    Npath = 500000
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    for col, x0 in enumerate(initial_values):
        true_init = x0
        x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
        
        # Generate ground truth samples - drift depends on model
        ode_path_true = true_init * np.ones((Npath, x_dim))
        if model_name == 'DoubleWell1d':
            # Double Well: dX = (X - X^3)dt + sig*dB
            drift_true = ode_path_true - ode_path_true**3  # Drift: x - x^3
            true_samples = ode_path_true + drift_true * sde_dt + sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
        elif model_name == 'EXP1d':
            # EXP1d: dX = th * X * dt + sig * Exp(1) * sqrt(dt)
            # Use exponential noise instead of normal noise
            drift_true = theta * ode_path_true  # Drift: th * x
            true_samples = ode_path_true + drift_true * sde_dt + sigma * np.sqrt(sde_dt) * np.random.exponential(scale=1.0, size=(Npath, x_dim))
        else:
            # OU1d: dX = theta*(mu - X)dt + sigma*dB
            drift_true = theta * (mu - ode_path_true)
            true_samples = ode_path_true + drift_true * sde_dt + sigma * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
        
        # Define plotting range
        x_min, x_max = np.min(true_samples) - 0.05, np.max(true_samples) + 0.05
        x_vals = np.linspace(x_min, x_max, 200)
        
        # Compute KDE for true distribution
        kde = gaussian_kde(true_samples.T)
        pdf_vals = kde(x_vals)
        
        # Top row: PDF plots
        ax_pdf = axes[0, col]
        ax_pdf.plot(x_vals, pdf_vals, color='black', linewidth=2, linestyle='dashed', label="Ground Truth", zorder=3)
        
        # Generate the same random noise z for all models
        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
        
        # Model predictions - skip TF-CDM and FEX-NN for initial values outside training domain
        # Determine which models to compute for this initial value
        skip_TF_CDM = (true_init < domain_start or true_init > domain_end)
        skip_FEX_NN = (true_init < domain_start or true_init > domain_end)
        
        predictions = {}
        # Plot in order: TF-CDM, FEX-VAE, FEX-NN first (bottom layer), then FEX-DM last (top layer)
        for model in ["TF-CDM", "FEX-VAE", "FEX-NN", "FEX-DM"]:
            if model == "TF-CDM":
                if skip_TF_CDM:
                    continue  # Skip TF-CDM for initial values outside training domain
                if FN_TF_CDM is not None:
                    with torch.no_grad():
                        prediction = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                        prediction = (prediction / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                    predictions[model] = prediction
            elif model == "FEX-VAE":
                if VAE_FEX is not None:
                    with torch.no_grad():
                        prediction = VAE_FEX.decoder(z)
                        prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                    predictions[model] = prediction
            elif model == "FEX-NN":
                if skip_FEX_NN:
                    continue  # Skip FEX-NN for initial values outside training domain
                if FEX_NN is not None:
                    with torch.no_grad():
                        cov_pred = FEX_NN(x_pred_new)
                        if dimension == 1:
                            std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                            prediction = (x_pred_new.squeeze(-1) + FEX(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                            prediction = prediction[:, np.newaxis]
                        else:
                            cov_matrix = cov_pred.reshape(Npath, dimension, dimension)
                            cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                            try:
                                L = torch.linalg.cholesky(cov_matrix)
                                noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                            except:
                                noise = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=1, dim2=2), min=1e-8)) * z
                            prediction = (x_pred_new + FEX(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                    predictions[model] = prediction
            elif model == "FEX-DM":
                # Always plot FEX-DM last so it's on top
                with torch.no_grad():
                    prediction = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
                    prediction = (prediction / diff_scale_FEX + x_pred_new + FEX(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                predictions[model] = prediction
        
        # Plot PDFs for each model - FEX-DM should be on top (highest zorder)
        for model, prediction in predictions.items():
            # Set zorder: FEX-DM on top, others below
            if model == "FEX-DM":
                plot_zorder = 5  # FEX-DM on top
            else:
                plot_zorder = 2  # Other models below
            ax_pdf.hist(prediction, bins=50, density=True, alpha=0.5, color=model_colors[model],
                       histtype='stepfilled', edgecolor=model_colors[model], label=f"{model}", zorder=plot_zorder)
        
        ax_pdf.set_xlabel('$x$', fontsize=22)
        ax_pdf.set_ylabel('pdf', fontsize=22)
        ax_pdf.set_title(f'$x_0$ = {true_init:.2f}', fontsize=24)
        ax_pdf.set_xlim([x_min, x_max])
        ax_pdf.tick_params(axis='both', labelsize=22)
        ax_pdf.grid(False)
        
        # Bottom row: Error plots
        ax_err = axes[1, col]
        
        # Compute errors for each model - plot FEX-DM last so it's on top
        # Sort models so FEX-DM is plotted last
        sorted_models = sorted(predictions.keys(), key=lambda x: (x != "FEX-DM", x))
        
        for model in sorted_models:
            prediction = predictions[model]
            # Compute KDE for model prediction
            kde_pred = gaussian_kde(prediction.T)
            pdf_pred = kde_pred(x_vals)
            
            # Compute error: prediction PDF - ground truth PDF
            error = pdf_pred - pdf_vals
            
            # Set zorder: FEX-DM on top, others below
            if model == "FEX-DM":
                plot_zorder = 5  # FEX-DM on top
            else:
                plot_zorder = 2  # Other models below
            
            ax_err.plot(x_vals, error, color=model_colors[model], linewidth=2,
                       linestyle=model_linestyles[model], label=f"{model}", zorder=plot_zorder)
        
        # Add subtle zero line
        ax_err.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax_err.set_xlabel('$x$', fontsize=22)
        ax_err.set_ylabel('Error (pdf)', fontsize=22)
        ax_err.set_title(f'Error: $x_0$ = {true_init:.2f}', fontsize=24)
        ax_err.set_xlim([x_min, x_max])
        ax_err.tick_params(axis='both', labelsize=22)
        ax_err.grid(False)
    
    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color=model_colors["FEX-DM"], linewidth=6, label="FEX-DM"),
        plt.Line2D([0], [0], color=model_colors["TF-CDM"], linewidth=6, label="TF-CDM"),
        plt.Line2D([0], [0], color=model_colors["FEX-VAE"], linewidth=6, label="FEX-VAE"),
        plt.Line2D([0], [0], color=model_colors["FEX-NN"], linewidth=6, label="FEX-NN"),
        plt.Line2D([0], [0], color="black", linestyle="dashed", linewidth=2, label="Ground Truth")
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=5, fontsize=16, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'conditional_distribution_with_errors.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    else:
        print(f"[WARNING] File was not created at: {save_path}")
    
    return save_path


def plot_time_dependent_trajectory_error(results_dict,
                                        initial_values,
                                        num_steps,
                                        dt,
                                        dimension,
                                        save_dir,
                                        model_name='Trigonometric1d',
                                        models_dict=None,
                                        scaler=None,
                                        All_stage_dir_TF_CDM=None,
                                        All_stage_dir_FEX_VAE=None,
                                        All_stage_dir_FEX_NN=None,
                                        scaler_TF_CDM=None,
                                        base_path=None,
                                        noise_level=1.0,
                                        device='cpu',
                                        figsize=(18, 12),
                                        dpi=300):
    """
    Plot time-dependent trajectory comparison with error estimation.
    Creates a 2x3 layout: top row shows mean trajectories, bottom row shows error (pred - true).
    
    Args:
        results_dict: Dictionary mapping initial_value to dict with:
            - 'u_all_ground_truth': Ground truth trajectories (NPATH, dimension, num_steps+1)
            - 'u_pred_all_FEX': FEX-DM prediction trajectories (NPATH, dimension, num_steps+1)
        initial_values: List of initial values (e.g., [-3, 0.6, 3])
        num_steps: Number of time steps
        dt: Time step size
        dimension: Number of dimensions
        save_dir: Directory to save the plots
        model_name: Model name (default: 'Trigonometric1d')
        figsize: Figure size tuple (default: (18, 12))
        dpi: Resolution for saved figure (default: 300)
    
    Returns:
        List of paths to saved figure files
    """
    # Time mesh for plotting (starting from dt, not 0)
    tmesh = np.linspace(dt, num_steps * dt, num_steps)
    
    # Model style (similar to OU1d plot)
    model_style = {
        "FEX-DM": {"color": "orange", "fill": "orange", "linestyle": "-", "linewidth": 3},
        "TF-CDM": {"color": "steelblue", "fill": "steelblue", "linestyle": "-", "linewidth": 3},
        "FEX-VAE": {"color": "green", "fill": "green", "linestyle": "-", "linewidth": 3},
        "FEX-NN": {"color": "purple", "fill": "purple", "linestyle": "-", "linewidth": 3}
    }
    
    # Load other models if provided
    FN_TF_CDM = None
    models_dict_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = scaler_TF_CDM
    TF_CDM_time_dependent = False
    
    VAE_FEX = None
    models_dict_VAE = None
    diff_scale_FEX = scaler
    VAE_time_dependent = False
    
    FEX_NN = None
    models_dict_FEX_NN = None
    FEX_NN_time_dependent = False
    
    # FEX deterministic model wrapper
    def FEX_deterministic(x):
        return FEX_model_learned(x, 
                               model_name=model_name,
                               noise_level=noise_level,
                               device=str(device),
                               base_path=base_path)
    
    if All_stage_dir_TF_CDM is not None:
        print("[INFO] Loading TF-CDM models for trajectory plotting...")
        from utils.helper import load_time_dependent_TF_CDM_models
        models_dict_TF_CDM = load_time_dependent_TF_CDM_models(All_stage_dir_TF_CDM, dimension, device=device)
        if models_dict_TF_CDM:
            TF_CDM_time_dependent = True
            if scaler_TF_CDM is None:
                data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
                if os.path.exists(data_inf_path_TF_CDM):
                    data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
                    diff_scale_TF_CDM = data_inf_TF_CDM.get('diff_scale', None)
        else:
            data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
            if os.path.exists(data_inf_path_TF_CDM):
                data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
                xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
                xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
                yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
                yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
                if scaler_TF_CDM is None:
                    diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
                from utils.helper import FN_Net
                FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
                if os.path.exists(FNET_path_TF_CDM):
                    FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                    FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                    FN_TF_CDM.eval()
    
    if All_stage_dir_FEX_VAE is not None:
        print("[INFO] Loading FEX-VAE models for trajectory plotting...")
        from utils.helper import load_time_dependent_VAE_models
        models_dict_VAE = load_time_dependent_VAE_models(All_stage_dir_FEX_VAE, dimension, device=device)
        if models_dict_VAE:
            VAE_time_dependent = True
        else:
            VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
            if os.path.exists(VAE_path):
                from utils.helper import VAE
                VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
                VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
                VAE_FEX.eval()
    
    if All_stage_dir_FEX_NN is not None:
        print("[INFO] Loading FEX-NN models for trajectory plotting...")
        from utils.helper import load_time_dependent_FEX_NN_models
        models_dict_FEX_NN = load_time_dependent_FEX_NN_models(All_stage_dir_FEX_NN, dimension, device=device)
        if models_dict_FEX_NN:
            FEX_NN_time_dependent = True
        else:
            from utils.ODEParser import CovarianceNet
            FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
            if os.path.exists(FEX_NN_path):
                output_dim_nn = dimension * dimension if dimension > 1 else 1
                FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
                FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
                FEX_NN.eval()
    
    saved_paths = []
    
    # Track which models are available
    has_TF_CDM = (TF_CDM_time_dependent and models_dict_TF_CDM) or FN_TF_CDM is not None
    has_VAE = (VAE_time_dependent and models_dict_VAE) or VAE_FEX is not None
    has_NN = (FEX_NN_time_dependent and models_dict_FEX_NN) or FEX_NN is not None
    
    # Plot for each dimension - create 2x3 subplots (top: trajectories, bottom: errors)
    for dim_idx in range(dimension):
        # Create figure with 2 rows, 3 columns (one for each initial value)
        fig, axes = plt.subplots(2, len(initial_values), figsize=figsize)
        if len(initial_values) == 1:
            axes = axes.reshape(2, 1)
        
        # Process each initial value in its own column
        for col, initial_value in enumerate(initial_values):
            ax_mean = axes[0, col]  # Top row: mean trajectories
            ax_error = axes[1, col]  # Bottom row: errors
            
            u_all_ground_truth = results_dict[initial_value]['u_all_ground_truth']
            u_pred_all_FEX = results_dict[initial_value]['u_pred_all_FEX']
            
            NPATH = u_all_ground_truth.shape[0]
            
            # Initialize arrays for statistics
            ode_mean_pred_FEX = np.zeros(num_steps)
            ode_std_pred_FEX = np.zeros(num_steps)
            ode_mean_true = np.zeros(num_steps)
            ode_std_true = np.zeros(num_steps)
            
            # Initialize arrays for other models
            ode_mean_pred_TF_CDM = np.zeros(num_steps)
            ode_std_pred_TF_CDM = np.zeros(num_steps)
            ode_mean_pred_VAE = np.zeros(num_steps)
            ode_std_pred_VAE = np.zeros(num_steps)
            ode_mean_pred_NN = np.zeros(num_steps)
            ode_std_pred_NN = np.zeros(num_steps)
            
            # Initialize arrays for error statistics
            ode_error_mean_FEX = np.zeros(num_steps)
            ode_error_std_FEX = np.zeros(num_steps)
            ode_error_mean_TF_CDM = np.zeros(num_steps)
            ode_error_std_TF_CDM = np.zeros(num_steps)
            ode_error_mean_VAE = np.zeros(num_steps)
            ode_error_std_VAE = np.zeros(num_steps)
            ode_error_mean_NN = np.zeros(num_steps)
            ode_error_std_NN = np.zeros(num_steps)
            
            # Initialize prediction arrays for other models
            u_pred_all_TF_CDM = None
            u_pred_all_VAE = None
            u_pred_all_NN = None
            
            # Skip TF-CDM for initial values -3 and 3
            skip_TF_CDM = (abs(initial_value - (-3)) < 0.01 or abs(initial_value - 3) < 0.01)
            
            # Compute trajectories for other models if available
            if not skip_TF_CDM and ((TF_CDM_time_dependent and models_dict_TF_CDM) or FN_TF_CDM is not None):
                u_pred_all_TF_CDM = np.zeros((NPATH, dimension, num_steps + 1), dtype=np.float32)
                initial_state = initial_value * np.ones((NPATH, dimension))
                u_pred_all_TF_CDM[:, :, 0] = initial_state
                current_pred_state_TF_CDM = initial_state.copy()
            
            if (VAE_time_dependent and models_dict_VAE) or VAE_FEX is not None:
                u_pred_all_VAE = np.zeros((NPATH, dimension, num_steps + 1), dtype=np.float32)
                initial_state = initial_value * np.ones((NPATH, dimension))
                u_pred_all_VAE[:, :, 0] = initial_state
                current_pred_state_VAE = initial_state.copy()
            
            if (FEX_NN_time_dependent and models_dict_FEX_NN) or FEX_NN is not None:
                u_pred_all_NN = np.zeros((NPATH, dimension, num_steps + 1), dtype=np.float32)
                initial_state = initial_value * np.ones((NPATH, dimension))
                u_pred_all_NN[:, :, 0] = initial_state
                current_pred_state_NN = initial_state.copy()
            
            # Load SDE parameters
            if params_init is not None:
                model_params = params_init(case_name=model_name)
                sig = model_params['sig'] * noise_level
            else:
                sig = 1.0
            
            # Recompute trajectories for other models
            np.random.seed(42)  # Use same seed for reproducibility
            for idx in range(1, num_steps + 1):
                # Generate Wiener increments (same as ground truth simulation)
                Winc = np.random.randn(NPATH, dimension)
                Winc_tensor = torch.tensor(Winc, dtype=torch.float32).to(device)
                dW = np.sqrt(dt) * Winc
                if models_dict:
                    max_time_step = max(models_dict.keys()) if models_dict else 0
                    t_idx = min(idx - 1, max_time_step)
                else:
                    t_idx = 0
                
                # TF-CDM prediction
                if u_pred_all_TF_CDM is not None:
                    current_tensor_TF_CDM = torch.tensor(current_pred_state_TF_CDM, dtype=torch.float32).to(device)
                    if TF_CDM_time_dependent and models_dict_TF_CDM and t_idx in models_dict_TF_CDM:
                        FN_TF_CDM_t, norm_params_TF_CDM_t = models_dict_TF_CDM[t_idx]
                        with torch.no_grad():
                            x_mean_t = torch.tensor(norm_params_TF_CDM_t['x_mean'], dtype=torch.float32).to(device)
                            x_std_t = torch.tensor(norm_params_TF_CDM_t['x_std'], dtype=torch.float32).to(device)
                            y_mean_t = torch.tensor(norm_params_TF_CDM_t['mean'], dtype=torch.float32).to(device)
                            y_std_t = torch.tensor(norm_params_TF_CDM_t['std'], dtype=torch.float32).to(device)
                            xz_concat = torch.hstack((current_tensor_TF_CDM, Winc_tensor))
                            xz_normalized = (xz_concat - x_mean_t) / x_std_t
                            prediction_TF_CDM_normalized = FN_TF_CDM_t(xz_normalized)
                            prediction_TF_CDM = prediction_TF_CDM_normalized * y_std_t + y_mean_t
                            if diff_scale_TF_CDM is not None:
                                if isinstance(diff_scale_TF_CDM, np.ndarray):
                                    diff_scale_value = float(diff_scale_TF_CDM[0]) if len(diff_scale_TF_CDM) > 0 else 1.0
                                else:
                                    diff_scale_value = float(diff_scale_TF_CDM)
                                stoch_update_TF_CDM = (prediction_TF_CDM / diff_scale_value).cpu().numpy()
                            else:
                                stoch_update_TF_CDM = prediction_TF_CDM.cpu().numpy()
                    elif FN_TF_CDM is not None:
                        with torch.no_grad():
                            prediction_TF_CDM = FN_TF_CDM((torch.hstack((current_tensor_TF_CDM, Winc_tensor)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                            if diff_scale_TF_CDM is not None:
                                if isinstance(diff_scale_TF_CDM, np.ndarray):
                                    diff_scale_value = float(diff_scale_TF_CDM[0]) if len(diff_scale_TF_CDM) > 0 else 1.0
                                else:
                                    diff_scale_value = float(diff_scale_TF_CDM)
                                stoch_update_TF_CDM = (prediction_TF_CDM / diff_scale_value).cpu().numpy()
                            else:
                                stoch_update_TF_CDM = prediction_TF_CDM.cpu().numpy()
                    else:
                        stoch_update_TF_CDM = np.zeros((NPATH, dimension))
                    next_pred_state_TF_CDM = current_pred_state_TF_CDM + stoch_update_TF_CDM
                    u_pred_all_TF_CDM[:, :, idx] = next_pred_state_TF_CDM
                    current_pred_state_TF_CDM = next_pred_state_TF_CDM
                
                # FEX-VAE prediction
                if u_pred_all_VAE is not None:
                    current_tensor_VAE = torch.tensor(current_pred_state_VAE, dtype=torch.float32).to(device)
                    FEX_det_VAE = FEX_deterministic(current_tensor_VAE).cpu().numpy()
                    det_update_VAE = FEX_det_VAE * dt
                    if VAE_time_dependent and models_dict_VAE and t_idx in models_dict_VAE:
                        VAE_model_t = models_dict_VAE[t_idx]
                        with torch.no_grad():
                            prediction_VAE = VAE_model_t.decoder(Winc_tensor)
                            if isinstance(diff_scale_FEX, np.ndarray):
                                diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                            else:
                                diff_scale_VAE_value = float(diff_scale_FEX)
                            stoch_update_VAE = (prediction_VAE / diff_scale_VAE_value).cpu().numpy()
                    elif VAE_FEX is not None:
                        with torch.no_grad():
                            prediction_VAE = VAE_FEX.decoder(Winc_tensor)
                            if isinstance(diff_scale_FEX, np.ndarray):
                                diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                            else:
                                diff_scale_VAE_value = float(diff_scale_FEX)
                            stoch_update_VAE = (prediction_VAE / diff_scale_VAE_value).cpu().numpy()
                    else:
                        stoch_update_VAE = np.zeros((NPATH, dimension))
                    next_pred_state_VAE = current_pred_state_VAE + det_update_VAE + stoch_update_VAE
                    u_pred_all_VAE[:, :, idx] = next_pred_state_VAE
                    current_pred_state_VAE = next_pred_state_VAE
                
                # FEX-NN prediction
                if u_pred_all_NN is not None:
                    current_tensor_NN = torch.tensor(current_pred_state_NN, dtype=torch.float32).to(device)
                    FEX_det_NN = FEX_deterministic(current_tensor_NN).cpu().numpy()
                    det_update_NN = FEX_det_NN * dt
                    if FEX_NN_time_dependent and models_dict_FEX_NN and t_idx in models_dict_FEX_NN:
                        FEX_NN_t = models_dict_FEX_NN[t_idx]
                        with torch.no_grad():
                            cov_pred = FEX_NN_t(current_tensor_NN)
                            if dimension == 1:
                                std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                                noise = std_pred * Winc_tensor.squeeze(-1) * np.sqrt(dt)
                                stoch_update_NN = noise.cpu().numpy()[:, np.newaxis]
                            else:
                                cov_matrix = cov_pred.reshape(NPATH, dimension, dimension)
                                cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                                try:
                                    L = torch.linalg.cholesky(cov_matrix)
                                    noise = torch.bmm(L, Winc_tensor.unsqueeze(-1)).squeeze(-1)
                                except:
                                    diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)
                                    noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * Winc_tensor
                                stoch_update_NN = (noise * np.sqrt(dt)).cpu().numpy()
                    elif FEX_NN is not None:
                        with torch.no_grad():
                            cov_pred = FEX_NN(current_tensor_NN)
                            if dimension == 1:
                                std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                                noise = std_pred * Winc_tensor.squeeze(-1) * np.sqrt(dt)
                                stoch_update_NN = noise.cpu().numpy()[:, np.newaxis]
                            else:
                                cov_matrix = cov_pred.reshape(NPATH, dimension, dimension)
                                cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                                try:
                                    L = torch.linalg.cholesky(cov_matrix)
                                    noise = torch.bmm(L, Winc_tensor.unsqueeze(-1)).squeeze(-1)
                                except:
                                    diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)
                                    noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * Winc_tensor
                                stoch_update_NN = (noise * np.sqrt(dt)).cpu().numpy()
                    else:
                        stoch_update_NN = np.zeros((NPATH, dimension))
                    next_pred_state_NN = current_pred_state_NN + det_update_NN + stoch_update_NN
                    u_pred_all_NN[:, :, idx] = next_pred_state_NN
                    current_pred_state_NN = next_pred_state_NN
            
            # Compute statistics for each time step
            for jj in range(num_steps):
                # Ground truth mean and std
                ode_mean_true[jj] = np.mean(u_all_ground_truth[:, dim_idx, jj+1])
                ode_std_true[jj] = np.std(u_all_ground_truth[:, dim_idx, jj+1])
                # FEX prediction mean and std
                ode_mean_pred_FEX[jj] = np.mean(u_pred_all_FEX[:, dim_idx, jj+1])
                ode_std_pred_FEX[jj] = np.std(u_pred_all_FEX[:, dim_idx, jj+1])
                
                # Compute error (prediction - ground truth) for each sample
                error_samples_FEX = u_pred_all_FEX[:, dim_idx, jj+1] - u_all_ground_truth[:, dim_idx, jj+1]
                ode_error_mean_FEX[jj] = np.mean(error_samples_FEX)
                ode_error_std_FEX[jj] = np.std(error_samples_FEX)
                
                # Other models statistics
                if u_pred_all_TF_CDM is not None:
                    ode_mean_pred_TF_CDM[jj] = np.mean(u_pred_all_TF_CDM[:, dim_idx, jj+1])
                    ode_std_pred_TF_CDM[jj] = np.std(u_pred_all_TF_CDM[:, dim_idx, jj+1])
                    error_samples_TF_CDM = u_pred_all_TF_CDM[:, dim_idx, jj+1] - u_all_ground_truth[:, dim_idx, jj+1]
                    ode_error_mean_TF_CDM[jj] = np.mean(error_samples_TF_CDM)
                    ode_error_std_TF_CDM[jj] = np.std(error_samples_TF_CDM)
                
                if u_pred_all_VAE is not None:
                    ode_mean_pred_VAE[jj] = np.mean(u_pred_all_VAE[:, dim_idx, jj+1])
                    ode_std_pred_VAE[jj] = np.std(u_pred_all_VAE[:, dim_idx, jj+1])
                    error_samples_VAE = u_pred_all_VAE[:, dim_idx, jj+1] - u_all_ground_truth[:, dim_idx, jj+1]
                    ode_error_mean_VAE[jj] = np.mean(error_samples_VAE)
                    ode_error_std_VAE[jj] = np.std(error_samples_VAE)
                
                if u_pred_all_NN is not None:
                    ode_mean_pred_NN[jj] = np.mean(u_pred_all_NN[:, dim_idx, jj+1])
                    ode_std_pred_NN[jj] = np.std(u_pred_all_NN[:, dim_idx, jj+1])
                    error_samples_NN = u_pred_all_NN[:, dim_idx, jj+1] - u_all_ground_truth[:, dim_idx, jj+1]
                    ode_error_mean_NN[jj] = np.mean(error_samples_NN)
                    ode_error_std_NN[jj] = np.std(error_samples_NN)
            
            # ========== TOP ROW: Mean Trajectories ==========
            # Plot ground truth mean (black dashed line)
            ax_mean.plot(tmesh, ode_mean_true, linewidth=4, label="Mean of ground truth", 
                       color='black', linestyle=':', zorder=10)
            
            # Plot other models first (bottom layers)
            # Skip TF-CDM for initial values -3 and 3
            if u_pred_all_TF_CDM is not None and not skip_TF_CDM:
                style = model_style["TF-CDM"]
                ax_mean.plot(tmesh, ode_mean_pred_TF_CDM, label="Pred Mean (TF-CDM)",
                           color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"], zorder=2)
                ax_mean.fill_between(tmesh, ode_mean_pred_TF_CDM - ode_std_pred_TF_CDM,
                                   ode_mean_pred_TF_CDM + ode_std_pred_TF_CDM,
                                   color=style["fill"], alpha=0.2, zorder=1)
            
            if u_pred_all_VAE is not None:
                style = model_style["FEX-VAE"]
                ax_mean.plot(tmesh, ode_mean_pred_VAE, label="Pred Mean (FEX-VAE)",
                           color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"], zorder=3)
                ax_mean.fill_between(tmesh, ode_mean_pred_VAE - ode_std_pred_VAE,
                                   ode_mean_pred_VAE + ode_std_pred_VAE,
                                   color=style["fill"], alpha=0.2, zorder=1)
            
            if u_pred_all_NN is not None:
                style = model_style["FEX-NN"]
                ax_mean.plot(tmesh, ode_mean_pred_NN, label="Pred Mean (FEX-NN)",
                           color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"], zorder=4)
                ax_mean.fill_between(tmesh, ode_mean_pred_NN - ode_std_pred_NN,
                                   ode_mean_pred_NN + ode_std_pred_NN,
                                   color=style["fill"], alpha=0.2, zorder=1)
            
            # Plot FEX-DM prediction mean (on top)
            style = model_style["FEX-DM"]
            ax_mean.plot(tmesh, ode_mean_pred_FEX, label="Pred Mean (FEX-DM)",
                       color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"], zorder=5)
            
            # Fill between for FEX prediction std
            ax_mean.fill_between(tmesh, ode_mean_pred_FEX - ode_std_pred_FEX,
                               ode_mean_pred_FEX + ode_std_pred_FEX,
                               color=style["fill"], alpha=0.2, zorder=1)
            
            # Labels & Titles for top subplot
            ax_mean.set_xlabel('Time', fontsize=20)
            ax_mean.set_ylabel('Value', fontsize=20)
            ax_mean.set_title(f'$x_0$ = {initial_value:.2f}', fontsize=20)
            ax_mean.tick_params(axis='both', labelsize=18)
            ax_mean.grid(False)
            
            # ========== BOTTOM ROW: Error Plots ==========
            # Plot horizontal line at y=0
            ax_error.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
            
            # Plot other models' errors first (bottom layers)
            # Skip TF-CDM for initial values -3 and 3
            if u_pred_all_TF_CDM is not None and not skip_TF_CDM:
                style = model_style["TF-CDM"]
                ax_error.plot(tmesh, ode_error_mean_TF_CDM, label="TF-CDM",
                             color=style["color"], linestyle=style["linestyle"], 
                             linewidth=style["linewidth"], zorder=2)
                ax_error.fill_between(tmesh, ode_error_mean_TF_CDM - ode_error_std_TF_CDM,
                                     ode_error_mean_TF_CDM + ode_error_std_TF_CDM,
                                     color=style["fill"], alpha=0.2, zorder=1)
            
            if u_pred_all_VAE is not None:
                style = model_style["FEX-VAE"]
                ax_error.plot(tmesh, ode_error_mean_VAE, label="FEX-VAE",
                             color=style["color"], linestyle=style["linestyle"], 
                             linewidth=style["linewidth"], zorder=3)
                ax_error.fill_between(tmesh, ode_error_mean_VAE - ode_error_std_VAE,
                                     ode_error_mean_VAE + ode_error_std_VAE,
                                     color=style["fill"], alpha=0.2, zorder=1)
            
            if u_pred_all_NN is not None:
                style = model_style["FEX-NN"]
                ax_error.plot(tmesh, ode_error_mean_NN, label="FEX-NN",
                             color=style["color"], linestyle=style["linestyle"], 
                             linewidth=style["linewidth"], zorder=4)
                ax_error.fill_between(tmesh, ode_error_mean_NN - ode_error_std_NN,
                                     ode_error_mean_NN + ode_error_std_NN,
                                     color=style["fill"], alpha=0.2, zorder=1)
            
            # Plot FEX-DM error (on top)
            style = model_style["FEX-DM"]
            ax_error.plot(tmesh, ode_error_mean_FEX, label="FEX-DM",
                         color=style["color"], linestyle=style["linestyle"], 
                         linewidth=style["linewidth"], zorder=5)
            
            # Fill between for error std
            ax_error.fill_between(tmesh, ode_error_mean_FEX - ode_error_std_FEX,
                                 ode_error_mean_FEX + ode_error_std_FEX,
                                 color=style["fill"], alpha=0.2, zorder=1)
            
            # Labels & Titles for bottom subplot
            ax_error.set_xlabel('Time', fontsize=20)
            ax_error.set_ylabel('Error (Pred - True)', fontsize=20)
            ax_error.tick_params(axis='both', labelsize=18)
            ax_error.grid(False)
        
        # Create legend (similar to OU1d plot style)
        legend_handles = [
            plt.Line2D([0], [0], color='black', linestyle=':', linewidth=4, label='Mean of ground truth')
        ]
        # Include TF-CDM in legend if it's available (it will be drawn for some initial values like 0.6)
        if has_TF_CDM:
            legend_handles.append(plt.Line2D([0], [0], color='steelblue', linestyle='-', linewidth=3, label='Pred Mean (TF-CDM)'))
        if has_VAE:
            legend_handles.append(plt.Line2D([0], [0], color='green', linestyle='-', linewidth=3, label='Pred Mean (FEX-VAE)'))
        if has_NN:
            legend_handles.append(plt.Line2D([0], [0], color='purple', linestyle='-', linewidth=3, label='Pred Mean (FEX-NN)'))
        legend_handles.append(plt.Line2D([0], [0], color='orange', linestyle='-', linewidth=3, label='Pred Mean (FEX-DM)'))
        
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                  ncol=len(legend_handles), fontsize=18)
        
        # Save and show
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_dir, f'prediction_comparison_dim{dim_idx+1}_3subplots_with_errors.pdf')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        saved_paths.append(save_path)
        print(f"[INFO] Saved 2x3-subplot figure (with errors) for dimension {dim_idx+1}")
    
    return saved_paths


def plot_drift_and_diffusion_time_dependent(second_stage_dir_FEX,
                                           models_dict,
                                           scaler,
                                           model_name='Trigonometric1d',
                                           All_stage_dir_TF_CDM=None,
                                           All_stage_dir_FEX_VAE=None,
                                           All_stage_dir_FEX_NN=None,
                                           scaler_TF_CDM=None,
                                           noise_level=1.0,
                                           device='cpu',
                                           base_path=None,
                                           Npath=5000,
                                           N_x0=500,
                                           x_min=-5,
                                           x_max=5,
                                           time_steps_to_plot=None,
                                           save_dir=None,
                                           figsize=(18, 12),
                                           dpi=300,
                                           seed=42):
    """
    Plot drift and diffusion for time-dependent FEX-DM models (with optional TF-CDM and VAE).
    - Drift at t=50 and t=100 (as function of x)
    - All sigmax_pred for all time steps (as function of time)
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        models_dict: Dictionary from load_time_dependent_models mapping time step to (model, norm_params)
        scaler: Scaling factor (if None, will load from data_inf.pt)
        model_name: Model name (e.g., 'Trigonometric1d')
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results (time-independent)
        All_stage_dir_FEX_VAE: Optional directory path for FEX-VAE second stage results (time-independent)
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        base_path: Base path for loading FEX deterministic model
        Npath: Number of paths for Monte Carlo simulation (default: 5000)
        N_x0: Number of initial values in grid (default: 500)
        x_min: Minimum state value (default: -5)
        x_max: Maximum state value (default: 5)
        time_steps_to_plot: List of time step indices to plot (default: all available)
        save_dir: Directory to save the figure
        figsize: Figure size tuple (default: (18, 12))
        dpi: Resolution for saved figure (default: 300)
        seed: Random seed for reproducibility
    
    Returns:
        str: Path to the saved figure file
    """
    # Load SDE parameters
    if params_init is not None:
        model_params = params_init(case_name=model_name)
        sigma_base = model_params['sig']
        sde_params = {
            'sig': sigma_base * noise_level,
            'sde_dt': model_params['Dt']
        }
    else:
        raise ValueError("params_init is not available")
    
    sig = sde_params['sig']
    sde_dt = sde_params['sde_dt']
    
    if save_dir is None:
        parent_dir = os.path.dirname(second_stage_dir_FEX)
        save_dir = os.path.join(parent_dir, 'plot')
    
    os.makedirs(save_dir, exist_ok=True)
    


    if scaler is None:
        raise ValueError("scaler is not provided")
    
    # Get available time steps
    available_time_steps = sorted(models_dict.keys())
    if not available_time_steps:
        raise ValueError("No time-dependent models found in models_dict")
    
    total_time_steps = max(available_time_steps) + 1
    
    # Select time steps to compute (all available if not specified)
    if time_steps_to_plot is None:
        time_steps_to_plot = available_time_steps
    else:
        # Map to available time steps
        time_steps_to_plot = [min(available_time_steps, key=lambda x: abs(x - t)) for t in time_steps_to_plot]
        time_steps_to_plot = sorted(list(set(time_steps_to_plot)))  # Remove duplicates and sort
    
    print(f"[INFO] Computing drift and diffusion for {len(time_steps_to_plot)} time steps...")
    
    # Create FEX deterministic model wrapper
    def FEX_deterministic(x):
        return FEX_model_learned(x, 
                                 model_name=model_name,
                                 noise_level=noise_level,
                                 device=device,
                                 base_path=base_path)
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Get dimension from first model
    first_model, first_norm = models_dict[available_time_steps[0]]
    dimension = first_model.input_dim
    
    # Load TF-CDM models (check for time-dependent first, then fall back to time-independent)
    FN_TF_CDM = None
    models_dict_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    # Use scaler_TF_CDM parameter instead of loading from file (scaler_TF_CDM is passed to function)
    diff_scale_TF_CDM = scaler_TF_CDM
    TF_CDM_time_dependent = False
    
    if All_stage_dir_TF_CDM is not None:
        print("[INFO] Loading TF-CDM models...")
        # Check for time-dependent TF-CDM models first
        from utils.helper import load_time_dependent_TF_CDM_models
        models_dict_TF_CDM = load_time_dependent_TF_CDM_models(All_stage_dir_TF_CDM, dimension, device=device)
        
        if models_dict_TF_CDM:
            print("[INFO] Found time-dependent TF-CDM models")
            TF_CDM_time_dependent = True
            # Use scaler_TF_CDM parameter if provided, otherwise try to load from file
            if scaler_TF_CDM is None:
                # Load diff_scale from data_inf.pt if it exists
                data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
                if os.path.exists(data_inf_path_TF_CDM):
                    data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
                    diff_scale_TF_CDM = data_inf_TF_CDM.get('diff_scale', None)
        else:
            # Fall back to time-independent model
            print("[INFO] No time-dependent TF-CDM models found, trying time-independent model...")
            data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
            if not os.path.exists(data_inf_path_TF_CDM):
                print(f"[WARNING] TF-CDM data_inf.pt not found at {data_inf_path_TF_CDM}, skipping TF-CDM")
            else:
                data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
                xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
                xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
                yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
                yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
                # Use scaler_TF_CDM parameter if provided, otherwise load from file
                if scaler_TF_CDM is None:
                    diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
                
                FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
                if os.path.exists(FNET_path_TF_CDM):
                    FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                    FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                    FN_TF_CDM.eval()
                else:
                    print(f"[WARNING] TF-CDM FNET.pth not found at {FNET_path_TF_CDM}, skipping TF-CDM")
                    FN_TF_CDM = None
    
    # Load FEX-VAE models (check for time-dependent first, then fall back to time-independent)
    VAE_FEX = None
    models_dict_VAE = None
    # Use scaler parameter instead of loading from file (scaler is passed to function)
    diff_scale_FEX = scaler
    VAE_time_dependent = False
    
    if All_stage_dir_FEX_VAE is not None:
        print("[INFO] Loading FEX-VAE models...")
        # Check for time-dependent VAE models first
        from utils.helper import load_time_dependent_VAE_models
        models_dict_VAE = load_time_dependent_VAE_models(All_stage_dir_FEX_VAE, dimension, device=device)
        
        if models_dict_VAE:
            print("[INFO] Found time-dependent FEX-VAE models")
            VAE_time_dependent = True
            # Load diff_scale from FEX data_inf.pt
            data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
            if os.path.exists(data_inf_path_FEX):
                data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
                diff_scale_FEX = data_inf_FEX['diff_scale']
        else:
            # Fall back to time-independent model
            print("[INFO] No time-dependent FEX-VAE models found, trying time-independent model...")
            VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
            if os.path.exists(VAE_path):
                from utils.helper import VAE
                VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
                VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
                VAE_FEX.eval()
                
                # Load diff_scale from FEX data_inf.pt
                data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
                if os.path.exists(data_inf_path_FEX):
                    data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
                    diff_scale_FEX = data_inf_FEX['diff_scale']
            else:
                print(f"[WARNING] FEX-VAE VAE_FEX.pth not found at {VAE_path}, skipping FEX-VAE")
                VAE_FEX = None
    
    # Load FEX-NN models (check for time-dependent first, then fall back to time-independent)
    FEX_NN = None
    models_dict_FEX_NN = None
    FEX_NN_time_dependent = False
    
    if All_stage_dir_FEX_NN is not None:
        print("[INFO] Loading FEX-NN models...")
        # Check for time-dependent FEX-NN models first
        from utils.helper import load_time_dependent_FEX_NN_models
        models_dict_FEX_NN = load_time_dependent_FEX_NN_models(All_stage_dir_FEX_NN, dimension, device=device)
        
        if models_dict_FEX_NN:
            print("[INFO] Found time-dependent FEX-NN models")
            FEX_NN_time_dependent = True
        else:
            # Fall back to time-independent model
            print("[INFO] No time-dependent FEX-NN models found, trying time-independent model...")
            FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
            if os.path.exists(FEX_NN_path):
                from utils.ODEParser import CovarianceNet
                output_dim_nn = dimension * dimension if dimension > 1 else 1
                FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
                FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
                FEX_NN.eval()
            else:
                print(f"[WARNING] FEX-NN FEX_NN.pth not found at {FEX_NN_path}, skipping FEX-NN")
                FEX_NN = None
    
    # Extract domain folder from second_stage_dir_FEX path
    domain_folder = None
    domain_start = 0.0
    domain_end = 1.0
    if second_stage_dir_FEX:
        path_parts = second_stage_dir_FEX.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                # Parse domain_start and domain_end from domain_folder (e.g., "domain_0.0_1.0")
                try:
                    parts = part.replace('domain_', '').split('_')
                    if len(parts) >= 2:
                        domain_start = float(parts[0])
                        domain_end = float(parts[1])
                except:
                    # If parsing fails, use defaults
                    pass
                break
    
    # Adjust x_min and x_max to extend by domain width on each side
    domain_width = domain_end - domain_start
    x_min = domain_start - domain_width
    x_max = domain_end + domain_width
    
    x_dim = dimension
    x0_grid = np.linspace(x_min, x_max, N_x0)  # Initial values on x-axis
    
    # Initialize arrays: drift for each time step and x0, diffusion for each time step only
    # Shape: (num_time_steps, N_x0) for drift, (num_time_steps,) for diffusion
    num_time_steps = len(time_steps_to_plot)
    bx_pred_all = np.zeros((num_time_steps, N_x0))  # FEX-DM drift for all time steps and x0
    sigmax_pred_all = np.zeros(num_time_steps)  # FEX-DM diffusion for each time step (function of time only)
    
    # TF-CDM and VAE predictions
    # TF-CDM and VAE can be time-dependent or time-independent
    bx_pred_TF_CDM = None
    sigmax_pred_TF_CDM = None
    bx_pred_VAE = None
    sigmax_pred_VAE = None
    
    if TF_CDM_time_dependent and models_dict_TF_CDM:
        # Time-dependent TF-CDM: separate predictions for each time step
        bx_pred_TF_CDM = np.zeros((num_time_steps, N_x0))  # (num_time_steps, N_x0)
        sigmax_pred_TF_CDM = np.zeros(num_time_steps)  # (num_time_steps,)
    elif FN_TF_CDM is not None:
        # Time-independent TF-CDM: single array
        bx_pred_TF_CDM = np.zeros(N_x0)
        sigmax_pred_TF_CDM = 0.0  # Single value (time-independent)
    
    if VAE_time_dependent and models_dict_VAE:
        # Time-dependent VAE: separate predictions for each time step
        bx_pred_VAE = np.zeros((num_time_steps, N_x0))  # (num_time_steps, N_x0)
        sigmax_pred_VAE = np.zeros(num_time_steps)  # (num_time_steps,)
    elif VAE_FEX is not None:
        # Time-independent VAE: single array
        bx_pred_VAE = np.zeros(N_x0)
        sigmax_pred_VAE = 0.0  # Single value (time-independent)
    
    # FEX-NN predictions (time-dependent or time-independent)
    bx_pred_NN = None
    sigmax_pred_NN = None
    if FEX_NN_time_dependent and models_dict_FEX_NN:
        # Time-dependent FEX-NN: separate predictions for each time step
        bx_pred_NN = np.zeros((num_time_steps, N_x0))  # (num_time_steps, N_x0)
        sigmax_pred_NN = np.zeros(num_time_steps)  # (num_time_steps,)
    elif FEX_NN is not None:
        # Time-independent FEX-NN: single array
        bx_pred_NN = np.zeros((num_time_steps, N_x0))  # Store for each time step (but use same model)
        sigmax_pred_NN = np.zeros(num_time_steps)  # Store for each time step
    
    # True drift (function of x, not time)
    if model_name == 'Trigonometric1d':
        k = 1  # frequency parameter
        bx_true = np.sin(2 * k * np.pi * x0_grid)  # Drift: sin(2*k*pi*X_t) - function of X_t

    
    # True diffusion (function of time, not x)
    time_values = np.array([t * sde_dt for t in time_steps_to_plot])
    if model_name == 'Trigonometric1d':
        k = 1
        sigmax_true_all = np.abs(sig * np.cos(2 * k * np.pi * time_values))  # Diffusion: sig*cos(2*k*pi*t) (no abs)

    
    # Compute drift and diffusion for each time step and each initial value
    print(f"[INFO] Computing drift and diffusion for {num_time_steps} time steps and {N_x0} initial values...")
    for t_idx, t in enumerate(time_steps_to_plot):
        if t_idx % 10 == 0:
            print(f"[INFO] Processing time step {t_idx+1}/{num_time_steps} (t={t})...")
        
        # Get model for this time step
        if t not in models_dict:
            # Find nearest available time step
            nearest_t = min(available_time_steps, key=lambda x: abs(x - t))
            print(f'[WARNING] Model for time step {t} not found, using nearest time step {nearest_t}')
            t = nearest_t
        
        FN, norm_params = models_dict[t]
        
        # Get normalization parameters
        # For time-dependent models, input is only Wiener increments (z), not [x, z]
        if 'x_mean' in norm_params and 'x_std' in norm_params:
            xTrain_mean = torch.tensor(norm_params['x_mean'], dtype=torch.float32).to(device)
            xTrain_std = torch.tensor(norm_params['x_std'], dtype=torch.float32).to(device)
        else:
            # Default: input is just Wiener increments, so shape is (dim,)
            xTrain_mean = torch.zeros(x_dim, dtype=torch.float32).to(device)
            xTrain_std = torch.ones(x_dim, dtype=torch.float32).to(device)
        
        yTrain_mean = torch.tensor(norm_params['mean'], dtype=torch.float32).to(device)
        yTrain_std = torch.tensor(norm_params['std'], dtype=torch.float32).to(device)
        
        # Collect predictions from all initial values to compute diffusion (function of time only)
        all_predictions = []
        all_true_inits = []
        all_bx_pred = []
        
        # Loop over initial values
        for jj in range(N_x0):
            true_init = x0_grid[jj]
            x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
            
            # Generate random noise (Wiener increments)
            z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
            
            # FEX-DM Prediction (matching time-independent format)
            with torch.no_grad():
                # Model takes only Wiener increments as input (matching time-independent format)
                prediction_FEX = FN((z - xTrain_mean) / xTrain_std) * yTrain_std + yTrain_mean
                
                # Add FEX deterministic update and apply diff_scale (matching time-independent format)
                FEX_det = FEX_deterministic(x_pred_new)  # Returns tensor
                # Convert scaler to scalar value for division
                if isinstance(scaler, np.ndarray):
                    scaler_value = float(scaler[0]) if len(scaler) > 0 else 1.0
                else:
                    scaler_value = float(scaler)
                
                # Match time-independent format: prediction / diff_scale + x_pred_new + FEX(x_pred_new) * sde_dt
                prediction = (prediction_FEX / scaler_value + x_pred_new + FEX_det * sde_dt).to('cpu').detach().numpy()
            
            # Compute drift: mean((prediction - x0) / dt)
            bx_pred_all[t_idx, jj] = np.mean((prediction - true_init) / sde_dt)
            
            # Store predictions and initial values for diffusion computation (averaged over all x0)
            all_predictions.append(prediction)
            all_true_inits.append(true_init)
            all_bx_pred.append(bx_pred_all[t_idx, jj])
            
            # TF-CDM Prediction
            if TF_CDM_time_dependent and models_dict_TF_CDM:
                # Time-dependent TF-CDM: use model for current time step
                if t in models_dict_TF_CDM:
                    FN_TF_CDM_t, norm_params_TF_CDM_t = models_dict_TF_CDM[t]
                    with torch.no_grad():
                        # Get normalization parameters
                        x_mean_t = torch.tensor(norm_params_TF_CDM_t['x_mean'], dtype=torch.float32).to(device)
                        x_std_t = torch.tensor(norm_params_TF_CDM_t['x_std'], dtype=torch.float32).to(device)
                        y_mean_t = torch.tensor(norm_params_TF_CDM_t['mean'], dtype=torch.float32).to(device)
                        y_std_t = torch.tensor(norm_params_TF_CDM_t['std'], dtype=torch.float32).to(device)
                        
                        # Concatenate current_state and z, then normalize
                        xz_concat = torch.hstack((x_pred_new, z))
                        xz_normalized = (xz_concat - x_mean_t) / x_std_t
                        
                        # Predict and denormalize
                        prediction_TF_CDM_normalized = FN_TF_CDM_t(xz_normalized)
                        prediction_TF_CDM = prediction_TF_CDM_normalized * y_std_t + y_mean_t
                        
                        # Apply diff_scale and add current state
                        if diff_scale_TF_CDM is not None:
                            if isinstance(diff_scale_TF_CDM, np.ndarray):
                                diff_scale_value = float(diff_scale_TF_CDM[0]) if len(diff_scale_TF_CDM) > 0 else 1.0
                            else:
                                diff_scale_value = float(diff_scale_TF_CDM)
                            prediction_TF_CDM = (prediction_TF_CDM / diff_scale_value + x_pred_new).to('cpu').detach().numpy()
                        else:
                            prediction_TF_CDM = (prediction_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                    
                    bx_pred_TF_CDM[t_idx, jj] = np.mean((prediction_TF_CDM - true_init) / sde_dt)
            elif FN_TF_CDM is not None and t_idx == 0:
                # Time-independent TF-CDM: compute only once at first time step
                with torch.no_grad():
                    prediction_TF_CDM = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                    prediction_TF_CDM = (prediction_TF_CDM / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                
                bx_pred_TF_CDM[jj] = np.mean((prediction_TF_CDM - true_init) / sde_dt)
            
            # FEX-VAE Prediction
            if VAE_time_dependent and models_dict_VAE:
                # Time-dependent VAE: use model for current time step
                if t in models_dict_VAE:
                    VAE_model_t = models_dict_VAE[t]
                    with torch.no_grad():
                        # Use the same z as FEX-DM, decode with VAE, then apply formula
                        prediction_VAE = VAE_model_t.decoder(z)
                        if isinstance(diff_scale_FEX, np.ndarray):
                            diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                        else:
                            diff_scale_VAE_value = float(diff_scale_FEX)
                        prediction_VAE = (prediction_VAE / diff_scale_VAE_value + x_pred_new + FEX_det * sde_dt).to('cpu').detach().numpy()
                    
                    bx_pred_VAE[t_idx, jj] = np.mean((prediction_VAE - true_init) / sde_dt)
            elif VAE_FEX is not None and t_idx == 0:
                # Time-independent VAE: compute only once at first time step
                with torch.no_grad():
                    # Use the same z as FEX-DM, decode with VAE, then apply formula
                    prediction_VAE = VAE_FEX.decoder(z)
                    if isinstance(diff_scale_FEX, np.ndarray):
                        diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                    else:
                        diff_scale_VAE_value = float(diff_scale_FEX)
                    prediction_VAE = (prediction_VAE / diff_scale_VAE_value + x_pred_new + FEX_det * sde_dt).to('cpu').detach().numpy()
                
                bx_pred_VAE[jj] = np.mean((prediction_VAE - true_init) / sde_dt)
            
            # FEX-NN Prediction
            if FEX_NN_time_dependent and models_dict_FEX_NN:
                # Time-dependent FEX-NN: use model for current time step
                if t in models_dict_FEX_NN:
                    FEX_NN_t = models_dict_FEX_NN[t]
                    with torch.no_grad():
                        # Predict covariance matrix from current state
                        cov_pred = FEX_NN_t(x_pred_new)  # (Npath, dim*dim) or (Npath, 1) for 1D
                        if x_dim == 1:
                            # 1D case: cov_pred is (Npath, 1), use as variance
                            std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)  # (Npath,)
                            prediction_NN = (x_pred_new.squeeze(-1) + FEX_det.squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                            prediction_NN = prediction_NN[:, np.newaxis]  # (Npath, 1)
                        else:
                            # Multi-D case: reshape to (Npath, dim, dim) and sample
                            cov_matrix = cov_pred.reshape(Npath, x_dim, x_dim)  # (Npath, dim, dim)
                            # Ensure positive semi-definite by adding small identity
                            cov_matrix = cov_matrix + 1e-6 * torch.eye(x_dim, device=device).unsqueeze(0)
                            # Sample from multivariate normal using Cholesky
                            try:
                                L = torch.linalg.cholesky(cov_matrix)  # (Npath, dim, dim)
                                noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)  # (Npath, dim)
                            except:
                                # Fallback: use diagonal approximation
                                diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)  # (Npath, dim)
                                noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * z  # (Npath, dim)
                            prediction_NN = (x_pred_new + FEX_det * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                        
                        bx_pred_NN[t_idx, jj] = np.mean((prediction_NN - true_init) / sde_dt)
            elif FEX_NN is not None:
                # Time-independent FEX-NN: use same model for all time steps
                with torch.no_grad():
                    # Predict covariance matrix from current state
                    cov_pred = FEX_NN(x_pred_new)  # (Npath, dim*dim) or (Npath, 1) for 1D
                    if x_dim == 1:
                        # 1D case: cov_pred is (Npath, 1), use as variance
                        std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)  # (Npath,)
                        prediction_NN = (x_pred_new.squeeze(-1) + FEX_det.squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                        prediction_NN = prediction_NN[:, np.newaxis]  # (Npath, 1)
                    else:
                        # Multi-D case: reshape to (Npath, dim, dim) and sample
                        cov_matrix = cov_pred.reshape(Npath, x_dim, x_dim)  # (Npath, dim, dim)
                        # Ensure positive semi-definite by adding small identity
                        cov_matrix = cov_matrix + 1e-6 * torch.eye(x_dim, device=device).unsqueeze(0)
                        # Sample from multivariate normal using Cholesky
                        try:
                            L = torch.linalg.cholesky(cov_matrix)  # (Npath, dim, dim)
                            noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)  # (Npath, dim)
                        except:
                            # Fallback: use diagonal approximation
                            diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)  # (Npath, dim)
                            noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * z  # (Npath, dim)
                        prediction_NN = (x_pred_new + FEX_det * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                    
                    bx_pred_NN[t_idx, jj] = np.mean((prediction_NN - true_init) / sde_dt)
            
            # Print progress for first and last time steps
            if t_idx == 0 and jj % 50 == 0:
                print(f"  jj={jj}, bx_true={bx_true[jj]:.4f}, bx_pred={bx_pred_all[t_idx, jj]:.4f}")
        
        # Compute diffusion for this time step (averaged over all initial values)
        # Concatenate all predictions and initial values
        all_predictions_concat = np.concatenate(all_predictions, axis=0)  # (Npath * N_x0, x_dim)
        all_true_inits_concat = np.concatenate([np.full((Npath, x_dim), init) for init in all_true_inits], axis=0)  # (Npath * N_x0, x_dim)
        all_bx_pred_concat = np.concatenate([np.full((Npath,), bx) for bx in all_bx_pred], axis=0)  # (Npath * N_x0,)
        
        # Compute diffusion: std((prediction - x0 - bx*dt)) * sqrt(1/dt) averaged over all x0
        residual = all_predictions_concat - all_true_inits_concat - all_bx_pred_concat[:, np.newaxis] * sde_dt
        sigmax_pred_all[t_idx] = np.mean(np.std(residual, axis=0)) * np.sqrt(1 / sde_dt)
        
        # Compute TF-CDM diffusion for current time step
        if TF_CDM_time_dependent and models_dict_TF_CDM:
            # Time-dependent TF-CDM: compute diffusion for current time step
            if t in models_dict_TF_CDM:
                all_predictions_TF_CDM = []
                all_bx_pred_TF_CDM = []
                FN_TF_CDM_t, norm_params_TF_CDM_t = models_dict_TF_CDM[t]
                x_mean_t = torch.tensor(norm_params_TF_CDM_t['x_mean'], dtype=torch.float32).to(device)
                x_std_t = torch.tensor(norm_params_TF_CDM_t['x_std'], dtype=torch.float32).to(device)
                y_mean_t = torch.tensor(norm_params_TF_CDM_t['mean'], dtype=torch.float32).to(device)
                y_std_t = torch.tensor(norm_params_TF_CDM_t['std'], dtype=torch.float32).to(device)
                
                for jj in range(N_x0):
                    true_init = x0_grid[jj]
                    x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
                    z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                    with torch.no_grad():
                        xz_concat = torch.hstack((x_pred_new, z))
                        xz_normalized = (xz_concat - x_mean_t) / x_std_t
                        prediction_TF_CDM_normalized = FN_TF_CDM_t(xz_normalized)
                        prediction_TF_CDM = prediction_TF_CDM_normalized * y_std_t + y_mean_t
                        if diff_scale_TF_CDM is not None:
                            if isinstance(diff_scale_TF_CDM, np.ndarray):
                                diff_scale_value = float(diff_scale_TF_CDM[0]) if len(diff_scale_TF_CDM) > 0 else 1.0
                            else:
                                diff_scale_value = float(diff_scale_TF_CDM)
                            prediction_TF_CDM = (prediction_TF_CDM / diff_scale_value + x_pred_new).to('cpu').detach().numpy()
                        else:
                            prediction_TF_CDM = (prediction_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                    all_predictions_TF_CDM.append(prediction_TF_CDM)
                    all_bx_pred_TF_CDM.append(bx_pred_TF_CDM[t_idx, jj])
                
                all_predictions_TF_CDM_concat = np.concatenate(all_predictions_TF_CDM, axis=0)
                all_true_inits_TF_CDM_concat = np.concatenate([np.full((Npath, x_dim), x0_grid[jj]) for jj in range(N_x0)], axis=0)
                all_bx_pred_TF_CDM_concat = np.concatenate([np.full((Npath,), bx) for bx in all_bx_pred_TF_CDM], axis=0)
                residual_TF_CDM = all_predictions_TF_CDM_concat - all_true_inits_TF_CDM_concat - all_bx_pred_TF_CDM_concat[:, np.newaxis] * sde_dt
                sigmax_pred_TF_CDM[t_idx] = np.mean(np.std(residual_TF_CDM, axis=0)) * np.sqrt(1 / sde_dt)
        
        # Compute VAE diffusion for current time step (if time-dependent)
        if VAE_time_dependent and models_dict_VAE:
            # Time-dependent VAE: compute diffusion for current time step
            if t in models_dict_VAE:
                all_predictions_VAE = []
                all_bx_pred_VAE = []
                VAE_model_t = models_dict_VAE[t]
                for jj in range(N_x0):
                    true_init = x0_grid[jj]
                    x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
                    z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                    with torch.no_grad():
                        prediction_VAE = VAE_model_t.decoder(z)
                        if isinstance(diff_scale_FEX, np.ndarray):
                            diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                        else:
                            diff_scale_VAE_value = float(diff_scale_FEX)
                        prediction_VAE = (prediction_VAE / diff_scale_VAE_value + x_pred_new + FEX_deterministic(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                    all_predictions_VAE.append(prediction_VAE)
                    all_bx_pred_VAE.append(bx_pred_VAE[t_idx, jj])
                
                all_predictions_VAE_concat = np.concatenate(all_predictions_VAE, axis=0)
                all_true_inits_VAE_concat = np.concatenate([np.full((Npath, x_dim), x0_grid[jj]) for jj in range(N_x0)], axis=0)
                all_bx_pred_VAE_concat = np.concatenate([np.full((Npath,), bx) for bx in all_bx_pred_VAE], axis=0)
                residual_VAE = all_predictions_VAE_concat - all_true_inits_VAE_concat - all_bx_pred_VAE_concat[:, np.newaxis] * sde_dt
                sigmax_pred_VAE[t_idx] = np.mean(np.std(residual_VAE, axis=0)) * np.sqrt(1 / sde_dt)
        
        # Compute FEX-NN diffusion for current time step
        if FEX_NN_time_dependent and models_dict_FEX_NN and t in models_dict_FEX_NN:
            # Time-dependent FEX-NN: use model for current time step
            FEX_NN_t = models_dict_FEX_NN[t]
            all_predictions_NN = []
            all_bx_pred_NN = []
            for jj in range(N_x0):
                true_init = x0_grid[jj]
                x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
                z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                with torch.no_grad():
                    cov_pred = FEX_NN_t(x_pred_new)
                    if x_dim == 1:
                        std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                        prediction_NN = (x_pred_new.squeeze(-1) + FEX_deterministic(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                        prediction_NN = prediction_NN[:, np.newaxis]
                    else:
                        cov_matrix = cov_pred.reshape(Npath, x_dim, x_dim)
                        cov_matrix = cov_matrix + 1e-6 * torch.eye(x_dim, device=device).unsqueeze(0)
                        try:
                            L = torch.linalg.cholesky(cov_matrix)
                            noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                        except:
                            diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)
                            noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * z
                        prediction_NN = (x_pred_new + FEX_deterministic(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                all_predictions_NN.append(prediction_NN)
                all_bx_pred_NN.append(bx_pred_NN[t_idx, jj])
            
            all_predictions_NN_concat = np.concatenate(all_predictions_NN, axis=0)
            all_true_inits_NN_concat = np.concatenate([np.full((Npath, x_dim), x0_grid[jj]) for jj in range(N_x0)], axis=0)
            all_bx_pred_NN_concat = np.concatenate([np.full((Npath,), bx) for bx in all_bx_pred_NN], axis=0)
            residual_NN = all_predictions_NN_concat - all_true_inits_NN_concat - all_bx_pred_NN_concat[:, np.newaxis] * sde_dt
            sigmax_pred_NN[t_idx] = np.mean(np.std(residual_NN, axis=0)) * np.sqrt(1 / sde_dt)
        elif FEX_NN is not None:
            # Time-independent FEX-NN: use same model for all time steps
            all_predictions_NN = []
            all_bx_pred_NN = []
            for jj in range(N_x0):
                true_init = x0_grid[jj]
                x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
                z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                with torch.no_grad():
                    cov_pred = FEX_NN(x_pred_new)
                    if x_dim == 1:
                        std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                        prediction_NN = (x_pred_new.squeeze(-1) + FEX_deterministic(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                        prediction_NN = prediction_NN[:, np.newaxis]
                    else:
                        cov_matrix = cov_pred.reshape(Npath, x_dim, x_dim)
                        cov_matrix = cov_matrix + 1e-6 * torch.eye(x_dim, device=device).unsqueeze(0)
                        try:
                            L = torch.linalg.cholesky(cov_matrix)
                            noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                        except:
                            diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)
                            noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * z
                        prediction_NN = (x_pred_new + FEX_deterministic(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                all_predictions_NN.append(prediction_NN)
                all_bx_pred_NN.append(bx_pred_NN[t_idx, jj])
            
            all_predictions_NN_concat = np.concatenate(all_predictions_NN, axis=0)
            all_true_inits_NN_concat = np.concatenate([np.full((Npath, x_dim), x0_grid[jj]) for jj in range(N_x0)], axis=0)
            all_bx_pred_NN_concat = np.concatenate([np.full((Npath,), bx) for bx in all_bx_pred_NN], axis=0)
            residual_NN = all_predictions_NN_concat - all_true_inits_NN_concat - all_bx_pred_NN_concat[:, np.newaxis] * sde_dt
            sigmax_pred_NN[t_idx] = np.mean(np.std(residual_NN, axis=0)) * np.sqrt(1 / sde_dt)
        
        # Compute TF-CDM and VAE diffusion (time-independent, compute only once at first time step)
        if t_idx == 0:
            print(f"  sigmax_true={sigmax_true_all[t_idx]:.4f}, sigmax_pred={sigmax_pred_all[t_idx]:.4f}")
            
            if FN_TF_CDM is not None and not TF_CDM_time_dependent:
                # Compute TF-CDM diffusion (time-independent, need to recompute predictions for all x0)
                all_predictions_TF_CDM = []
                all_bx_pred_TF_CDM = []
                for jj in range(N_x0):
                    true_init = x0_grid[jj]
                    x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
                    z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                    with torch.no_grad():
                        prediction_TF_CDM = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                        prediction_TF_CDM = (prediction_TF_CDM / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
                    all_predictions_TF_CDM.append(prediction_TF_CDM)
                    all_bx_pred_TF_CDM.append(bx_pred_TF_CDM[jj])
                
                all_predictions_TF_CDM_concat = np.concatenate(all_predictions_TF_CDM, axis=0)
                all_true_inits_TF_CDM_concat = np.concatenate([np.full((Npath, x_dim), x0_grid[jj]) for jj in range(N_x0)], axis=0)
                all_bx_pred_TF_CDM_concat = np.concatenate([np.full((Npath,), bx) for bx in all_bx_pred_TF_CDM], axis=0)
                residual_TF_CDM = all_predictions_TF_CDM_concat - all_true_inits_TF_CDM_concat - all_bx_pred_TF_CDM_concat[:, np.newaxis] * sde_dt
                sigmax_pred_TF_CDM = np.mean(np.std(residual_TF_CDM, axis=0)) * np.sqrt(1 / sde_dt)
            
            if VAE_FEX is not None and not VAE_time_dependent:
                # Time-independent VAE: compute diffusion (need to recompute predictions for all x0)
                all_predictions_VAE = []
                all_bx_pred_VAE = []
                for jj in range(N_x0):
                    true_init = x0_grid[jj]
                    x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
                    z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
                    with torch.no_grad():
                        prediction_VAE = VAE_FEX.decoder(z)
                        if isinstance(diff_scale_FEX, np.ndarray):
                            diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                        else:
                            diff_scale_VAE_value = float(diff_scale_FEX)
                        prediction_VAE = (prediction_VAE / diff_scale_VAE_value + x_pred_new + FEX_deterministic(x_pred_new) * sde_dt).to('cpu').detach().numpy()
                    all_predictions_VAE.append(prediction_VAE)
                    all_bx_pred_VAE.append(bx_pred_VAE[jj])
                
                all_predictions_VAE_concat = np.concatenate(all_predictions_VAE, axis=0)
                all_true_inits_VAE_concat = np.concatenate([np.full((Npath, x_dim), x0_grid[jj]) for jj in range(N_x0)], axis=0)
                all_bx_pred_VAE_concat = np.concatenate([np.full((Npath,), bx) for bx in all_bx_pred_VAE], axis=0)
                residual_VAE = all_predictions_VAE_concat - all_true_inits_VAE_concat - all_bx_pred_VAE_concat[:, np.newaxis] * sde_dt
                sigmax_pred_VAE = np.mean(np.std(residual_VAE, axis=0)) * np.sqrt(1 / sde_dt)
    
    # Find indices for t=50 and final time step (or closest available)
    t_50_idx = None
    t_final_idx = None
    final_time_step = max(time_steps_to_plot)  # Get the actual final time step
    
    for t_idx, t in enumerate(time_steps_to_plot):
        if abs(t - 50) < 1:
            t_50_idx = t_idx
        if t == final_time_step:
            t_final_idx = t_idx
    
    # Create plot: 2x3 subplots - top row: predictions, bottom row: errors
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    
    # Color & Style Setup (matching OU1d plot)
    colors = {'FEX-DM': 'orange', 'TF-CDM': 'steelblue', 'FEX-VAE': 'green', 'FEX-NN': 'purple', 'Ground-Truth': 'black'}
    linestyles = {'FEX-DM': '-', 'TF-CDM': '--', 'FEX-VAE': '-', 'FEX-NN': '-', 'Ground-Truth': ':'}
    markers = {'FEX-DM': 'o', 'TF-CDM': 's', 'FEX-VAE': '^', 'FEX-NN': 'D'}
    
    # ========== TOP ROW: PREDICTIONS ==========
    # Drift Plot at t=50
    if t_50_idx is not None:
        # Draw VAE (time-dependent or time-independent)
        if VAE_time_dependent and models_dict_VAE and bx_pred_VAE is not None:
            # Time-dependent VAE
            if t_50_idx is not None and t_50_idx < bx_pred_VAE.shape[0]:
                ax[0, 0].plot(x0_grid, bx_pred_VAE[t_50_idx, :], label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                           color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5, zorder=1)
        elif VAE_FEX is not None:
            # Time-independent VAE
            ax[0, 0].plot(x0_grid, bx_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                       color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5, zorder=1)
        
        # Draw FEX-DM, TF-CDM, and FEX-NN on top
        ax[0, 0].plot(x0_grid, bx_pred_all[t_50_idx, :], label='FEX-DM', linestyle=linestyles['FEX-DM'], 
                   color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5, zorder=3)
        
        # Draw FEX-NN
        if (FEX_NN_time_dependent and models_dict_FEX_NN) or (FEX_NN is not None):
            if bx_pred_NN is not None:
                ax[0, 0].plot(x0_grid, bx_pred_NN[t_50_idx, :], label='FEX-NN', linestyle=linestyles['FEX-NN'], 
                           color=colors['FEX-NN'], linewidth=3, marker=markers['FEX-NN'], markersize=5, zorder=2)
        
        if TF_CDM_time_dependent and models_dict_TF_CDM and bx_pred_TF_CDM is not None:
            # Time-dependent TF-CDM
            if t_50_idx is not None and t_50_idx < bx_pred_TF_CDM.shape[0]:
                training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
                x0_training = x0_grid[training_mask]
                bx_pred_TF_CDM_training = bx_pred_TF_CDM[t_50_idx, :][training_mask]
                ax[0, 0].plot(x0_training, bx_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                           color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2, zorder=3)
        elif FN_TF_CDM is not None:
            # Time-independent TF-CDM
            training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
            x0_training = x0_grid[training_mask]
            bx_pred_TF_CDM_training = bx_pred_TF_CDM[training_mask]
            ax[0, 0].plot(x0_training, bx_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                       color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2, zorder=3)
        
        ax[0, 0].plot(x0_grid, bx_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
                   color=colors['Ground-Truth'], linewidth=2, zorder=4)
        ax[0, 0].axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
        ax[0, 0].axvline(domain_start, color='gray', linestyle='--', linewidth=2)
        ax[0, 0].axvline(domain_end, color='gray', linestyle='--', linewidth=2)
        ax[0, 0].set_xlabel('$x$', fontsize=30)
        ax[0, 0].set_ylabel('$\\hat{\\mu}(x)$', fontsize=30)
        ax[0, 0].set_title(f'Drift at $t=50$', fontsize=24)
        ax[0, 0].tick_params(axis='both', labelsize=25)
        xticks = [x_min, domain_start, domain_end, x_max]
        ax[0, 0].set_xticks(xticks)
    else:
        ax[0, 0].text(0.5, 0.5, 't=50 not available', ha='center', va='center', fontsize=20)
        ax[0, 0].set_title('Drift at $t=50$', fontsize=24)
    
    # Drift Plot at final time step
    if t_final_idx is not None:
        final_time_value = final_time_step * sde_dt
        # Draw VAE (time-dependent or time-independent)
        if VAE_time_dependent and models_dict_VAE and bx_pred_VAE is not None:
            # Time-dependent VAE
            if t_final_idx is not None and t_final_idx < bx_pred_VAE.shape[0]:
                ax[0, 1].plot(x0_grid, bx_pred_VAE[t_final_idx, :], label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                           color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5, zorder=1)
        elif VAE_FEX is not None:
            # Time-independent VAE
            ax[0, 1].plot(x0_grid, bx_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                       color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5, zorder=1)
        
        # Draw FEX-DM, TF-CDM, and FEX-NN on top
        ax[0, 1].plot(x0_grid, bx_pred_all[t_final_idx, :], label='FEX-DM', linestyle=linestyles['FEX-DM'], 
                   color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5, zorder=3)
        
        # Draw FEX-NN
        if (FEX_NN_time_dependent and models_dict_FEX_NN) or (FEX_NN is not None):
            if bx_pred_NN is not None:
                ax[0, 1].plot(x0_grid, bx_pred_NN[t_final_idx, :], label='FEX-NN', linestyle=linestyles['FEX-NN'], 
                           color=colors['FEX-NN'], linewidth=3, marker=markers['FEX-NN'], markersize=5, zorder=2)
        
        if TF_CDM_time_dependent and models_dict_TF_CDM and bx_pred_TF_CDM is not None:
            # Time-dependent TF-CDM
            if t_final_idx is not None and t_final_idx < bx_pred_TF_CDM.shape[0]:
                training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
                x0_training = x0_grid[training_mask]
                bx_pred_TF_CDM_training = bx_pred_TF_CDM[t_final_idx, :][training_mask]
                ax[0, 1].plot(x0_training, bx_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                           color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2, zorder=3)
        elif FN_TF_CDM is not None:
            # Time-independent TF-CDM
            training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
            x0_training = x0_grid[training_mask]
            bx_pred_TF_CDM_training = bx_pred_TF_CDM[training_mask]
            ax[0, 1].plot(x0_training, bx_pred_TF_CDM_training, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                       color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2, zorder=3)
        
        ax[0, 1].plot(x0_grid, bx_true, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
                   color=colors['Ground-Truth'], linewidth=2, zorder=4)
        ax[0, 1].axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
        ax[0, 1].axvline(domain_start, color='gray', linestyle='--', linewidth=2)
        ax[0, 1].axvline(domain_end, color='gray', linestyle='--', linewidth=2)
        ax[0, 1].set_xlabel('$x$', fontsize=30)
        ax[0, 1].set_ylabel('$\\hat{\\mu}(x)$', fontsize=30)
        # Display t=100 instead of t=99
        display_time = 100 if final_time_step == 99 else final_time_step
        ax[0, 1].set_title(f'Drift at $t={display_time}$', fontsize=24)
        ax[0, 1].tick_params(axis='both', labelsize=25)
        xticks = [x_min, domain_start, domain_end, x_max]
        ax[0, 1].set_xticks(xticks)
    else:
        ax[0, 1].text(0.5, 0.5, 'Final time step not available', ha='center', va='center', fontsize=20)
        ax[0, 1].set_title('Drift at Final Time', fontsize=24)
    
    # Diffusion Plot: plot vs time (sigmax_pred_all is already a single value per time step)
    # Draw VAE (time-dependent or time-independent)
    if VAE_time_dependent and models_dict_VAE and len(sigmax_pred_VAE) == len(time_values):
        # Time-dependent VAE: plot curve vs time
        ax[0, 2].plot(time_values, sigmax_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                   color=colors['FEX-VAE'], linewidth=3, marker=markers['FEX-VAE'], markersize=5, zorder=1)
    elif VAE_FEX is not None:
        # Time-independent VAE: constant line
        ax[0, 2].axhline(y=sigmax_pred_VAE, label='FEX-VAE', linestyle=linestyles['FEX-VAE'], 
                       color=colors['FEX-VAE'], linewidth=3, zorder=1)
    
    # Draw FEX-DM (time-dependent)
    ax[0, 2].plot(time_values, sigmax_pred_all, label='FEX-DM', linestyle=linestyles['FEX-DM'], 
               color=colors['FEX-DM'], linewidth=3, marker=markers['FEX-DM'], markersize=5, zorder=3)
    
    # Draw TF-CDM (time-dependent or time-independent)
    if TF_CDM_time_dependent and models_dict_TF_CDM and len(sigmax_pred_TF_CDM) == len(time_values):
        # Time-dependent TF-CDM: plot curve vs time
        ax[0, 2].plot(time_values, sigmax_pred_TF_CDM, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                   color=colors['TF-CDM'], linewidth=3, marker=markers['TF-CDM'], markersize=2, zorder=3)
    elif FN_TF_CDM is not None:
        # Time-independent TF-CDM: constant line
        ax[0, 2].axhline(y=sigmax_pred_TF_CDM, label='TF-CDM', linestyle=linestyles['TF-CDM'], 
                       color=colors['TF-CDM'], linewidth=3, zorder=3)
    
    # Draw FEX-NN (time-dependent or time-independent)
    if ((FEX_NN_time_dependent and models_dict_FEX_NN) or (FEX_NN is not None)) and sigmax_pred_NN is not None and len(sigmax_pred_NN) == len(time_values):
        ax[0, 2].plot(time_values, sigmax_pred_NN, label='FEX-NN', linestyle=linestyles['FEX-NN'], 
                   color=colors['FEX-NN'], linewidth=3, marker=markers['FEX-NN'], markersize=5, zorder=2)
    
    ax[0, 2].plot(time_values, sigmax_true_all, label='Ground-Truth', linestyle=linestyles['Ground-Truth'], 
               color=colors['Ground-Truth'], linewidth=2, zorder=4)
    ax[0, 2].set_xlabel('$t$', fontsize=30)
    ax[0, 2].set_ylabel('$\\hat{\\sigma}(t)$', fontsize=30)
    ax[0, 2].set_title('Diffusion vs Time', fontsize=24)
    ax[0, 2].tick_params(axis='both', labelsize=25)
    ax[0, 2].grid(False)
    
    # ========== BOTTOM ROW: ERRORS ==========
    # Error: Drift at t=50 (absolute error: |prediction - ground truth|)
    if t_50_idx is not None:
        # VAE error (plot first, bottom layer)
        if VAE_time_dependent and models_dict_VAE and bx_pred_VAE is not None:
            if t_50_idx is not None and t_50_idx < bx_pred_VAE.shape[0]:
                error_drift_t50_VAE = np.abs(bx_pred_VAE[t_50_idx, :] - bx_true)
                ax[1, 0].plot(x0_grid, error_drift_t50_VAE, color=colors['FEX-VAE'], linewidth=2, marker=markers['FEX-VAE'], markersize=4, label='FEX-VAE', zorder=1)
        elif VAE_FEX is not None:
            error_drift_t50_VAE = np.abs(bx_pred_VAE - bx_true)
            ax[1, 0].plot(x0_grid, error_drift_t50_VAE, color=colors['FEX-VAE'], linewidth=2, marker=markers['FEX-VAE'], markersize=4, label='FEX-VAE', zorder=1)
        
        # FEX-NN error (plot second, bottom layer)
        if ((FEX_NN_time_dependent and models_dict_FEX_NN) or (FEX_NN is not None)) and bx_pred_NN is not None:
            error_drift_t50_NN = np.abs(bx_pred_NN[t_50_idx, :] - bx_true)
            ax[1, 0].plot(x0_grid, error_drift_t50_NN, color=colors['FEX-NN'], linewidth=2, marker=markers['FEX-NN'], markersize=4, label='FEX-NN', zorder=2)
        
        # FEX-DM error (plot third, top layer)
        error_drift_t50_FEX = np.abs(bx_pred_all[t_50_idx, :] - bx_true)
        ax[1, 0].plot(x0_grid, error_drift_t50_FEX, color=colors['FEX-DM'], linewidth=2, marker=markers['FEX-DM'], markersize=4, label='FEX-DM', zorder=3)
        
        # TF-CDM error (plot last, top layer)
        if TF_CDM_time_dependent and models_dict_TF_CDM and bx_pred_TF_CDM is not None:
            # Time-dependent TF-CDM
            if t_50_idx is not None and t_50_idx < bx_pred_TF_CDM.shape[0]:
                training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
                x0_training = x0_grid[training_mask]
                error_drift_t50_TF_CDM = np.abs(bx_pred_TF_CDM[t_50_idx, :][training_mask] - bx_true[training_mask])
                ax[1, 0].plot(x0_training, error_drift_t50_TF_CDM, color=colors['TF-CDM'], linewidth=2, marker=markers['TF-CDM'], markersize=3, label='TF-CDM', zorder=4)
        elif FN_TF_CDM is not None:
            # Time-independent TF-CDM
            training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
            x0_training = x0_grid[training_mask]
            error_drift_t50_TF_CDM = np.abs(bx_pred_TF_CDM[training_mask] - bx_true[training_mask])
            ax[1, 0].plot(x0_training, error_drift_t50_TF_CDM, color=colors['TF-CDM'], linewidth=2, marker=markers['TF-CDM'], markersize=3, label='TF-CDM', zorder=4)
        
        ax[1, 0].axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
        ax[1, 0].axvline(domain_start, color='gray', linestyle='--', linewidth=2)
        ax[1, 0].axvline(domain_end, color='gray', linestyle='--', linewidth=2)
        ax[1, 0].set_xlabel('$x$', fontsize=30)
        ax[1, 0].set_ylabel('$|\\hat{\\mu}(x) - \\mu(x)|$', fontsize=30)
        ax[1, 0].set_title(f'Drift Error at $t=50$', fontsize=24)
        ax[1, 0].tick_params(axis='both', labelsize=25)
        xticks = [x_min, domain_start, domain_end, x_max]
        ax[1, 0].set_xticks(xticks)
        ax[1, 0].set_yscale('log')
        ax[1, 0].grid(True, alpha=0.3)
    else:
        ax[1, 0].text(0.5, 0.5, 't=50 not available', ha='center', va='center', fontsize=20)
        ax[1, 0].set_title('Drift Error at $t=50$', fontsize=24)
    
    # Error: Drift at final time step (absolute error: |prediction - ground truth|)
    if t_final_idx is not None:
        # VAE error (plot first, bottom layer)
        if VAE_time_dependent and models_dict_VAE and bx_pred_VAE is not None:
            if t_final_idx is not None and t_final_idx < bx_pred_VAE.shape[0]:
                error_drift_final_VAE = np.abs(bx_pred_VAE[t_final_idx, :] - bx_true)
                ax[1, 1].plot(x0_grid, error_drift_final_VAE, color=colors['FEX-VAE'], linewidth=2, marker=markers['FEX-VAE'], markersize=4, label='FEX-VAE', zorder=1)
        elif VAE_FEX is not None:
            error_drift_final_VAE = np.abs(bx_pred_VAE - bx_true)
            ax[1, 1].plot(x0_grid, error_drift_final_VAE, color=colors['FEX-VAE'], linewidth=2, marker=markers['FEX-VAE'], markersize=4, label='FEX-VAE', zorder=1)
        
        # FEX-NN error (plot second, bottom layer)
        if ((FEX_NN_time_dependent and models_dict_FEX_NN) or (FEX_NN is not None)) and bx_pred_NN is not None:
            error_drift_final_NN = np.abs(bx_pred_NN[t_final_idx, :] - bx_true)
            ax[1, 1].plot(x0_grid, error_drift_final_NN, color=colors['FEX-NN'], linewidth=2, marker=markers['FEX-NN'], markersize=4, label='FEX-NN', zorder=2)
        
        # FEX-DM error (plot third, top layer)
        error_drift_final_FEX = np.abs(bx_pred_all[t_final_idx, :] - bx_true)
        ax[1, 1].plot(x0_grid, error_drift_final_FEX, color=colors['FEX-DM'], linewidth=2, marker=markers['FEX-DM'], markersize=4, label='FEX-DM', zorder=3)
        
        # TF-CDM error (plot last, top layer)
        if TF_CDM_time_dependent and models_dict_TF_CDM and bx_pred_TF_CDM is not None:
            # Time-dependent TF-CDM
            if t_final_idx is not None and t_final_idx < bx_pred_TF_CDM.shape[0]:
                training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
                x0_training = x0_grid[training_mask]
                error_drift_final_TF_CDM = np.abs(bx_pred_TF_CDM[t_final_idx, :][training_mask] - bx_true[training_mask])
                ax[1, 1].plot(x0_training, error_drift_final_TF_CDM, color=colors['TF-CDM'], linewidth=2, marker=markers['TF-CDM'], markersize=3, label='TF-CDM', zorder=4)
        elif FN_TF_CDM is not None:
            # Time-independent TF-CDM
            training_mask = (x0_grid >= domain_start) & (x0_grid <= domain_end)
            x0_training = x0_grid[training_mask]
            error_drift_final_TF_CDM = np.abs(bx_pred_TF_CDM[training_mask] - bx_true[training_mask])
            ax[1, 1].plot(x0_training, error_drift_final_TF_CDM, color=colors['TF-CDM'], linewidth=2, marker=markers['TF-CDM'], markersize=3, label='TF-CDM', zorder=4)
        
        ax[1, 1].axvspan(domain_start, domain_end, color='gray', alpha=0.2, label="Training Domain")
        ax[1, 1].axvline(domain_start, color='gray', linestyle='--', linewidth=2)
        ax[1, 1].axvline(domain_end, color='gray', linestyle='--', linewidth=2)
        ax[1, 1].set_xlabel('$x$', fontsize=30)
        ax[1, 1].set_ylabel('$|\\hat{\\mu}(x) - \\mu(x)|$', fontsize=30)
        # Display t=100 instead of t=99
        display_time = 100 if final_time_step == 99 else final_time_step
        ax[1, 1].set_title(f'Drift Error at $t={display_time}$', fontsize=24)
        ax[1, 1].tick_params(axis='both', labelsize=25)
        xticks = [x_min, domain_start, domain_end, x_max]
        ax[1, 1].set_xticks(xticks)
        ax[1, 1].set_yscale('log')
        ax[1, 1].grid(True, alpha=0.3)
    else:
        ax[1, 1].text(0.5, 0.5, 'Final time step not available', ha='center', va='center', fontsize=20)
        ax[1, 1].set_title('Drift Error at Final Time', fontsize=24)
    
    # Error: Diffusion vs time (absolute error: |prediction - ground truth|)
    # Plot VAE and FEX-NN first (bottom layer)
    if VAE_time_dependent and models_dict_VAE and len(sigmax_pred_VAE) == len(time_values):
        # Time-dependent VAE: plot error curve vs time
        error_diffusion_VAE = np.abs(sigmax_pred_VAE - sigmax_true_all)
        ax[1, 2].plot(time_values, error_diffusion_VAE, color=colors['FEX-VAE'], linewidth=2, marker=markers['FEX-VAE'], markersize=4, linestyle=linestyles['FEX-VAE'], label='FEX-VAE', zorder=1)
    elif VAE_FEX is not None:
        # Time-independent VAE: constant line
        error_diffusion_VAE = np.abs(sigmax_pred_VAE - sigmax_true_all)
        ax[1, 2].axhline(y=np.mean(error_diffusion_VAE), color=colors['FEX-VAE'], linewidth=2, linestyle=linestyles['FEX-VAE'], label='FEX-VAE', zorder=1)
    
    # FEX-NN diffusion error (bottom layer)
    if ((FEX_NN_time_dependent and models_dict_FEX_NN) or (FEX_NN is not None)) and sigmax_pred_NN is not None and len(sigmax_pred_NN) == len(time_values):
        error_diffusion_NN = np.abs(sigmax_pred_NN - sigmax_true_all)
        ax[1, 2].plot(time_values, error_diffusion_NN, color=colors['FEX-NN'], linewidth=2, marker=markers['FEX-NN'], markersize=4, linestyle=linestyles['FEX-NN'], label='FEX-NN', zorder=2)
    
    # Plot FEX-DM and TF-CDM on top (top layer)
    error_diffusion_FEX = np.abs(sigmax_pred_all - sigmax_true_all)
    ax[1, 2].plot(time_values, error_diffusion_FEX, color=colors['FEX-DM'], linewidth=2, marker=markers['FEX-DM'], markersize=4, label='FEX-DM', zorder=3)
    
    # TF-CDM error (top layer)
    if TF_CDM_time_dependent and models_dict_TF_CDM and len(sigmax_pred_TF_CDM) == len(time_values):
        # Time-dependent TF-CDM: plot error curve vs time
        error_diffusion_TF_CDM = np.abs(sigmax_pred_TF_CDM - sigmax_true_all)
        ax[1, 2].plot(time_values, error_diffusion_TF_CDM, color=colors['TF-CDM'], linewidth=2, marker=markers['TF-CDM'], markersize=3, linestyle=linestyles['TF-CDM'], label='TF-CDM', zorder=4)
    elif FN_TF_CDM is not None:
        # Time-independent TF-CDM: constant line (compute error at each time step and show mean)
        error_diffusion_TF_CDM = np.abs(sigmax_pred_TF_CDM - sigmax_true_all)
        ax[1, 2].axhline(y=np.mean(error_diffusion_TF_CDM), color=colors['TF-CDM'], linewidth=2, linestyle=linestyles['TF-CDM'], label='TF-CDM', zorder=4)
    
    ax[1, 2].set_xlabel('$t$', fontsize=30)
    ax[1, 2].set_ylabel('$|\\hat{\\sigma}(t) - \\sigma(t)|$', fontsize=30)
    ax[1, 2].set_title('Diffusion Error vs Time', fontsize=24)
    ax[1, 2].tick_params(axis='both', labelsize=25)
    ax[1, 2].set_yscale('log')
    ax[1, 2].grid(True, alpha=0.3)
    
    # Legend for top row plots
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # Build legend with all available models
    legend_handles = []
    legend_labels = []
    for handle, label in zip(handles, labels):
        if label in ['FEX-DM', 'TF-CDM', 'FEX-VAE', 'FEX-NN', 'Ground-Truth', 'Training Domain']:
            legend_handles.append(handle)
            legend_labels.append(label)
    
    # # Add models from other subplots if not already included
    # handles_1, labels_1 = ax[0, 1].get_legend_handles_labels()
    # for handle, label in zip(handles_1, labels_1):
    #     if label not in legend_labels and label in ['FEX-DM', 'TF-CDM', 'FEX-VAE', 'Ground-Truth', 'Training Domain']:
    #         legend_handles.append(handle)
    #         legend_labels.append(label)
    
    # handles_2, labels_2 = ax[0, 2].get_legend_handles_labels()
    # for handle, label in zip(handles_2, labels_2):
    #     if label not in legend_labels and label in ['FEX-DM', 'TF-CDM', 'FEX-VAE', 'Ground-Truth']:
    #         legend_handles.append(handle)
    #         legend_labels.append(label)
    
    fig.legend(legend_handles, legend_labels, loc='upper center', fontsize=22, frameon=True, 
               ncol=len(legend_labels), bbox_to_anchor=(0.5, 1.02))
    
    # Save and show - give more space at top for legend above plot
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(save_dir, 'drift_and_diffusion_time_dependent.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    else:
        print(f"[WARNING] File was not created at: {save_path}")
    
    return save_path


def plot_conditional_distribution_time_dependent(second_stage_dir_FEX,
                                                models_dict,
                                                scaler,
                                                model_name='Trigonometric1d',
                                                All_stage_dir_TF_CDM=None,
                                                All_stage_dir_FEX_VAE=None,
                                                All_stage_dir_FEX_NN=None,
                                                scaler_TF_CDM=None,
                                                noise_level=1.0,
                                                device='cpu',
                                                base_path=None,
                                                initial_values=None,
                                                Npath=500000,
                                                save_dir=None,
                                                figsize=(18, 12),
                                                dpi=300,
                                                seed=42):
    """
    Plot conditional distribution for time-dependent FEX-DM models at t=0.
    Creates 1x3 subplots (one for each initial value).
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        models_dict: Dictionary from load_time_dependent_models mapping time step to (model, norm_params)
        scaler: Scaling factor (numpy array or scalar)
        model_name: Model name (e.g., 'Trigonometric1d')
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        base_path: Base path for loading FEX deterministic model
        initial_values: List of initial values to plot (default: [-3, 0.6, 3] for Trigonometric1d)
        Npath: Number of paths for Monte Carlo simulation (default: 500000)
        save_dir: Directory to save the figure
        figsize: Figure size tuple (default: (18, 6))
        dpi: Resolution for saved figure (default: 300)
        seed: Random seed for reproducibility
    
    Returns:
        str: Path to the saved figure file
    """
    # Load SDE parameters
    if params_init is not None:
        model_params = params_init(case_name=model_name)
        sigma_base = model_params['sig']
        sde_params = {
            'sig': sigma_base * noise_level,
            'sde_dt': model_params['Dt']
        }
    else:
        raise ValueError("params_init is not available")
    
    sig = sde_params['sig']
    sde_dt = sde_params['sde_dt']
    
    if save_dir is None:
        parent_dir = os.path.dirname(second_stage_dir_FEX)
        save_dir = os.path.join(parent_dir, 'plot')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set default initial values
    if initial_values is None:
        if model_name == 'Trigonometric1d':
            initial_values = [-3, 0.6, 3]
        else:
            initial_values = [-3, 0.6, 3]  # Default fallback
    
    # Get model for t=0
    available_time_steps = sorted(models_dict.keys())
    if not available_time_steps:
        raise ValueError("No time-dependent models found in models_dict")
    
    # Find model at t=0 (or nearest)
    t_0 = 0
    if t_0 not in models_dict:
        nearest_t = min(available_time_steps, key=lambda x: abs(x - t_0))
        print(f'[WARNING] Model for time step {t_0} not found, using nearest time step {nearest_t}')
        t_0 = nearest_t
    
    FN, norm_params = models_dict[t_0]
    
    # Get normalization parameters
    if 'x_mean' in norm_params and 'x_std' in norm_params:
        xTrain_mean = torch.tensor(norm_params['x_mean'], dtype=torch.float32).to(device)
        xTrain_std = torch.tensor(norm_params['x_std'], dtype=torch.float32).to(device)
    else:
        # Default: input is just Wiener increments, so shape is (dim,)
        dimension = FN.input_dim
        xTrain_mean = torch.zeros(dimension, dtype=torch.float32).to(device)
        xTrain_std = torch.ones(dimension, dtype=torch.float32).to(device)
    
    yTrain_mean = torch.tensor(norm_params['mean'], dtype=torch.float32).to(device)
    yTrain_std = torch.tensor(norm_params['std'], dtype=torch.float32).to(device)
    
    # Get dimension from model
    dimension = FN.input_dim
    
    # Load TF-CDM models (check for time-dependent first, then fall back to time-independent)
    FN_TF_CDM = None
    models_dict_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = scaler_TF_CDM
    TF_CDM_time_dependent = False
    
    if All_stage_dir_TF_CDM is not None:
        print("[INFO] Loading TF-CDM models...")
        from utils.helper import load_time_dependent_TF_CDM_models
        models_dict_TF_CDM = load_time_dependent_TF_CDM_models(All_stage_dir_TF_CDM, dimension, device=device)
        
        if models_dict_TF_CDM:
            print("[INFO] Found time-dependent TF-CDM models")
            TF_CDM_time_dependent = True
            if scaler_TF_CDM is None:
                data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
                if os.path.exists(data_inf_path_TF_CDM):
                    data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
                    diff_scale_TF_CDM = data_inf_TF_CDM.get('diff_scale', None)
        else:
            print("[INFO] No time-dependent TF-CDM models found, trying time-independent model...")
            data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
            if os.path.exists(data_inf_path_TF_CDM):
                data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
                xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
                xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
                yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
                yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
                if scaler_TF_CDM is None:
                    diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
                
                FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
                if os.path.exists(FNET_path_TF_CDM):
                    FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                    FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                    FN_TF_CDM.eval()
    
    # Load FEX-VAE models (check for time-dependent first, then fall back to time-independent)
    VAE_FEX = None
    models_dict_VAE = None
    diff_scale_FEX = scaler
    VAE_time_dependent = False
    
    if All_stage_dir_FEX_VAE is not None:
        print("[INFO] Loading FEX-VAE models...")
        from utils.helper import load_time_dependent_VAE_models
        models_dict_VAE = load_time_dependent_VAE_models(All_stage_dir_FEX_VAE, dimension, device=device)
        
        if models_dict_VAE:
            print("[INFO] Found time-dependent FEX-VAE models")
            VAE_time_dependent = True
        else:
            print("[INFO] No time-dependent FEX-VAE models found, trying time-independent model...")
            VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
            if os.path.exists(VAE_path):
                from utils.helper import VAE
                VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
                VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
                VAE_FEX.eval()
    
    # Load FEX-NN models (check for time-dependent first, then fall back to time-independent)
    FEX_NN = None
    models_dict_FEX_NN = None
    FEX_NN_time_dependent = False
    
    if All_stage_dir_FEX_NN is not None:
        print("[INFO] Loading FEX-NN models...")
        from utils.helper import load_time_dependent_FEX_NN_models
        models_dict_FEX_NN = load_time_dependent_FEX_NN_models(All_stage_dir_FEX_NN, dimension, device=device)
        
        if models_dict_FEX_NN:
            print("[INFO] Found time-dependent FEX-NN models")
            FEX_NN_time_dependent = True
        else:
            print("[INFO] No time-dependent FEX-NN models found, trying time-independent model...")
            FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
            if os.path.exists(FEX_NN_path):
                from utils.ODEParser import CovarianceNet
                output_dim_nn = dimension * dimension if dimension > 1 else 1
                FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
                FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
                FEX_NN.eval()
    
    # Create FEX deterministic model wrapper
    def FEX_deterministic(x):
        return FEX_model_learned(x, 
                                 model_name=model_name,
                                 noise_level=noise_level,
                                 device=device,
                                 base_path=base_path)
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Define model colors
    model_colors = {
        "FEX-DM": "orange",
        "TF-CDM": "steelblue",
        "FEX-VAE": "green",
        "FEX-NN": "purple",
        "Ground-Truth": "black"
    }
    
    x_dim = dimension
    
    def plot_conditional_distribution_single(true_init, ax_pdf, ax_err):
        """
        Plot conditional distribution for a given initial value at t=0.
        ax_pdf: axis for PDF plot (top row)
        ax_err: axis for error plot (bottom row)
        """
        x_pred_new = torch.clone((true_init * torch.ones(Npath, x_dim)).to(device))
        
        # Ground truth samples at t=0
        # For Trigonometric1d: dX_t = sin(2*k*pi*X_t)dt + sig*cos(2*k*pi*t)*dW_t
        # At t=0: dX_0 = sin(2*k*pi*X_0)dt + sig*cos(0)*dW_0 = sin(2*k*pi*X_0)dt + sig*dW_0
        if model_name == 'Trigonometric1d':
            k = 1  # frequency parameter
            # Generate ground truth samples
            ode_path_true = true_init * np.ones((Npath, x_dim))
            drift = np.sin(2 * k * np.pi * ode_path_true)
            # At t=0, cos(2*k*pi*0) = cos(0) = 1
            diffusion = sig * np.ones((Npath, x_dim))
            true_samples = ode_path_true + drift * sde_dt + diffusion * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
        else:
            # Default: simple diffusion
            ode_path_true = true_init * np.ones((Npath, x_dim))
            true_samples = ode_path_true + sig * np.random.normal(0, np.sqrt(sde_dt), size=(Npath, x_dim))
        
        # Define Plotting Range
        x_min, x_max = np.min(true_samples) - 0.05, np.max(true_samples) + 0.05
        x_vals = np.linspace(x_min, x_max, 200)
        
        # Compute KDE for True Distribution
        kde = gaussian_kde(true_samples.T)
        pdf_vals = kde(x_vals)
        
        # Top row: PDF plot
        ax_pdf.plot(x_vals, pdf_vals, color='black', linewidth=2, linestyle='dashed', label="Ground Truth", zorder=5)
        
        # Generate random noise z (use same z for all models for fair comparison)
        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
        
        # Store predictions for error computation
        predictions_dict = {}
        
        # Skip TF-CDM and FEX-NN for initial values -3 and 3
        skip_TF_CDM_cond = (abs(true_init - (-3)) < 0.01 or abs(true_init - 3) < 0.01)
        skip_FEX_NN_cond = (abs(true_init - (-3)) < 0.01 or abs(true_init - 3) < 0.01)
        
        # TF-CDM Prediction (plot first, bottom layer)
        if not skip_TF_CDM_cond and TF_CDM_time_dependent and models_dict_TF_CDM and t_0 in models_dict_TF_CDM:
            FN_TF_CDM_t, norm_params_TF_CDM_t = models_dict_TF_CDM[t_0]
            with torch.no_grad():
                x_mean_t = torch.tensor(norm_params_TF_CDM_t['x_mean'], dtype=torch.float32).to(device)
                x_std_t = torch.tensor(norm_params_TF_CDM_t['x_std'], dtype=torch.float32).to(device)
                y_mean_t = torch.tensor(norm_params_TF_CDM_t['mean'], dtype=torch.float32).to(device)
                y_std_t = torch.tensor(norm_params_TF_CDM_t['std'], dtype=torch.float32).to(device)
                xz_concat = torch.hstack((x_pred_new, z))
                xz_normalized = (xz_concat - x_mean_t) / x_std_t
                prediction_TF_CDM_normalized = FN_TF_CDM_t(xz_normalized)
                prediction_TF_CDM = prediction_TF_CDM_normalized * y_std_t + y_mean_t
                if diff_scale_TF_CDM is not None:
                    if isinstance(diff_scale_TF_CDM, np.ndarray):
                        diff_scale_value = float(diff_scale_TF_CDM[0]) if len(diff_scale_TF_CDM) > 0 else 1.0
                    else:
                        diff_scale_value = float(diff_scale_TF_CDM)
                    prediction_TF_CDM = (prediction_TF_CDM / diff_scale_value + x_pred_new).to('cpu').detach().numpy()
                else:
                    prediction_TF_CDM = (prediction_TF_CDM + x_pred_new).to('cpu').detach().numpy()
            predictions_dict['TF-CDM'] = prediction_TF_CDM
            ax_pdf.hist(prediction_TF_CDM, bins=50, density=True, alpha=0.3, color=model_colors["TF-CDM"], 
                    histtype='stepfilled', edgecolor=model_colors["TF-CDM"], label="TF-CDM", zorder=1)
        elif not skip_TF_CDM_cond and FN_TF_CDM is not None:
            with torch.no_grad():
                prediction_TF_CDM = FN_TF_CDM((torch.hstack((x_pred_new, z)) - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                prediction_TF_CDM = (prediction_TF_CDM / diff_scale_TF_CDM + x_pred_new).to('cpu').detach().numpy()
            predictions_dict['TF-CDM'] = prediction_TF_CDM
            ax_pdf.hist(prediction_TF_CDM, bins=50, density=True, alpha=0.3, color=model_colors["TF-CDM"], 
                    histtype='stepfilled', edgecolor=model_colors["TF-CDM"], label="TF-CDM", zorder=1)
        
        # FEX-VAE Prediction (plot second, bottom layer)
        if VAE_time_dependent and models_dict_VAE and t_0 in models_dict_VAE:
            VAE_model_t = models_dict_VAE[t_0]
            with torch.no_grad():
                prediction_VAE = VAE_model_t.decoder(z)
                if isinstance(diff_scale_FEX, np.ndarray):
                    diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                else:
                    diff_scale_VAE_value = float(diff_scale_FEX)
                FEX_det = FEX_deterministic(x_pred_new)
                prediction_VAE = (prediction_VAE / diff_scale_VAE_value + x_pred_new + FEX_det * sde_dt).to('cpu').detach().numpy()
            predictions_dict['FEX-VAE'] = prediction_VAE
            ax_pdf.hist(prediction_VAE, bins=50, density=True, alpha=0.5, color=model_colors["FEX-VAE"], 
                    histtype='stepfilled', edgecolor=model_colors["FEX-VAE"], label="FEX-VAE", zorder=2)
        elif VAE_FEX is not None:
            with torch.no_grad():
                prediction_VAE = VAE_FEX.decoder(z)
                if isinstance(diff_scale_FEX, np.ndarray):
                    diff_scale_VAE_value = float(diff_scale_FEX[0]) if len(diff_scale_FEX) > 0 else 1.0
                else:
                    diff_scale_VAE_value = float(diff_scale_FEX)
                FEX_det = FEX_deterministic(x_pred_new)
                prediction_VAE = (prediction_VAE / diff_scale_VAE_value + x_pred_new + FEX_det * sde_dt).to('cpu').detach().numpy()
            predictions_dict['FEX-VAE'] = prediction_VAE
            ax_pdf.hist(prediction_VAE, bins=50, density=True, alpha=0.5, color=model_colors["FEX-VAE"], 
                    histtype='stepfilled', edgecolor=model_colors["FEX-VAE"], label="FEX-VAE", zorder=2)
        
        # FEX-NN Prediction (plot third, bottom layer)
        # Skip FEX-NN for initial values -3 and 3
        if not skip_FEX_NN_cond and FEX_NN_time_dependent and models_dict_FEX_NN and t_0 in models_dict_FEX_NN:
            FEX_NN_t = models_dict_FEX_NN[t_0]
            with torch.no_grad():
                cov_pred = FEX_NN_t(x_pred_new)
                if x_dim == 1:
                    std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                    prediction_NN = (x_pred_new.squeeze(-1) + FEX_deterministic(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                    prediction_NN = prediction_NN[:, np.newaxis]
                else:
                    cov_matrix = cov_pred.reshape(Npath, x_dim, x_dim)
                    cov_matrix = cov_matrix + 1e-6 * torch.eye(x_dim, device=device).unsqueeze(0)
                    try:
                        L = torch.linalg.cholesky(cov_matrix)
                        noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                    except:
                        diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)
                        noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * z
                    prediction_NN = (x_pred_new + FEX_deterministic(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
            predictions_dict['FEX-NN'] = prediction_NN
            ax_pdf.hist(prediction_NN, bins=50, density=True, alpha=0.3, color=model_colors["FEX-NN"], 
                    histtype='stepfilled', edgecolor=model_colors["FEX-NN"], label="FEX-NN", zorder=3)
        elif not skip_FEX_NN_cond and FEX_NN is not None:
            with torch.no_grad():
                cov_pred = FEX_NN(x_pred_new)
                if x_dim == 1:
                    std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                    prediction_NN = (x_pred_new.squeeze(-1) + FEX_deterministic(x_pred_new).squeeze(-1) * sde_dt + std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to('cpu').detach().numpy()
                    prediction_NN = prediction_NN[:, np.newaxis]
                else:
                    cov_matrix = cov_pred.reshape(Npath, x_dim, x_dim)
                    cov_matrix = cov_matrix + 1e-6 * torch.eye(x_dim, device=device).unsqueeze(0)
                    try:
                        L = torch.linalg.cholesky(cov_matrix)
                        noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                    except:
                        diag_var = torch.diagonal(cov_matrix, dim1=1, dim2=2)
                        noise = torch.sqrt(torch.clamp(diag_var, min=1e-8)) * z
                    prediction_NN = (x_pred_new + FEX_deterministic(x_pred_new) * sde_dt + noise * np.sqrt(sde_dt)).to('cpu').detach().numpy()
            predictions_dict['FEX-NN'] = prediction_NN
            ax_pdf.hist(prediction_NN, bins=50, density=True, alpha=0.3, color=model_colors["FEX-NN"], 
                    histtype='stepfilled', edgecolor=model_colors["FEX-NN"], label="FEX-NN", zorder=3)
        
        # FEX-DM Prediction (plot last, top layer)
        with torch.no_grad():
            # Model takes only Wiener increments as input
            prediction_FEX = FN((z - xTrain_mean) / xTrain_std) * yTrain_std + yTrain_mean
            
            # Add FEX deterministic update and apply diff_scale
            FEX_det = FEX_deterministic(x_pred_new)
            # Convert scaler to scalar value for division
            if isinstance(scaler, np.ndarray):
                scaler_value = float(scaler[0]) if len(scaler) > 0 else 1.0
            else:
                scaler_value = float(scaler)
            
            prediction_FEX_DM = (prediction_FEX / scaler_value + x_pred_new + FEX_det * sde_dt).to('cpu').detach().numpy()
        
        predictions_dict['FEX-DM'] = prediction_FEX_DM
        # Plot Histogram of Learned Distribution (on top)
        ax_pdf.hist(prediction_FEX_DM, bins=50, density=True, alpha=0.5, color=model_colors["FEX-DM"], 
                histtype='stepfilled', edgecolor=model_colors["FEX-DM"], label="FEX-DM", zorder=4)
        
        # PDF plot settings
        ax_pdf.set_xlabel('$x$', fontsize=22)
        ax_pdf.set_ylabel('pdf', fontsize=22)
        ax_pdf.set_title(f'$x_0$ = {true_init:.2f}', fontsize=24)
        ax_pdf.set_xlim([x_min, x_max])
        ax_pdf.tick_params(axis='both', labelsize=22)
        ax_pdf.grid(False)
        
        # Bottom row: Error plot (prediction PDF - ground truth PDF)
        # Plot errors for all models, with FEX-DM on top
        ax_err.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        
        for model_name_plot in ['TF-CDM', 'FEX-VAE', 'FEX-NN', 'FEX-DM']:
            # Skip TF-CDM and FEX-NN for initial values -3 and 3
            if model_name_plot == 'TF-CDM' and skip_TF_CDM_cond:
                continue
            if model_name_plot == 'FEX-NN' and skip_FEX_NN_cond:
                continue
            if model_name_plot in predictions_dict:
                kde_pred = gaussian_kde(predictions_dict[model_name_plot].T)
                pdf_pred = kde_pred(x_vals)
                error = pdf_pred - pdf_vals
                zorder_val = 2 if model_name_plot != 'FEX-DM' else 3
                ax_err.plot(x_vals, error, color=model_colors[model_name_plot], linewidth=2, label=model_name_plot, zorder=zorder_val)
        
        ax_err.set_xlabel('$x$', fontsize=22)
        ax_err.set_ylabel('Error (pdf)', fontsize=22)
        ax_err.set_title(f'Error: $x_0$ = {true_init:.2f}', fontsize=24)
        ax_err.set_xlim([x_min, x_max])
        ax_err.tick_params(axis='both', labelsize=22)
        ax_err.grid(False)
    
    # Create 2×3 Subplot Grid (Two rows, three subplots)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    for col, x0 in enumerate(initial_values):
        plot_conditional_distribution_single(x0, axes[0, col], axes[1, col])
    
    # Manually Add Legend (include TF-CDM if available, it will be drawn for some initial values like 0.6)
    legend_handles = [
        plt.Line2D([0], [0], color="black", linestyle="dashed", linewidth=2, label="Ground Truth")
    ]
    # Include TF-CDM in legend if it's available (it will be drawn for some initial values like 0.6)
    if 'TF-CDM' in model_colors:
        legend_handles.append(plt.Line2D([0], [0], color=model_colors["TF-CDM"], linewidth=6, label="TF-CDM"))
    if 'FEX-VAE' in model_colors:
        legend_handles.append(plt.Line2D([0], [0], color=model_colors["FEX-VAE"], linewidth=6, label="FEX-VAE"))
    if 'FEX-NN' in model_colors:
        legend_handles.append(plt.Line2D([0], [0], color=model_colors["FEX-NN"], linewidth=6, label="FEX-NN"))
    legend_handles.append(plt.Line2D([0], [0], color=model_colors["FEX-DM"], linewidth=6, label="FEX-DM"))
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=len(legend_handles), fontsize=16, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'conditional_distribution_time_dependent_t0.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"[INFO] File verified: {save_path} ({file_size} bytes)")
    else:
        print(f"[WARNING] File was not created at: {save_path}")
    
    return save_path


def plot_conditional_distribution_doublewell_timeseries(second_stage_dir_FEX,
                                                       All_stage_dir_TF_CDM=None,
                                                       All_stage_dir_FEX_VAE=None,
                                                       All_stage_dir_FEX_NN=None,
                                                       model_name='DoubleWell1d',
                                                       noise_level=1.0,
                                                       device='cpu',
                                                       initial_value=1.5,
                                                       times_to_plot=[5, 30, 100],
                                                       save_dir=None,
                                                       figsize=(16, 5),
                                                       dpi=300,
                                                       seed=42):
    """
    Plot conditional distributions for DoubleWell1d at specific time points in a single row.
    
    Creates a 1xN layout showing conditional distributions at different time points (e.g., T=5, 30, 100)
    for a given initial value, comparing FEX-DM, TF-CDM, FEX-NN, FEX-VAE, and Ground Truth.
    
    Args:
        second_stage_dir_FEX: Directory path for FEX-DM second stage results
        All_stage_dir_TF_CDM: Optional directory path for TF-CDM second stage results
        All_stage_dir_FEX_VAE: Optional directory path for FEX-VAE second stage results
        All_stage_dir_FEX_NN: Optional directory path for FEX-NN second stage results
        model_name: Model name (should be 'DoubleWell1d')
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        initial_value: Initial value x₀ (default: 1.5)
        times_to_plot: List of time points to plot (default: [5, 30, 100])
        save_dir: Directory to save the figure
        figsize: Figure size tuple (default: (16, 5))
        dpi: Resolution for saved figure (default: 300)
        seed: Random seed for reproducibility
    
    Returns:
        str: Path to the saved figure file
    """
    if model_name != 'DoubleWell1d':
        raise ValueError(f"This function is only for DoubleWell1d, got {model_name}")
    
    # Load SDE parameters
    model_params = params_init(case_name=model_name)
    sigma_base = model_params['sig']
    sigma = sigma_base * noise_level
    sde_T = model_params['T']
    sde_dt = model_params['Dt']
    
    if save_dir is None:
        parent_dir = os.path.dirname(second_stage_dir_FEX)
        save_dir = os.path.join(parent_dir, 'plot')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Fixed color map
    model_colors = {
        "FEX-DM": "orange",
        "TF-CDM": "steelblue",
        "FEX-VAE": "green",
        "FEX-NN": "purple"
    }
    
    # Load models
    print("[INFO] Loading models for conditional distribution plotting...")
    data_inf_path_FEX = os.path.join(second_stage_dir_FEX, 'data_inf.pt')
    if not os.path.exists(data_inf_path_FEX):
        raise FileNotFoundError(f"FEX-DM data_inf.pt not found at {data_inf_path_FEX}")
    
    data_inf_FEX = torch.load(data_inf_path_FEX, map_location=device)
    xTrain_mean_FEX = data_inf_FEX['ZT_Train_mean'].to(device)
    xTrain_std_FEX = data_inf_FEX['ZT_Train_std'].to(device)
    yTrain_mean_FEX = data_inf_FEX['ODE_Train_mean'].to(device)
    yTrain_std_FEX = data_inf_FEX['ODE_Train_std'].to(device)
    diff_scale_FEX = data_inf_FEX['diff_scale']
    
    dimension = data_inf_FEX['ZT_Train_new'].shape[1]
    FNET_path_FEX = os.path.join(second_stage_dir_FEX, 'FNET.pth')
    if not os.path.exists(FNET_path_FEX):
        raise FileNotFoundError(f"FEX-DM FNET.pth not found at {FNET_path_FEX}")
    
    FN_FEX = FN_Net(input_dim=dimension, output_dim=dimension, hid_size=50).to(device)
    FN_FEX.load_state_dict(torch.load(FNET_path_FEX, map_location=device))
    FN_FEX.eval()
    
    # Load TF-CDM
    FN_TF_CDM = None
    xTrain_mean_TF_CDM = None
    xTrain_std_TF_CDM = None
    yTrain_mean_TF_CDM = None
    yTrain_std_TF_CDM = None
    diff_scale_TF_CDM = None
    
    if All_stage_dir_TF_CDM is not None:
        data_inf_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'data_inf.pt')
        if os.path.exists(data_inf_path_TF_CDM):
            data_inf_TF_CDM = torch.load(data_inf_path_TF_CDM, map_location=device)
            xTrain_mean_TF_CDM = data_inf_TF_CDM['ZT_Train_mean'].to(device)
            xTrain_std_TF_CDM = data_inf_TF_CDM['ZT_Train_std'].to(device)
            yTrain_mean_TF_CDM = data_inf_TF_CDM['ODE_Train_mean'].to(device)
            yTrain_std_TF_CDM = data_inf_TF_CDM['ODE_Train_std'].to(device)
            diff_scale_TF_CDM = data_inf_TF_CDM['diff_scale']
            
            FNET_path_TF_CDM = os.path.join(All_stage_dir_TF_CDM, 'FNET.pth')
            if os.path.exists(FNET_path_TF_CDM):
                FN_TF_CDM = FN_Net(input_dim=dimension * 2, output_dim=dimension, hid_size=50).to(device)
                FN_TF_CDM.load_state_dict(torch.load(FNET_path_TF_CDM, map_location=device))
                FN_TF_CDM.eval()
    
    # Load FEX-VAE
    VAE_FEX = None
    if All_stage_dir_FEX_VAE is not None:
        VAE_path = os.path.join(All_stage_dir_FEX_VAE, 'VAE_FEX.pth')
        if os.path.exists(VAE_path):
            VAE_FEX = VAE(input_dim=dimension, hidden_dim=50, latent_dim=dimension).to(device)
            VAE_FEX.load_state_dict(torch.load(VAE_path, map_location=device))
            VAE_FEX.eval()
    
    # Load FEX-NN
    FEX_NN = None
    if All_stage_dir_FEX_NN is not None:
        FEX_NN_path = os.path.join(All_stage_dir_FEX_NN, 'FEX_NN.pth')
        if os.path.exists(FEX_NN_path):
            output_dim_nn = dimension * dimension if dimension > 1 else 1
            FEX_NN = CovarianceNet(input_dim=dimension, output_dim=output_dim_nn, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(FEX_NN_path, map_location=device))
            FEX_NN.eval()
    
    # Extract domain folder and boundaries
    domain_folder = None
    domain_start = -2.0
    domain_end = 2.0
    if second_stage_dir_FEX:
        path_parts = second_stage_dir_FEX.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                try:
                    parts = part.replace('domain_', '').split('_')
                    if len(parts) >= 2:
                        domain_start = float(parts[0])
                        domain_end = float(parts[1])
                except:
                    pass
                break
    
    # Construct base_path for FEX_model_learned
    base_path = os.path.dirname(os.path.dirname(second_stage_dir_FEX))
    
    def FEX(x):
        return FEX_model_learned(x, model_name=model_name,  
                                  noise_level=noise_level, device=device,
                                  domain_folder=domain_folder, base_path=base_path)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    x_dim = dimension
    Npath = 50000
    
    # Determine which models to compute
    models_to_compute = ["FEX-DM"]
    if FN_TF_CDM is not None and (domain_start <= initial_value <= domain_end):
        models_to_compute.append("TF-CDM")
    if VAE_FEX is not None:
        models_to_compute.append("FEX-VAE")
    if FEX_NN is not None and (domain_start <= initial_value <= domain_end):
        models_to_compute.append("FEX-NN")
    
    # Convert times to time indices
    time_indices = {t: int(t / sde_dt) for t in times_to_plot}
    max_time_idx = max(time_indices.values())
    
    # Initialize state dictionaries
    base_input = torch.clone((initial_value * torch.ones(Npath, x_dim)).to(device))
    x_pred_dict = {model: torch.clone(base_input) for model in models_to_compute}
    ode_path_true = initial_value * np.ones((Npath, x_dim))
    
    # Storage for samples at specific time points
    pred_samples = {model: {} for model in models_to_compute}
    true_samples = {}
    
    # Store initial state
    if 0 in time_indices.values():
        true_samples[0] = ode_path_true.copy()
        for model in models_to_compute:
            pred_samples[model][0] = x_pred_dict[model].cpu().numpy()
    
    # Simulate trajectories
    print(f"[INFO] Simulating trajectories for {max_time_idx + 1} time steps...")
    for jj in range(max_time_idx + 1):
        # Generate random noise (same for all models for fair comparison)
        z = torch.randn(Npath, x_dim).to(device, dtype=torch.float32)
        
        # Model predictions
        for model in models_to_compute:
            x_pred = x_pred_dict[model]
            
            if model == "FEX-DM":
                with torch.no_grad():
                    pred = FN_FEX((z - xTrain_mean_FEX) / xTrain_std_FEX) * yTrain_std_FEX + yTrain_mean_FEX
                    pred = (pred / diff_scale_FEX + x_pred + FEX(x_pred) * sde_dt).to("cpu").detach().numpy()
            
            elif model == "TF-CDM":
                with torch.no_grad():
                    input_tensor = torch.hstack((x_pred, z))
                    normed_input = (input_tensor - xTrain_mean_TF_CDM) / xTrain_std_TF_CDM
                    pred = FN_TF_CDM(normed_input) * yTrain_std_TF_CDM + yTrain_mean_TF_CDM
                    pred = (pred / diff_scale_TF_CDM + x_pred).to("cpu").detach().numpy()
            
            elif model == "FEX-VAE":
                with torch.no_grad():
                    pred = VAE_FEX.decoder(z)
                    pred = (pred / diff_scale_FEX + x_pred + FEX(x_pred) * sde_dt).to("cpu").detach().numpy()
            
            elif model == "FEX-NN":
                with torch.no_grad():
                    cov_pred = FEX_NN(x_pred)
                    if dimension == 1:
                        std_pred = torch.sqrt(torch.clamp(cov_pred, min=1e-8)).squeeze(-1)
                        pred = (x_pred.squeeze(-1) + FEX(x_pred).squeeze(-1) * sde_dt + 
                               std_pred * z.squeeze(-1) * np.sqrt(sde_dt)).to("cpu").detach().numpy()
                        pred = pred[:, np.newaxis]
                    else:
                        cov_matrix = cov_pred.reshape(Npath, dimension, dimension)
                        cov_matrix = cov_matrix + 1e-6 * torch.eye(dimension, device=device).unsqueeze(0)
                        try:
                            L = torch.linalg.cholesky(cov_matrix)
                            noise = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
                        except:
                            noise = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=1, dim2=2), min=1e-8)) * z
                        pred = (x_pred + FEX(x_pred) * sde_dt + noise * np.sqrt(sde_dt)).to("cpu").detach().numpy()
            
            # Store samples at specific time points
            if (jj + 1) in time_indices.values():
                pred_samples[model][jj + 1] = pred
            
            # Update state for next iteration
            x_pred_dict[model] = torch.tensor(pred).to(device, dtype=torch.float32)
        
        # True trajectory update (DoubleWell1d: dX = (X - X^3)dt + sig*dB)
        drift_true = ode_path_true - ode_path_true**3
        ode_path_true = ode_path_true + drift_true * sde_dt + sigma * np.sqrt(sde_dt) * np.random.randn(Npath, x_dim)
        
        if (jj + 1) in time_indices.values():
            true_samples[jj + 1] = ode_path_true.copy()
    
    # Create figure
    fig, axes = plt.subplots(1, len(times_to_plot), figsize=figsize)
    if len(times_to_plot) == 1:
        axes = [axes]
    
    # Plotting
    x_vals = np.linspace(-4, 4, 200)
    
    for i, t in enumerate(times_to_plot):
        idx = time_indices[t]
        ax = axes[i]
        
        # Ground Truth KDE
        if idx in true_samples:
            kde = gaussian_kde(true_samples[idx].T)
            ax.plot(x_vals, kde(x_vals), color='black', linewidth=2, linestyle='--', label="Ground Truth", zorder=10)
        
        # Model predictions - sort so FEX-DM is plotted last (on top)
        models_sorted = sorted(models_to_compute, key=lambda x: (x != "FEX-DM", x))
        
        for model in models_sorted:
            if idx in pred_samples[model]:
                # Set zorder: FEX-DM on top, others below
                if model == "FEX-DM":
                    plot_zorder = 5  # FEX-DM on top
                else:
                    plot_zorder = 2  # Other models below
                
                ax.hist(
                    pred_samples[model][idx],
                    bins=50, density=True,
                    alpha=0.5,
                    color=model_colors[model],
                    edgecolor=model_colors[model],
                    histtype='stepfilled',
                    label=model if i == 0 else "",  # Avoid repeated legends
                    zorder=plot_zorder
                )
        
        ax.set_title(f'$T = {t}$', fontsize=18)
        ax.set_xlim([-4, 4])
        ax.set_xlabel('$x$', fontsize=18)
        if i == 0:
            ax.set_ylabel('pdf', fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
    
    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=4, label='Ground Truth'),
    ]
    for model in models_to_compute:
        legend_handles.append(
            plt.Line2D([0], [0], color=model_colors[model], linestyle='-', linewidth=3, label=model)
        )
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=len(legend_handles), fontsize=18)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, 'conditional_1row_x1.5.pdf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {save_path}")
    
    return save_path
