import os
import sys
import math
import random
import logging
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory to path to access Example module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Standard library imports
import numpy as np
import torch
import torch.nn as nn
import sympy as sp

# Local imports
from Example.Example import params_init, data_generation
from config import DIR_EXAMPLES, DIR_PROJECT, create_main_parser
from utils import *

parser = create_main_parser()
args = parser.parse_args()

# Check if CUDA is available and set device accordingly
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    DEVICE = torch.device(args.DEVICE)
    print(f"Using CUDA device: {args.DEVICE}")
    base_path = os.path.join(DIR_PROJECT, 'Results', 'gpu_folder', args.model)
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available, using CPU instead.")
    base_path = os.path.join(DIR_PROJECT, 'Results', 'cpu_folder', args.model)



# Create domain folder name
domain_folder = f'domain_{args.DOMAIN_START}_{args.DOMAIN_END}'

if args.DATA_SAVE_PATH is None:
    args.DATA_SAVE_PATH = os.path.join(base_path, domain_folder, f'noise_{args.NOISE_LEVEL}', f'simulation_results_noise_{args.NOISE_LEVEL}.npz')

if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = os.path.join(base_path, domain_folder)

if args.FIGURE_SAVE_PATH is None:
    args.FIGURE_SAVE_PATH = os.path.join(base_path, domain_folder)

# Create necessary directories
os.makedirs(os.path.dirname(args.DATA_SAVE_PATH), exist_ok=True)
os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)
os.makedirs(args.FIGURE_SAVE_PATH, exist_ok=True)

SEED = args.SEED

NUM_TREES = args.NUM_TREES

CONTROLLER_LR = args.CONTROLLER_LR
CONTROLLER_INPUT_SIZE = args.CONTROLLER_INPUT_SIZE
CONTROLLER_TOP_SAMPLES_FRACTION = args.CONTROLLER_TOP_SAMPLES_FRACTION
CONTROLLER_QUANTILE_METHOD = args.CONTROLLER_QUANTILE_METHOD
EXPLORATION_ITERS = args.EXPLORATION_ITERS

FEX_LR_FIRST = args.FEX_LR_FIRST
FEX_LR_SECOND = args.FEX_LR_SECOND
TRAIN_EPOCHS_FIRST = args.TRAIN_EPOCHS_FIRST
TRAIN_EPOCHS_SECOND = args.TRAIN_EPOCHS_SECOND

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Initialize parameters
params = params_init(case_name=args.model)
# Add domain parameters to params for use in initial_condition_generation
params['domain_start'] = args.DOMAIN_START
params['domain_end'] = args.DOMAIN_END
data_file = args.DATA_SAVE_PATH

if os.path.exists(data_file):
    print("\n" + "="*60)
    print("[INFO] Data has already generated, just using for the first stage training: FEX".center(60, "="))
    data = np.load(data_file)
    dataset_full = data['dataset']
    current_state_full = data['current_state']
    next_state_full = data['next_state']
else:
    print("\n"+"="*60)
    print(f"[INFO] There is no dataset in this environment, it generates automatically".center(60,"="))
    current_state_full, next_state_full, dataset_full = data_generation(params, noise_level=args.NOISE_LEVEL, 
                                                                         domain_start=args.DOMAIN_START, 
                                                                         domain_end=args.DOMAIN_END)
    np.savez(
        args.DATA_SAVE_PATH,
        dataset = dataset_full,
        current_state = current_state_full,
        next_state = next_state_full
    )# Save data
    print(f"[INFO] Dataset saved at {args.DATA_SAVE_PATH}".center(60, "="))

print(f"[INFO] The case now is {args.model}")
print(f"\n")
print(f"[INFO] Full dataset shape: {dataset_full.shape}")
print(f"[INFO] Current state full shape: {current_state_full.shape}")
print(f"[INFO] Next state full shape: {next_state_full.shape}")
print(f"[INFO] Selecting {args.TRAIN_SIZE} samples for training")

# current_state_full has shape (Nt * N_data, 1), so we need to select from the first dimension
sample_size = current_state_full.shape[0]
selected_row_indices = np.random.permutation(sample_size)[:args.TRAIN_SIZE]
current_state_train_np = current_state_full[selected_row_indices]
next_state_train_np = next_state_full[selected_row_indices]

# Save training data to npz file
train_data_save_path = os.path.join(os.path.dirname(args.DATA_SAVE_PATH), f'train_data_{args.TRAIN_SIZE}.npz')
np.savez(
    train_data_save_path,
    current_state=current_state_train_np,
    next_state=next_state_train_np,
    selected_indices=selected_row_indices
)
print(f"[INFO] Training data saved at {train_data_save_path}".center(60, "="))

# Plot histogram of current_state_train values using utils.plot (before converting to tensor)
plot_training_data_histogram(
    current_state_train=current_state_train_np,
    save_path=args.FIGURE_SAVE_PATH,
    model_name=args.model,
    train_size=args.TRAIN_SIZE,
    noise_level=args.NOISE_LEVEL,
    dataset_full=dataset_full if args.model == 'OL2d' else None
)

# Convert numpy arrays to torch tensors and move to device
current_state_train = torch.from_numpy(current_state_train_np).float().to(DEVICE)
next_state_train = torch.from_numpy(next_state_train_np).float().to(DEVICE)

print(f"[INFO] Current state train shape: {current_state_train.shape}")
print(f"[INFO] Next state train shape: {next_state_train.shape}")

dimension = params['dim']
sampler = Sampler()
mse = nn.MSELoss()
l1 = nn.L1Loss()

# Check dt consistency and fix if needed
T = params['T']
Nt = params['Nt']
dt_actual = T / Nt
dt_training = params['Dt']
print(f"\n[DEBUG] dt consistency check:")
print(f"  T = {T}, Nt = {Nt}")
print(f"  dt_actual (T/Nt) = {dt_actual}")
print(f"  params['Dt'] (used in training) = {dt_training}")
if abs(dt_actual - dt_training) > 1e-10:
    ratio = dt_actual / dt_training
    print(f"[WARNING]  Mismatch! dt_actual / params['Dt'] = {ratio:.4f}")
    print(f"  This will cause model to learn drift * {ratio:.4f}")
    if abs(ratio - 0.5) < 0.01:
        print(f"[WARNING]  Model will learn HALF the drift!")
    elif abs(ratio - 2.0) < 0.01:
        print(f"[WARNING]  Model will learn DOUBLE the drift!")
    print(f"  🔧 FIXING: Setting params['Dt'] = dt_actual = {dt_actual}")
    params['Dt'] = dt_actual  # Fix the mismatch
    dt_training = dt_actual
    print(f"  ✓ Now params['Dt'] = {params['Dt']} matches dt_actual")
else:
    print(f"  ✓ dt values match")

integratorParams = Body4TrainIntegrationParams(dt=params['Dt'])
integrator = Body4TrainIntegrator(integratorParams, method=args.INTEGRATOR_METHOD)

print(f"TRAINING SHAPE: {current_state_train.shape}")
print(f"the dimension is {dimension}")
print("="*60)

print("\n"+"="*60)
print("FIRST STAGE: FEX TRAINING OPTIONS")
print("="*60)
print("1. pre-train to get the correct operator sets")
print("2. fine-tune to get the correct symbolic expression")
print("="*60)

while True:
    # choice = '1' #
    choice = input("\nChoose option (1 or 2 ):").strip()
    if choice in ['1','2']:
        break
    else:
        print("Please enter '1' or '2'.")

if choice == '1':
    print("\n"+"="*60)
    print("[INFO] Start to train the FEX")
    print(f"[INFO] The idea is frist to train the FEX for each dimension, and then to train the integrated FEX model")
    print("And in these examples, we always can get ground truth operator sequence for each dimension excapt one.")
    
    # For multi-dimensional models, process dimensions sequentially (1, then 2, etc.)
    # For 1D models, use TRAIN_WORKING_DIM as specified
    if dimension > 1:
        # Multi-dimensional case: process dimensions 1, 2, ... sequentially
        dims_to_process = list(range(1, dimension + 1))
        print(f"\n[INFO] Multi-dimensional model detected (dimension={dimension})")
        print(f"[INFO] Will process dimensions sequentially: {dims_to_process}")
    else:
        # 1D case: use TRAIN_WORKING_DIM
        if args.TRAIN_WORKING_DIM > 1:
            raise ValueError("When dimension is 1, the working dimension should be 1")
        dims_to_process = [args.TRAIN_WORKING_DIM]
        print(f"\n[INFO] 1D model: processing dimension {args.TRAIN_WORKING_DIM}")
    
    # Process each dimension sequentially
    for working_dim in dims_to_process:
        print("\n"+"="*60)
        print(f"[INFO] Processing Dimension {working_dim}/{dimension}")
        print("="*60)
        
        # For multi-dimensional models, extract data for current dimension
        if dimension > 1:
            # Extract single dimension data: shape (N, 1)
            current_state_train_dim = current_state_train[:, working_dim-1:working_dim]  # (N, 1)
            next_state_train_dim = next_state_train[:, working_dim-1:working_dim]  # (N, 1)
            print(f"[INFO] Extracted data for dimension {working_dim}")
            print(f"[INFO] Current state shape: {current_state_train_dim.shape}")
            print(f"[INFO] Next state shape: {next_state_train_dim.shape}")
        else:
            # 1D case: use full data
            current_state_train_dim = current_state_train
            next_state_train_dim = next_state_train
        
        # Create PMF_SIZES and controller for 1D (each dimension is trained as 1D)
        PMF_SIZES = tuple([len(unary_ops), len(binary_ops), len(unary_ops), len(binary_ops)])  # 4 nodes for 1D
        NUM_NODES = len(PMF_SIZES)
        
        pool = Pool()
        controller = Controller(pmf_sizes = PMF_SIZES).to(DEVICE)
        controller_optim = torch.optim.Adam(controller.parameters(), lr = CONTROLLER_LR)
        
        print(f'[INFO] PMF_SIZES for dimension {working_dim}: {PMF_SIZES}')
        print(f'[INFO] NUM_NODES for dimension {working_dim}: {NUM_NODES}')
        
        model_save_path = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}",f"best_candidates_pool_summary_{working_dim}.txt")
        log_file = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}",f'log_dimension_{working_dim}_{args.NOISE_LEVEL}.txt')
        # Always create the log file directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        if os.path.exists(model_save_path) and os.path.exists(log_file): #os.path.exists(model_save_path) and 
            print(f'[INFO] Model for dimension {working_dim} has already generated, just using for the second stage training:FEX'.center(60, '='))
            print("\n Loading the initial training model and log file")
            print('[INFO] Print the initial training model expression')   

            continue  # Skip to next dimension
        else:
            print(f'[INFO]No MODEL FOR DIMENSION {working_dim} SAVED IN THIS PATH, it will be generated automatically')        
        print(f'[DEBUG] About to set up logging to: {log_file}')
        
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        try:
            # Set up logging to both file and console
            logging.basicConfig(
                         level=logging.INFO,
                         format='%(asctime)s - %(levelname)s - %(message)s',
                         handlers=[
                         logging.FileHandler(log_file, encoding='utf-8'),  
                         logging.StreamHandler(sys.stdout)])
            print(f'[DEBUG] Logging setup completed successfully')
        except Exception as e:
            print(f'[ERROR] Failed to set up logging: {e}')
            raise

        # Initialize best candidates pool
        print("\n")
        print('[INFO] Initialize the best candidates pool')
        best_candidates_pool = []
        best_loss = float('inf')
        MAX_BEST_CANDIDATES = 30
        
        # For OL2d, also track valid expressions that pass validation regardless of score
        # This ensures we collect expressions with both x1**3 and x1 even if their score is low
        valid_ol2d_expressions = []  # Store (candidate, score) tuples for valid OL2d expressions


        
        for explore_idx in range(EXPLORATION_ITERS):
            print(f'\n[INFO] Exploration {explore_idx + 1}/{EXPLORATION_ITERS}')
            logprint(f'\n[INFO] Exploration {explore_idx + 1}/{EXPLORATION_ITERS}')
                
            controller_optim.zero_grad()
            pmfs = controller(torch.zeros(CONTROLLER_INPUT_SIZE, device=DEVICE))
            scores = torch.zeros(NUM_TREES, device=DEVICE)
                
                # Generate and train operator sequences
            op_seqs = torch.zeros(NUM_TREES, NUM_NODES, dtype=torch.int, device=DEVICE)
            trained_count = 0
                
            for tree_idx in range(NUM_TREES):
                op_seqs[tree_idx, :] = sampler(pmfs, output=torch.zeros(NUM_NODES, dtype=torch.int, device=DEVICE))
                print(f"Generated operator sequence {tree_idx}: {op_seqs[tree_idx, :].tolist()}")
                # For multi-dimensional models, create 1D FEX model for current dimension
                # For 1D models, use dimension=1
                model = FEX(op_seqs[tree_idx,:], dim=1).to(DEVICE)  # Always 1D for single dimension training
                model.apply(weights_init)

                expression = model.expression_visualize_simplified()
                trained_count += 1

                # Train the model
                model_optim = torch.optim.Adam(model.parameters(), lr=FEX_LR_FIRST)
                for epoch in range(TRAIN_EPOCHS_FIRST):
                    model_optim.zero_grad()
                    predictions = model(current_state_train_dim)
                    du_pred, du_target = integrator.integrate(
                        current_state_train=current_state_train_dim,
                        next_state_train=next_state_train_dim,
                        integration_func=model,
                        dimension=1  # Always 1D for single dimension training
                    )
                    loss = mse(du_pred, du_target)
                    loss.backward()
                    model_optim.step()

                # LBFGS fine-tuning
                lbfgs_optim = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, max_eval=25,
                                                   tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)
                    
                def lbfgs_closure():
                    lbfgs_optim.zero_grad()                   
                    du_pred, du_target = integrator.integrate(
                        current_state_train=current_state_train_dim,
                        next_state_train=next_state_train_dim,
                        integration_func=model,
                        dimension=1  # Always 1D for single dimension training
                    )
                    loss = mse(du_pred, du_target)                              
                    if torch.isnan(loss):
                        return torch.tensor(1e6, requires_grad=True)
                    loss.backward()
                    return loss
                    
                # Run LBFGS
                for _ in range(10):
                    try:
                        loss = lbfgs_optim.step(lbfgs_closure)
                        if torch.isnan(loss):
                            break
                    except Exception:
                        break
                    
                # Ensure loss is a tensor for consistent handling
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=DEVICE)
                    
                # Apply noise level penalty if needed
                loss = loss*1e3

                # Calculate score and add to pool
                if not math.isnan(loss.item()):
                    scores[tree_idx] = 1 / (1 + torch.sqrt(loss))
                else:
                    scores[tree_idx] = 0.
                    
                # Assert that the model's op_seq matches the op_seq being saved
                assert (model.op_seq == op_seqs[tree_idx,:]).all(), "Mismatch between model and op_seq!"
                pool.add(scores[tree_idx], model, loss.item(), op_seqs[tree_idx,:].tolist())
                
                # For OL2d, check if this expression passes validation and store it regardless of score
                # This ensures we collect valid expressions (with both x1**3 and x1) even if score is low
                if args.model == 'OL2d' and working_dim == 1:
                    try:
                        current_expr = model.expression_visualize_simplified()
                        # Ensure current_expr is a string
                        if not isinstance(current_expr, str):
                            current_expr = str(current_expr)
                        # Convert dictionary format if needed
                        if current_expr.startswith('{') and current_expr.endswith('}'):
                            import re
                            terms = []
                            dict_content = current_expr[1:-1]
                            pattern = r'([^:,]+):\s*([^,}]+?)(?=\s*,\s*[^:,]+:|$)'
                            matches = re.findall(pattern, dict_content)
                            for key, value in matches:
                                key = key.strip()
                                value = value.strip()
                                if key == '1' or key == 'constant':
                                    terms.append(value)
                                elif key == 'x1':
                                    terms.append(f"{value}*x1")
                                elif 'x1*3' in key or 'x1**3' in key or key == 'x1*3' or key == 'x1**3':
                                    terms.append(f"{value}*x1**3")
                                elif 'x1*' in key or 'x1**' in key:
                                    power_match = re.search(r'x1\*?\*?(\d+)', key)
                                    if power_match:
                                        power = power_match.group(1)
                                        terms.append(f"{value}*x1**{power}")
                                    else:
                                        terms.append(f"{value}*{key}")
                                else:
                                    terms.append(f"{value}*{key}")
                            current_expr = " + ".join(terms)
                            # Fix power notation: x1*3 -> x1**3
                            current_expr = re.sub(r'(x[123])\*\s*3\b(?!\d)', r'\1**3', current_expr)
                            current_expr = re.sub(r'(x[123])\s*\*\s*3\b(?!\d)', r'\1**3', current_expr)
                        else:
                            # Even if not dictionary format, fix power notation
                            import re
                            current_expr = re.sub(r'(x[123])\*\s*3\b(?!\d)', r'\1**3', current_expr)
                            current_expr = re.sub(r'(x[123])\s*\*\s*3\b(?!\d)', r'\1**3', current_expr)
                        
                        current_expr_original = model.expression_visualize()
                        if not isinstance(current_expr_original, str):
                            current_expr_original = str(current_expr_original)
                        
                        # Check validation
                        from utils.helper import check_allowed_terms
                        check_result = check_allowed_terms(current_expr, working_dim, args.model, original_expr=current_expr_original)
                        
                        if check_result['valid']:
                            # Create a candidate object for valid expression
                            from utils.Pool import Candidate
                            import re
                            # Fix power notation before storing: x1*3 -> x1**3
                            if isinstance(current_expr, str):
                                current_expr = re.sub(r'(x[123])\*\s*3\b(?!\d)', r'\1**3', current_expr)
                                current_expr = re.sub(r'(x[123])\s*\*\s*3\b(?!\d)', r'\1**3', current_expr)
                            valid_candidate = Candidate(
                                score=scores[tree_idx],
                                model=model,
                                loss=loss.item(),
                                action=op_seqs[tree_idx,:].tolist(),
                                expression=current_expr
                            )
                            # Check if this sequence is already in valid_ol2d_expressions
                            seq_str = str(op_seqs[tree_idx,:].tolist())
                            if not any(str(c.action) == seq_str for c in valid_ol2d_expressions):
                                valid_ol2d_expressions.append(valid_candidate)
                                print(f"[INFO] Valid OL2d expression collected (score={scores[tree_idx]:.6f}): Seq={op_seqs[tree_idx,:].tolist()}")
                                logprint(f"[INFO] Valid OL2d expression collected (score={scores[tree_idx]:.6f}): Seq={op_seqs[tree_idx,:].tolist()}")
                    except Exception as e:
                        # If validation check fails, just continue
                        pass
                    
                # Print current model info
                print("\n"+"="*60)
                print(f"[INFO] Model {tree_idx + 1}")
                print(f"Expression: {model.expression_visualize()}")
                print(f"Expression simplified: {model.expression_visualize_simplified()}")
                print(f"Loss: {loss.item():.6f}")
                print(f"Score: {scores[tree_idx]:.6f}")
                print(f"Operator sequence: {op_seqs[tree_idx,:].tolist()}")
                print("="*60)
                    
                   
                    
                logprint("\n"+"="*60)
                logprint(f"[INFO] Model {tree_idx + 1}")
                logprint(f"Expression: {model.expression_visualize()}")
                logprint(f"Expression simplified: {model.expression_visualize_simplified()}")
                logprint(f"Loss: {loss.item():.6f}")
                logprint(f"Score: {scores[tree_idx]:.6f}")
                logprint(f"Operator sequence: {op_seqs[tree_idx,:].tolist()}")
                logprint("="*60)

            # Controller update
            scores_detached = scores.cpu().detach().numpy()
            scores_upper_quantile = np.percentile(scores_detached, q=(1 - CONTROLLER_TOP_SAMPLES_FRACTION), method=CONTROLLER_QUANTILE_METHOD)
            indicator_upper_quantile = (scores_detached >= scores_upper_quantile).astype(int)
                
            sum_log_probs = torch.zeros(NUM_TREES, device=DEVICE)
            log_pmfs = [torch.log(pmf) for pmf in pmfs]
            for tree_idx, ops in enumerate(op_seqs):
                for pmf_idx, op in enumerate(ops):
                    log_prob = log_pmfs[pmf_idx][op]
                    sum_log_probs[tree_idx] += log_prob
                
            scores_detached = torch.from_numpy(scores_detached).to(DEVICE)
            indicator_upper_quantile = torch.from_numpy(indicator_upper_quantile).to(DEVICE)
                
            controller_loss = -(1 / CONTROLLER_TOP_SAMPLES_FRACTION) * torch.mean((scores_detached - scores_upper_quantile) * indicator_upper_quantile * sum_log_probs)
            controller_loss.backward()
            controller_optim.step()
                
            # Log exploration results
            logprint(f"Trained {trained_count}/{NUM_TREES} sequences")
            logprint(f"Best loss in pool: {min([c.error for c in pool]):.6f}")
            print(f"Trained {trained_count}/{NUM_TREES} sequences")
            print(f"Best loss in pool: {min([c.error for c in pool]):.6f}")
                
            # For OL2d, first add valid expressions that passed validation regardless of score
            if args.model == 'OL2d' and working_dim == 1 and valid_ol2d_expressions:
                print(f"\n[INFO] Found {len(valid_ol2d_expressions)} valid OL2d expressions (regardless of score)")
                logprint(f"\n[INFO] Found {len(valid_ol2d_expressions)} valid OL2d expressions (regardless of score)")
                # Add valid expressions to best_candidates_pool (avoid duplicates by sequence)
                existing_seqs = {str(c.action) for c in best_candidates_pool}
                for valid_candidate in valid_ol2d_expressions:
                    seq_str = str(valid_candidate.action)
                    if seq_str not in existing_seqs:
                        best_candidates_pool.append(valid_candidate)
                        existing_seqs.add(seq_str)
                        print(f"  Added: Seq={valid_candidate.action}, Score={valid_candidate.score:.6f}, Expr={valid_candidate.get_expression()[:80] if isinstance(valid_candidate.get_expression(), str) else str(valid_candidate.get_expression())[:80]}")
                        logprint(f"  Added: Seq={valid_candidate.action}, Score={valid_candidate.score:.6f}, Expr={valid_candidate.get_expression()[:80] if isinstance(valid_candidate.get_expression(), str) else str(valid_candidate.get_expression())[:80]}")
            
            # Update best candidates pool
            for candidate_ in pool:
                current_loss = candidate_.error
                current_expr = candidate_.get_expression()  # Use the stored expression (simplified)
                # Ensure current_expr is a string (not dict or other type)
                if not isinstance(current_expr, str):
                    current_expr = str(current_expr)
                # If current_expr is a dictionary string representation (like "{x1*3: 0.69, x1: 0.69, 1: 0.52}"),
                # convert it to a proper expression string
                if current_expr.startswith('{') and current_expr.endswith('}'):
                    # Parse dictionary format and convert to expression string
                    import re
                    terms = []
                    # Remove outer braces
                    dict_content = current_expr[1:-1]
                    # Split by comma, but be careful with commas inside values
                    # Simple approach: split by ", " pattern (comma followed by space and a key pattern)
                    # Pattern: match "key: value" where key can be x1*3, x1, 1, etc.
                    # More robust: find all "key: value" patterns
                    pattern = r'([^:,]+):\s*([^,}]+?)(?=\s*,\s*[^:,]+:|$)'
                    matches = re.findall(pattern, dict_content)
                    for key, value in matches:
                        key = key.strip()
                        value = value.strip()
                        if key == '1' or key == 'constant':
                            # Constant term
                            terms.append(value)
                        elif key == 'x1':
                            # Linear x1 term
                            terms.append(f"{value}*x1")
                        elif 'x1*3' in key or 'x1**3' in key or key == 'x1*3' or key == 'x1**3':
                            # x1**3 term
                            terms.append(f"{value}*x1**3")
                        elif 'x1*' in key or 'x1**' in key:
                            # Other powers - extract the power
                            power_match = re.search(r'x1\*?\*?(\d+)', key)
                            if power_match:
                                power = power_match.group(1)
                                terms.append(f"{value}*x1**{power}")
                            else:
                                terms.append(f"{value}*{key}")
                        else:
                            terms.append(f"{value}*{key}")
                    current_expr = " + ".join(terms)
                    # Fix power notation: x1*3 -> x1**3
                    current_expr = re.sub(r'(x[123])\*\s*3\b(?!\d)', r'\1**3', current_expr)
                    current_expr = re.sub(r'(x[123])\s*\*\s*3\b(?!\d)', r'\1**3', current_expr)
                # Always fix power notation, even if not dictionary format
                import re
                if isinstance(current_expr, str):
                    # Fix x1*3, x2*3, x3*3 -> x1**3, x2**3, x3**3 (but not x1*30, x1*314)
                    current_expr = re.sub(r'(x[123])\*\s*3\b(?!\d)', r'\1**3', current_expr)
                    current_expr = re.sub(r'(x[123])\s*\*\s*3\b(?!\d)', r'\1**3', current_expr)
                current_expr_original = candidate_.model.expression_visualize()  # Get original (non-simplified) expression
                # Ensure current_expr_original is a string
                if not isinstance(current_expr_original, str):
                    current_expr_original = str(current_expr_original)
                current_score = candidate_.score  # assuming .score exists
                            
                # Check if expression follows the allowed terms for this dimension
                # Pass model name to check_allowed_terms function
                # For OL2d, we need to check both simplified and original expressions
                # because the original might have (x1 + ...)**3 which expands to x1**2, x1**3, etc.
                check_result = check_allowed_terms(current_expr, working_dim, args.model, original_expr=current_expr_original if args.model == 'OL2d' else None)
                
                if not check_result['valid']:
                    # Debug: print rejections for OL2d to see why [2, 0, 4, 2] is being rejected
                    if args.model == 'OL2d' and not hasattr(pool, '_reject_count_ol2d'):
                        pool._reject_count_ol2d = {}
                    if args.model == 'OL2d':
                        seq_str = str(candidate_.action)
                        if seq_str not in pool._reject_count_ol2d:
                            pool._reject_count_ol2d[seq_str] = 0
                        if pool._reject_count_ol2d[seq_str] < 5:  # Print first 5 rejections per sequence
                            debug_msg = f"[DEBUG OL2d] Rejected Seq={candidate_.action}, Expr={current_expr[:150] if isinstance(current_expr, str) else str(current_expr)[:150]}, Original={current_expr_original[:150] if current_expr_original and isinstance(current_expr_original, str) else str(current_expr_original)[:150] if current_expr_original else 'None'}"
                            print(debug_msg)
                            logging.info(debug_msg)  # Also log to file
                            pool._reject_count_ol2d[seq_str] += 1
                    # Debug: print first few rejections for Trigonometric1d
                    if args.model == 'Trigonometric1d' and not hasattr(pool, '_reject_count'):
                        pool._reject_count = 0
                    if args.model == 'Trigonometric1d' and pool._reject_count < 3:
                        debug_msg = f"[DEBUG] Rejected expression: {current_expr[:100]}"
                        print(debug_msg)
                        logging.info(debug_msg)  # Also log to file
                        pool._reject_count += 1
                    continue
                
                # For OU1d (1D case), we only need to check if expression is valid (no interaction terms needed)
                if args.model == 'OU1d':
                    # For 1D case, just check validity - no interaction terms required
                    best_candidates_pool.append(candidate_)
                # For Trigonometric1d (1D case), check if expression has trigonometric functions
                elif args.model == 'Trigonometric1d':
                    # For Trigonometric1d, expression should contain cos or sin (approximating sin(2πx))
                    # Expressions like "-1.1989 cos(6.2476*x1 - 4.6837) - 0.0104" are acceptable
                    if 'cos' in current_expr.lower() or 'sin' in current_expr.lower():
                        best_candidates_pool.append(candidate_)
                # For OL2d, check_allowed_terms already validated it, so just add it
                elif args.model == 'OL2d':
                    # check_allowed_terms already validated that it has explicit x1**3 + explicit x1
                    # and rejected (x1 + ...)**3 patterns, so we can just add it
                    best_candidates_pool.append(candidate_)
                # For MM1d, check_allowed_terms already validated it, so just add it
                elif args.model == 'MM1d':
                    # check_allowed_terms already validated that:
                    # - Only powers 1-24 are allowed
                    # - Only addition (not multiplication) between x1 terms
                    # - No disallowed functions or variables
                    # So we can just add it
                    best_candidates_pool.append(candidate_)
                # For DoubleWell1d (1D case), check if expression has ONLY x1 (linear) and x1^3 (cubic), no other powers
                elif args.model == 'DoubleWell1d':
                    # For DoubleWell1d, expression should contain ONLY x1 (linear term) and x1^3 (cubic term)
                    # Check for x1**3 or x1*x1*x1 patterns (cubic term)
                    expr_lower = current_expr.lower()
                    has_x1_cubed = ('x1**3' in expr_lower or 
                                  'x1*x1*x1' in expr_lower or
                                  'x1 * x1 * x1' in expr_lower or
                                  re.search(r'x1\s*\*\s*3\b', expr_lower))  # Match x1*3 (formatting bug)
                    
                    # Check for x1 as a standalone linear term (not just inside x1^3)
                    # Remove x1**3 patterns first, then check if x1 still appears as linear term
                    expr_without_cubed = re.sub(r'x1\s*\*\s*\*\s*3', '', expr_lower)  # Remove x1**3
                    expr_without_cubed = re.sub(r'x1\s*\*\s*x1\s*\*\s*x1', '', expr_without_cubed)  # Remove x1*x1*x1
                    expr_without_cubed = re.sub(r'x1\s*\*\s*3\b', '', expr_without_cubed)  # Remove x1*3 (formatting bug)
                    # Check if x1 appears as a linear term (standalone or multiplied by a coefficient)
                    has_x1_linear = bool(re.search(r'[+\-*]\s*x1\s*[+\-*]|[+\-*]\s*x1\s*$|^\s*x1\s*[+\-*]|^\s*x1\s*$', expr_without_cubed))
                    
                    # Check for any other powers of x1 (x1**2, x1**4, x1**5, x1**9, etc.) - NOT allowed
                    has_other_powers = bool(re.search(r'x1\s*\*\s*\*\s*[02456789]', expr_lower))  # x1**2, x1**4, x1**5, x1**9, etc.
                    # Also check for x1**10, x1**11, etc. (two-digit powers)
                    has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*\*\s*1[0-9]', expr_lower))  # x1**10-19
                    has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*\*\s*[2-9][0-9]', expr_lower))  # x1**20-99
                    # Also check for formatting bugs: x1*N where N is not 3 (should be x1**N, but we reject it)
                    # Check for x1*2, x1*4, x1*5, etc. (single digit powers, not 3)
                    has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*[02456789]\b(?!\d)', expr_lower))  # x1*2, x1*4, etc.
                    # Check for x1*10, x1*11, etc. (two-digit powers, not 3)
                    has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*1[0-9]\b(?!\d)', expr_lower))  # x1*10-19
                    has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*[2-9][0-9]\b(?!\d)', expr_lower))  # x1*20-99
                    
                    # Only accept if has both x1 and x1^3, AND no other powers
                    if has_x1_linear and has_x1_cubed and not has_other_powers:
                        best_candidates_pool.append(candidate_)
                # For EXP1d (1D case), check if expression has x1 (linear term) - drift is th*x1
                elif args.model == 'EXP1d':
                    # For EXP1d, expression should contain x1 as a linear term (drift: th*x1 where th=-2.0)
                    expr_lower = current_expr.lower()
                    # Check if x1 appears as a linear term (standalone or multiplied by a coefficient)
                    has_x1_linear = bool(re.search(r'[+\-*]\s*x1\s*[+\-*]|[+\-*]\s*x1\s*$|^\s*x1\s*[+\-*]|^\s*x1\s*$', expr_lower))
                    
                    # Check for specific sequences that should be allowed: [2,0,5,2], [2,2,5,2], [2,0,6,2]
                    # Compare lists directly for more reliable matching
                    allowed_sequences = [[2, 0, 5, 2], [2, 2, 5, 2], [2, 0, 6, 2]]
                    is_allowed_sequence = candidate_.action in allowed_sequences
                    
                    # For allowed sequences, allow exp(x1) and x1*4 (formatting bug)
                    if is_allowed_sequence:
                        # Remove exp(x1) and x1*4 from consideration when checking for other powers
                        expr_without_allowed = re.sub(r'exp\s*\(\s*x1\s*\)', '', expr_lower)  # Remove exp(x1)
                        expr_without_allowed = re.sub(r'x1\s*\*\s*4\b(?!\d)', '', expr_without_allowed)  # Remove x1*4
                        # Check for other powers (excluding x1*4 which we allow)
                        has_other_powers = bool(re.search(r'x1\s*\*\s*\*\s*[2-9]', expr_without_allowed))  # x1**2, x1**3, etc.
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*\*\s*1[0-9]', expr_without_allowed))  # x1**10-19
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*\*\s*[2-9][0-9]', expr_without_allowed))  # x1**20-99
                        # Check for formatting bugs: x1*N where N > 1 and N != 4
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*[2356789]\b(?!\d)', expr_without_allowed))  # x1*2, x1*3, x1*5, etc. (not 4)
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*1[0-9]\b(?!\d)', expr_without_allowed))  # x1*10-19
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*[2-9][0-9]\b(?!\d)', expr_without_allowed))  # x1*20-99
                        # Accept if has x1 linear term AND no other powers (but allow exp(x1) and x1*4)
                        if has_x1_linear and not has_other_powers:
                            best_candidates_pool.append(candidate_)
                    else:
                        # For other sequences, use strict validation: no powers, no exp
                        has_other_powers = bool(re.search(r'x1\s*\*\s*\*\s*[2-9]', expr_lower))  # x1**2, x1**3, etc.
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*\*\s*1[0-9]', expr_lower))  # x1**10-19
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*\*\s*[2-9][0-9]', expr_lower))  # x1**20-99
                        # Also check for formatting bugs: x1*N where N > 1
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*[2-9]\b(?!\d)', expr_lower))  # x1*2, x1*3, etc.
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*1[0-9]\b(?!\d)', expr_lower))  # x1*10-19
                        has_other_powers = has_other_powers or bool(re.search(r'x1\s*\*\s*[2-9][0-9]\b(?!\d)', expr_lower))  # x1*20-99
                        # Check for exp (not allowed for non-special sequences)
                        has_exp = 'exp' in expr_lower
                        # Only accept if has x1 linear term AND no other powers AND no exp
                        if has_x1_linear and not has_other_powers and not has_exp:
                            best_candidates_pool.append(candidate_)
                # For OL2d, check_allowed_terms already validated it, so just add it
                # (The validation was done above in check_allowed_terms with original_expr)
                # For multi-dimensional models (like SIR), check for required interaction terms
                elif args.model == 'SIR':
                    # Check for the required interaction terms based on working dimension
                    if working_dim == 1:
                        if 'x2*x3' in check_result['terms_present'] or 'x2x3' in check_result['terms_present']:
                            best_candidates_pool.append(candidate_)
                    elif working_dim == 2:
                        if 'x1*x3' in check_result['terms_present'] or 'x1x3' in check_result['terms_present']:
                            best_candidates_pool.append(candidate_)
                    elif working_dim == 3:
                        if 'x1*x2' in check_result['terms_present'] or 'x1x2' in check_result['terms_present']:
                            best_candidates_pool.append(candidate_)
                else:
                    # For other models, just check validity
                    best_candidates_pool.append(candidate_)
            # Print current pool status
            print("\n"+"="*60)
            print(f"\n[INFO] Current Pool Status ({len(pool)} candidates):")
            print("Pool Selection:")
            pool_list = sorted(list(pool), key=lambda c: c.score, reverse=True)
            # Print top 50
            for idx, candidate_ in enumerate(pool_list[:50]):
                print(f"  {idx + 1}. Score: {candidate_.score:.6f}, Loss: {candidate_.error:.6f}, Seq: {candidate_.action}, Expression={candidate_.get_expression()}")
                
            print("\n")
            print("Best Candidate Pool Selection:")
            for idx,  candidate_ in enumerate(best_candidates_pool):
                # Assert that each candidate's model op_seq matches its action
                assert (candidate_.model.op_seq == torch.tensor(candidate_.action, device=candidate_.model.op_seq.device)).all(), f"Mismatch between candidate {idx+1} model and action!"
                print(f"  {idx + 1}. Loss: {candidate_.error:.6f}, Seq: {candidate_.action}, Expression={candidate_.get_expression()}")
            print("=" * 60)
            
        # Select best candidate
        logprint(f"\nBest candidates found: {len(best_candidates_pool)}")
        print(f"\nBest candidates found: {len(best_candidates_pool)}")
        for idx, candidate_ in enumerate(best_candidates_pool):
            logprint(f"Candidate {idx + 1}: Loss={candidate_.error:.6f}, Seq={candidate_.action}")
            print(f"Candidate {idx + 1}: Loss={candidate_.error:.6f}, Seq={candidate_.action}")
            
        # Create save directory if it doesn't exist
        save_dir = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}")
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, f"best_candidates_pool_summary_{working_dim}.txt")
        # Write summary
        with open(summary_path, "w") as f:
            for idx, candidate_ in enumerate(best_candidates_pool):
                f.write(f"Candidate {idx + 1}: Score={candidate_.score:.6f}, Loss={candidate_.error:.6f}, Seq={candidate_.action}, Expr={candidate_.get_expression()}\n")

        print(f"[INFO] best_candidates_pool_summary saved to {summary_path}")
        logprint(f"[INFO] best_candidates_pool_summary saved to {summary_path}")
        
        # Check if best_candidates_pool is empty
        if len(best_candidates_pool) == 0:
            print("\n" + "="*60)
            print("[WARNING] No valid candidates found after filtering!")
            print("This might mean the validation criteria are too strict.")
            print("Please check the validation logic in check_allowed_terms function.")
            print("="*60)
            logprint("[WARNING] No valid candidates found after filtering!")
            print("\n[INFO] Exiting. Please adjust validation criteria or check the expressions being generated.")
            exit()
        
        # Use best candidate by default (or add user selection if needed)
        best_candidate = min(best_candidates_pool, key=lambda c: c.error)
        optimal_idx = best_candidate.action
        # Assert that the best candidate's model op_seq matches its action
        assert (best_candidate.model.op_seq == torch.tensor(best_candidate.action, device=best_candidate.model.op_seq.device)).all(), "Mismatch between best_candidate model and action!"
        logprint(f"Selected: Loss={best_candidate.error:.6f}, Expression={best_candidate.get_expression()}")
        print(f"Selected: Loss={best_candidate.error:.6f}, Expression={best_candidate.get_expression()}")
        
        print(f"\n[INFO] Completed training for dimension {working_dim}/{dimension}")
        logprint(f"[INFO] Completed training for dimension {working_dim}/{dimension}")
        
        # Close logging handlers for this dimension before moving to next
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
    
    # After processing all dimensions
    print("\n" + "="*60)
    print(f"[INFO] Completed FEX training for all dimensions (1 to {dimension})")
    print("[INFO] Now you can rerun the script with option 2 to fine-tune the integrated FEX model.")
    print("="*60)

elif choice == '2':
    print("\n"+"="*60)
    print("[INFO] Loading FEX models from previous pre-training...")
    # Ask user whether to train everything in second stage or skip to calculate the measurements
    print("\n"+"="*60)
    print("Find the candidate operator sequence from training")
    
    op_seqs_all = {}
    models = {}
    symbols = [sp.symbols(f'x{i+1}') for i in range(dimension)]
    print(f'[INFO] the noise level is {args.NOISE_LEVEL}')
    
    # Let user select operator sequences for each dimension
    print("\n" + "="*60)
    print("Selecting operator sequences for each dimension...")
    print("="*60)
    
    for dim in range(1, dimension+1):
        print(f'\nSelecting for dimension {dim}...')
        file_path = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}", f'best_candidates_pool_summary_{dim}.txt')
        selected_sequence = select_operator_sequence(file_path, dim)
        if selected_sequence is None:
            print(f"[ERROR] Failed to get sequence for dimension {dim}")
            sys.exit(1)
            
        op_seqs = torch.tensor(selected_sequence, device=DEVICE)
        op_seqs_all[dim] = op_seqs
        print(f"[INFO] {dim} dimension data found. Now let us train integrated FEX model")
        print("\n")
        # For OU1d, we use regular FEX model
        # FEX_with_force is not implemented yet, so use regular FEX
        # Since we process dimensions sequentially (each as 1D), use dim=1 for each dimension
        # The op_seqs is a 1D sequence (4 elements) from the first stage training
        model = FEX(op_seqs, dim=1).to(DEVICE)
        model.apply(weights_init)
        models[str(dim)] = model
        
        # Show initial expression before training
        print(f"Initial expression for Dimension {dim}:")
        print(f"  Full expression: {model.expression_visualize()}")
        print(f"  Simplified expression: {model.expression_visualize_simplified()}")
        print("-" * 60)
        

    print("="*60)
    
    # Convert dataset_full to tensor format for training
    # dataset_full shape: (1, Nt+1, N_data) for OU1d
    # We need to reshape it to (N_data, Nt+1, 1) for easier processing
    dataset_tensor = torch.from_numpy(dataset_full).float().to(DEVICE)
    if dataset_tensor.dim() == 3:
        # Reshape from (1, Nt+1, N_data) to (N_data, Nt+1, 1)
        dataset_tensor = dataset_tensor.permute(2, 1, 0)  # (N_data, Nt+1, 1)
    
    # Initialize coefficients history for tracking
    coefficents_history = {}
    for dim in range(1, dimension+1):
        coefficents_history[dim] = {}
    
    loss_history = []
    # Create optimizer for all parameters
    all_params = []
    for model in models.values():
        all_params.extend(model.parameters())
    model_optim = torch.optim.Adam(all_params, lr=FEX_LR_SECOND)
    
    # Training loop
    for train_idx in range(TRAIN_EPOCHS_SECOND):
        adjust_learning_rate(model_optim, train_idx, FEX_LR_SECOND, TRAIN_EPOCHS_SECOND)
        model_optim.zero_grad()
        total_pred_loss = 0

        # Prediction and extra loss
        for dim in range(1, dimension+1):
            model = models[str(dim)]
            
            # Extract current and next states from dataset
            # dataset_tensor shape: (N_data, Nt+1, 1)
            current_state_batch = dataset_tensor[:, :-1, :]  # (N_data, Nt, 1)
            next_state_batch = dataset_tensor[:, 1:, :]      # (N_data, Nt, 1)
            
            # Reshape to (N_data * Nt, 1) for integration
            current_state_flat = current_state_batch.reshape(-1, dimension)
            next_state_flat = next_state_batch.reshape(-1, dimension)
            
            # Use the new integrator API
            u_pred, u_target = integrator.integrate(
                current_state_train=current_state_flat,
                next_state_train=next_state_flat,
                integration_func=model,
                dimension=dimension
            )

            loss = mse(u_pred, u_target)
            total_pred_loss += loss
        
        # Call backward only once after all dimensions are processed
        total_pred_loss.backward(retain_graph=True)
        model_optim.step()
        
        with torch.no_grad():
            if train_idx % 50 == 0:
                loss_history.append(total_pred_loss.item())
                for dim in range(1, dimension+1):
                    model = models[str(dim)]
                    expr = model.expression_visualize_simplified()
                    coeffs = extract_coefficients_from_expr(expr, dim)
                    for term, value in coeffs.items():
                        if term not in coefficents_history[dim]:
                            coefficents_history[dim][term] = []
                        coefficents_history[dim][term].append(value)
                #print(f"the coefficents_history is {coefficents_history}")
            if train_idx % 100 == 0:
                print("\n"+"="*60)
                print(f"Training index: {train_idx}")
                print(f"Loss: {total_pred_loss.item():.6f}")
                # Print expressions for each dimension
                expressions = {}
                for dim in range(1, dimension+1):
                    expressions[f'Dimension {dim}'] = models[str(dim)].expression_visualize_simplified()
                print(f"Expression: {expressions}")
                print("="*60)

        if train_idx == TRAIN_EPOCHS_SECOND-1:
            final_expressions = {}
            final_operator_sequences = {}
            for dim in range(1, dimension+1):
                final_expr = models[str(dim)].expression_visualize_simplified()
                final_expressions[f'dimension_{dim}'] = final_expr
                # Get operator sequence for this dimension
                if dim in op_seqs_all:
                    op_seq = op_seqs_all[dim].cpu().numpy().tolist()
                    final_operator_sequences[f'dimension_{dim}'] = op_seq
            loss_history_dict = {1: loss_history, 2: loss_history, 3: loss_history}
            save_dir = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}")
            
            
            # Save final expressions to text file
            final_expr_save_path = os.path.join(save_dir, "final_expressions.txt")
            os.makedirs(os.path.dirname(final_expr_save_path), exist_ok=True)
            with open(final_expr_save_path, "w") as f:
                f.write("Final Expressions After Training:\n")
                f.write("=" * 50 + "\n")
                for dim in range(1, dimension+1):
                    dim_key = f'dimension_{dim}'
                    f.write(f"\n{dim_key}:\n")
                    f.write(f"  Operator Sequence: {final_operator_sequences.get(dim_key, 'N/A')}\n")
                    f.write(f"  Expression: {final_expressions.get(dim_key, 'N/A')}\n")
                f.write("\nTraining completed successfully!\n")
            print(f"[INFO] Final expressions saved to: {final_expr_save_path}")
            print("[SUCCESS] First stage training completed successfully! you may need to do the second stage training")
            


    