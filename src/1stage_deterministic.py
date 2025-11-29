import os
import sys
import math
import random
import logging

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



if args.DATA_SAVE_PATH is None:
    args.DATA_SAVE_PATH = os.path.join(base_path, f'noise_{args.NOISE_LEVEL}', f'simulation_results_noise_{args.NOISE_LEVEL}.npz')

if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}'

if args.FIGURE_SAVE_PATH is None:
    args.FIGURE_SAVE_PATH = f'{base_path}'

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
    current_state_full, next_state_full, dataset_full = data_generation(params, noise_level=args.NOISE_LEVEL)
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
    train_size=args.TRAIN_SIZE
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
integratorParams = Body4TrainIntegrationParams(dt=params['Dt'])
integrator = Body4TrainIntegrator(integratorParams, method=args.INTEGRATOR_METHOD)
pool = Pool()

PMF_SIZES = tuple([len(unary_ops), len(binary_ops), len(unary_ops), len(binary_ops)] * dimension)
NUM_NODES = len(PMF_SIZES)

controller = Controller(pmf_sizes = PMF_SIZES).to(DEVICE)
controller_optim = torch.optim.Adam(controller.parameters(), lr = CONTROLLER_LR)

print(f"TRAINING SHAPE: {current_state_train.shape}")
print(f"the dimension is {dimension}")
print(f'the PMF_SIZES is {PMF_SIZES}')
print(f'the NUM_NODES is {NUM_NODES}')
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
    
    print("\n"+"="*60)
    print(f" Working dimension is {args.TRAIN_WORKING_DIM}")
    if dimension == 1 and args.TRAIN_WORKING_DIM> 1:
        raise ValueError("When dimension is 1, the working dimension should be 1")
    model_save_path = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}",f"best_candidates_pool_summary_{args.TRAIN_WORKING_DIM}.txt")
    log_file = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}",f'log_dimension_{args.TRAIN_WORKING_DIM}_{args.NOISE_LEVEL}.txt')
    # Always create the log file directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if os.path.exists(model_save_path) and os.path.exists(log_file): #os.path.exists(model_save_path) and 
        print(f'[INFO] Model for dimension {args.TRAIN_WORKING_DIM} has already generated, just using for the second stage training:FEX'.center(60, '='))
        print("\n Loading the initial training model and log file")
        print('[INFO] Print the initial training model expression')   

        pass
    else:
        print(f'[INFO]No MODEL FOR DIMENSION {args.TRAIN_WORKING_DIM} SAVED IN THIS PATH, it will be generated automatically')        
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
                model = FEX(op_seqs[tree_idx,:], dim=dimension).to(DEVICE)
                model.apply(weights_init)

                expression = model.expression_visualize_simplified()
                trained_count += 1

                # Train the model
                model_optim = torch.optim.Adam(model.parameters(), lr=FEX_LR_FIRST)
                for epoch in range(TRAIN_EPOCHS_FIRST):
                    model_optim.zero_grad()
                    predictions = model(current_state_train)
                    du_pred, du_target = integrator.integrate(
                        current_state_train=current_state_train,
                        next_state_train=next_state_train,
                        integration_func=model,
                        dimension=dimension
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
                        current_state_train=current_state_train,
                        next_state_train=next_state_train,
                        integration_func=model,
                        dimension=dimension
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
                
            # Update best candidates pool
            for candidate_ in pool:
                current_loss = candidate_.error
                current_expr = candidate_.get_expression()  # Use the stored expression
                current_score = candidate_.score  # assuming .score exists
                            
                # Check if expression follows the allowed terms for this dimension
                # Pass model name to check_allowed_terms function
                check_result = check_allowed_terms(current_expr, args.TRAIN_WORKING_DIM, args.model)
                
                if not check_result['valid']:
                    continue
                
                # For OU1d (1D case), we only need to check if expression is valid (no interaction terms needed)
                if args.model == 'OU1d':
                    # For 1D case, just check validity - no interaction terms required
                    best_candidates_pool.append(candidate_)
                # For multi-dimensional models (like SIR), check for required interaction terms
                elif args.model == 'SIR':
                    # Check for the required interaction terms based on working dimension
                    if args.TRAIN_WORKING_DIM == 1:
                        if 'x2*x3' in check_result['terms_present'] or 'x2x3' in check_result['terms_present']:
                            best_candidates_pool.append(candidate_)
                    elif args.TRAIN_WORKING_DIM == 2:
                        if 'x1*x3' in check_result['terms_present'] or 'x1x3' in check_result['terms_present']:
                            best_candidates_pool.append(candidate_)
                    elif args.TRAIN_WORKING_DIM == 3:
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
        summary_path = os.path.join(save_dir, f"best_candidates_pool_summary_{args.TRAIN_WORKING_DIM}.txt")
        # Write summary
        with open(summary_path, "w") as f:
            for idx, candidate_ in enumerate(best_candidates_pool):
                f.write(f"Candidate {idx + 1}: Score={candidate_.score:.6f}, Loss={candidate_.error:.6f}, Seq={candidate_.action}, Expr={candidate_.get_expression()}\n")

        print(f"[INFO] best_candidates_pool_summary saved to {summary_path}")
        logprint(f"[INFO] best_candidates_pool_summary saved to {summary_path}")
        # Use best candidate by default (or add user selection if needed)
        best_candidate = min(best_candidates_pool, key=lambda c: c.error)
        optimal_idx = best_candidate.action
        # Assert that the best candidate's model op_seq matches its action
        assert (best_candidate.model.op_seq == torch.tensor(best_candidate.action, device=best_candidate.model.op_seq.device)).all(), "Mismatch between best_candidate model and action!"
        logprint(f"Selected: Loss={best_candidate.error:.6f}, Expression={best_candidate.get_expression()}")
        print(f"Selected: Loss={best_candidate.error:.6f}, Expression={best_candidate.get_expression()}")

            
        logprint(f"[INFO] Now we need to fine-tune the FEX model")
        print(f"[INFO] Now we need to fine-tune the FEX model, please rerun the script with option 2.")
        exit()

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
        model = FEX(op_seqs, dim=dimension).to(DEVICE)
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
            


    