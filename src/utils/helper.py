import torch
import numpy as np
import torch.nn as nn
import logging
import math
import os
import re
import sympy as sp

def weights_init(m):
    if isinstance(m, torch.nn.Linear):  # or whatever layers you use
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Helper function for logging
def logprint(message):
    """Log message to both file and console."""
    logging.info(message)

def adjust_learning_rate(optimizer, epoch, start_lr, num_iter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * 0.5* (math.cos(math.pi*epoch /num_iter)+1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def check_allowed_terms(expression, dimension,model_name):
    """
    Check if expression contains only allowed terms for the given dimension.
                    
    Args:
        expression (str): The expression to check
        dimension (int): Dimension (1, 2, or 3)
                    
    Returns:
        dict: Dictionary with 'valid' (bool) and 'terms_present' (list of terms found)
    """
    if model_name == 'OU1d':
        allowed_terms = {
            1: ['x1']
        }
        # Check for disallowed terms (terms that should NOT be present)
        disallowed_terms = {
        1: ['x1**2', 'x1**3', 'x1**4', 'cos', 'sin','exp', '**2','**3','**4','**5','**6','**7','**8'],  # x1x2 and x1x3 are not allowed in dim 1
        }
        allowed_vars = ['x1']
    elif model_name == 'SIR':
        # Define allowed terms for each dimension
        allowed_terms = {
    1: ['x1', 'x2', 'x3', 'x2*x3', 'x2x3'],  # x1, x2, x3, x2x3, and constants
    2: ['x1', 'x2', 'x3', 'x1*x3', 'x1x3'],  # x1, x2, x3, x1x3, and constants
    3: ['x1', 'x2', 'x3', 'x1*x2', 'x1x2']   # x1, x2, x3, x1x2, and constants
    } 
        disallowed_terms = {
        1: ['x1**2', 'x1**3', 'x1**4', 'cos', 'sin','exp', '**2','**3','**4','**5','**6','**7','**8'],  # x1x2 and x1x3 are not allowed in dim 1
        2: ['x1**2', 'x1**3', 'x1**4', 'cos', 'sin','exp', '**2','**3','**4','**5','**6','**7','**8'],  # x1x2 and x1x3 are not allowed in dim 1
        3: ['x1**2', 'x1**3', 'x1**4', 'cos', 'sin','exp', '**2','**3','**4','**5','**6','**7','**8'],  # x1x2 and x1x3 are not allowed in dim 1
        }
        allowed_vars = ['x1', 'x2', 'x3']
      
    # Convert expression to lowercase for easier checking
    expr_lower = expression.lower()
                    
    
                    
    # Check if any disallowed terms are present
    for term in disallowed_terms[dimension]:
        if term in expr_lower:
            return {'valid': False, 'terms_present': []}
                    
    # Check which allowed terms are present
    terms_present = [term for term in allowed_terms[dimension] if term in expr_lower]
    
    # Check if at least one allowed term is present (excluding constants)
    
    has_allowed_var = any(var in expr_lower for var in allowed_vars)
                    
    return {'valid': has_allowed_var, 'terms_present': terms_present}

def get_sequence_from_candidate(file_path: str, candidate_num: int) -> list:
    """
    Extract the sequence (Seq=[...]) from a specific candidate in the given file.
    Args:
        file_path (str): Path to the file containing candidate information
        candidate_num (int): Candidate number to extract (1-based)
    Returns:
        list: Sequence as a list of integers (or empty list if not found)
    """
    import re
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line for the specified candidate
    candidate_line = None
    for line in lines:
        if line.startswith(f'Candidate {candidate_num}:'):
            candidate_line = line
            break
    
    if candidate_line:
        seq_match = re.search(r'Seq=\[([^\]]+)\]', candidate_line)
        if seq_match:
            seq_str = seq_match.group(1)
            seq_list = [int(x.strip()) for x in seq_str.split(',')]
            return seq_list
        else:
            print(f"[WARN] No sequence found in candidate {candidate_num} for {file_path}")
            return []
    else:
        print(f"[WARN] Candidate {candidate_num} not found in {file_path}")
        return []

def select_operator_sequence(file_path: str, dim: int) -> list:
    """
    Display unique candidates and let user select one for the given dimension
    Args:
        file_path (str): Path to the candidate file
        dim (int): Dimension number for display purposes
    Returns:
        list: Selected sequence as a list of integers (or None if failed)
    """
    import sys
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse all candidates with their expressions
    candidates = []
    for line in lines:
        if line.startswith('Candidate'):
            # Extract candidate number, score, loss, sequence, and expression
            candidate_match = re.search(r'Candidate (\d+): Score=([^,]+), Loss=([^,]+), Seq=\[([^\]]+)\], Expr=(.+)', line)
            if candidate_match:
                candidate_num = int(candidate_match.group(1))
                score = candidate_match.group(2)
                loss = candidate_match.group(3)
                seq_str = candidate_match.group(4)
                expr = candidate_match.group(5).strip()
                seq_list = [int(x.strip()) for x in seq_str.split(',')]
                candidates.append({
                    'num': candidate_num,
                    'score': score,
                    'loss': loss,
                    'seq': seq_list,
                    'expr': expr
                })
    
    if not candidates:
        print(f"[ERROR] No candidates found in {file_path}")
        return None
    
    # Filter to unique sequences only
    unique_candidates = []
    seen_sequences = set()
    
    for candidate in candidates:
        seq_tuple = tuple(candidate['seq'])
        if seq_tuple not in seen_sequences:
            seen_sequences.add(seq_tuple)
            unique_candidates.append(candidate)
    
    # Display unique candidates
    print(f"\n" + "="*80)
    print(f"Unique candidates for Dimension {dim} (showing {len(unique_candidates)} out of {len(candidates)} total):")
    print("="*80)
    
    for i, candidate in enumerate(unique_candidates, 1):
        print(f"Option {i}: Score={candidate['score']}, Loss={candidate['loss']}")
        print(f"  Sequence: {candidate['seq']}")
        print(f"  Expression: {candidate['expr']}")
        print("-" * 80)
    
    # Get user selection
    while True:
        try:
            selection = input(f"Select option for Dimension {dim} (1-{len(unique_candidates)}): ").strip()
            selection_idx = int(selection)
            if 1 <= selection_idx <= len(unique_candidates):
                selected_candidate = unique_candidates[selection_idx - 1]
                print(f"Selected for Dimension {dim}: Option {selection_idx}")
                print(f"  Sequence: {selected_candidate['seq']}")
                print(f"  Expression: {selected_candidate['expr']}")
                return selected_candidate['seq']
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(unique_candidates)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def extract_coefficients_from_expr(expr_str: str, dim: int) -> dict:
    """
    Extract coefficients from a simplified expression string.
    
    Args:
        expr_str (str): The simplified expression string (e.g., "x1**2 + 2*x1 + 1")
        dim (int): Dimension number (for variable names like x1, x2, x3)
    
    Returns:
        dict: Dictionary mapping term names to coefficient values
    """
    try:
        # Parse the expression using sympy
        expr = sp.sympify(expr_str)
        
        # Get all symbols in the expression
        symbols = list(expr.free_symbols)
        
        # Initialize coefficient dictionary
        coeffs = {}
        
        # If expression is a polynomial, use as_poly to get coefficients
        try:
            # Try to convert to polynomial form
            if len(symbols) == 1:
                # Single variable case
                var = symbols[0]
                poly = sp.Poly(expr, var)
                coeff_dict = poly.as_dict()
                
                for (power,), coeff in coeff_dict.items():
                    if power == 0:
                        term_name = 'constant'
                    elif power == 1:
                        term_name = str(var)
                    else:
                        term_name = f"{var}**{power}"
                    coeffs[term_name] = float(coeff)
            else:
                # Multi-variable case - extract terms manually
                if expr.is_Add:
                    # Expression is a sum of terms
                    for term in expr.args:
                        if term.is_Mul:
                            # Term is a product (e.g., 2*x1*x2)
                            coeff = 1.0
                            var_part = []
                            for factor in term.args:
                                if factor.is_Number:
                                    coeff = float(factor)
                                else:
                                    var_part.append(str(factor))
                            term_name = '*'.join(var_part) if var_part else 'constant'
                            coeffs[term_name] = coeff
                        elif term.is_Number:
                            # Constant term
                            coeffs['constant'] = float(term)
                        else:
                            # Single variable term (e.g., x1)
                            coeffs[str(term)] = 1.0
                elif expr.is_Mul:
                    # Expression is a product
                    coeff = 1.0
                    var_part = []
                    for factor in expr.args:
                        if factor.is_Number:
                            coeff = float(factor)
                        else:
                            var_part.append(str(factor))
                    term_name = '*'.join(var_part) if var_part else 'constant'
                    coeffs[term_name] = coeff
                else:
                    # Single term
                    coeffs[str(expr)] = 1.0
        except:
            # Fallback: just return the expression as a single term
            coeffs[str(expr)] = 1.0
        
        return coeffs
    except Exception as e:
        # If parsing fails, return empty dict
        return {}

def save_parameters(ZT_Solution, ODE_Solution, second_stage_dir, args, device):
    # Check for infinite values in yTrain
    is_finite_ODETrain = np.isfinite(ODE_Solution) & ~np.isnan(ODE_Solution)
    print(f"[INFO] Finite values check: {np.sum(is_finite_ODETrain)}/{ODE_Solution.size} values are finite")
    
    # Shuffle the data
    ZT_Train_filtered = ZT_Solution[is_finite_ODETrain.all(axis=1)]
    ODE_Train_filtered = ODE_Solution[is_finite_ODETrain.all(axis=1)]
    train_size_new = ZT_Train_filtered.shape[0]
    
    indices = np.random.permutation(train_size_new)
    ZT_Train_filtered = ZT_Train_filtered[indices]
    ODE_Train_filtered = ODE_Train_filtered[indices]

    # Normalize the data
    ZT_Train_mean = np.mean(ZT_Train_filtered, axis=0, keepdims=True)
    ZT_Train_std = np.std(ZT_Train_filtered, axis=0, keepdims=True)
    ODE_Train_mean = np.mean(ODE_Train_filtered, axis=0, keepdims=True)
    ODE_Train_std = np.std(ODE_Train_filtered, axis=0, keepdims=True)
    # Convert data to tensors
    ZT_Train_new = (ZT_Train_filtered - ZT_Train_mean) / ZT_Train_std
    ODE_Train_new = (ODE_Train_filtered - ODE_Train_mean) / ODE_Train_std
    
    ZT_Train_new = torch.tensor(ZT_Train_new, dtype=torch.float32).to(device)
    ODE_Train_new = torch.tensor(ODE_Train_new, dtype=torch.float32).to(device)
    ZT_Train_mean = torch.tensor(ZT_Train_mean, dtype=torch.float32).to(device)
    ZT_Train_std = torch.tensor(ZT_Train_std, dtype=torch.float32).to(device)
    ODE_Train_mean = torch.tensor(ODE_Train_mean, dtype=torch.float32).to(device)
    ODE_Train_std = torch.tensor(ODE_Train_std, dtype=torch.float32).to(device)
    # Save normalization parameters
    dataname2 = os.path.join(second_stage_dir, 'data_inf.pt')
    # Check if this is TF_CDM based on directory path
    if 'TF_CDM' in second_stage_dir or 'All_stage_TF_CDM' in second_stage_dir:
        diff_scale = 10  # TF_CDM uses diff_scale = 10
    else:
        diff_scale = args.DIFF_SCALE  # Get diff_scale from args for FEX-DM
    torch.save({
        'ZT_Train_new': ZT_Train_new,
        'ODE_Train_new': ODE_Train_new,
        'ZT_Train_mean': ZT_Train_mean,
        'ZT_Train_std': ZT_Train_std,
        'ODE_Train_mean': ODE_Train_mean,
        'ODE_Train_std': ODE_Train_std,
        'diff_scale': diff_scale
    }, dataname2)
    print(f'[INFO] Normalization parameters saved to {dataname2}')
    print('ZT_Train_mean:', ZT_Train_mean)
    print('ZT_Train_std:', ZT_Train_std)
    print('ODE_Train_mean:', ODE_Train_mean)
    print('ODE_Train_std:', ODE_Train_std)