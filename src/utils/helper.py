import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
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
        model_name (str): Model name ('OU1d', 'Trigonometric1d', 'SIR', etc.)
                    
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
    elif model_name == 'Trigonometric1d':
        allowed_terms = {
            1: ['x1', 'cos', 'sin']  # Allow trigonometric functions
        }
        # For Trigonometric1d, we allow cos and sin (which approximate sin(2πx))
        # Disallow high powers and exp, but allow trigonometric functions
        disallowed_terms = {
        1: ['x1**2', 'x1**3', 'x1**4', 'exp', '**2','**3','**4','**5','**6','**7','**8'],  # Allow cos and sin
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
    
    # For Trigonometric1d, also check if expression contains trigonometric functions
    # Expressions like "-1.1989 cos(6.2476*x1 - 4.6837) - 0.0104" are acceptable
    # as they approximate sin(2πx)
    # IMPORTANT: Only ONE trigonometric function (either sin OR cos, not both) and only ONE instance
    # Also: x1 must ONLY appear inside sin() or cos(), not as a standalone term
    if model_name == 'Trigonometric1d':
        has_cos = 'cos' in expr_lower
        has_sin = 'sin' in expr_lower
        
        # Count occurrences of cos and sin
        cos_count = expr_lower.count('cos')
        sin_count = expr_lower.count('sin')
        
        # Check: must have exactly one trigonometric function (either sin OR cos, not both)
        # and only one instance of that function
        has_exactly_one_trig = False
        if has_cos and not has_sin:
            # Only cos, check it appears exactly once
            has_exactly_one_trig = (cos_count == 1)
        elif has_sin and not has_cos:
            # Only sin, check it appears exactly once
            has_exactly_one_trig = (sin_count == 1)
        # If both are present or neither is present, it's invalid
        
        # Check that x1 only appears inside sin() or cos(), not as a standalone term
        # We need to check if x1 appears outside of trigonometric functions
        # Strategy: remove all sin(...) and cos(...) patterns, then check if x1 remains
        # Use a more robust method to handle nested parentheses
        def remove_balanced_parens(text, func_name):
            """Remove function calls with balanced parentheses"""
            result = text
            while True:
                # Find the function name
                pattern = rf'{func_name}\s*\('
                match = re.search(pattern, result)
                if not match:
                    break
                # Find the matching closing parenthesis
                start = match.end() - 1  # Position of opening (
                depth = 0
                end = start
                for i in range(start, len(result)):
                    if result[i] == '(':
                        depth += 1
                    elif result[i] == ')':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                # Remove the function call
                result = result[:match.start()] + ' ' + result[end:]
            return result
        
        # Remove all sin(...) and cos(...) patterns
        expr_without_trig = remove_balanced_parens(expr_lower, 'sin')
        expr_without_trig = remove_balanced_parens(expr_without_trig, 'cos')
        
        # After removing all sin(...) and cos(...), check if x1 still appears
        # This means x1 appears outside of trigonometric functions
        # Also check for patterns like *x1, x1*, +x1, x1+, -x1, x1-, or standalone x1
        x1_outside_trig = False
        if 'x1' in expr_without_trig:
            # Check if x1 appears in a way that suggests it's a standalone term
            # Patterns: *x1, x1*, +x1, x1+, -x1, x1-, or x1 at start/end
            x1_patterns = [
                r'\*x1', r'x1\*',  # multiplication
                r'\+x1', r'x1\+',  # addition
                r'-x1', r'x1-',    # subtraction (but -x1 could be negative, so be careful)
                r'^x1', r'x1$',    # at start or end
                r'\sx1\s',         # standalone with spaces
            ]
            for pattern in x1_patterns:
                if re.search(pattern, expr_without_trig):
                    x1_outside_trig = True
                    break
        
        # Also check that x1 appears in a more complex expression inside the trig function
        # Reject simple cases like sin(x1) or cos(x1) - need something like sin(2*pi*x1) or cos(6.2476*x1 - 4.6837)
        x1_in_complex_trig = False
        def extract_trig_argument(text, func_name):
            """Extract the argument of a trigonometric function with balanced parentheses"""
            pattern = rf'{func_name}\s*\('
            match = re.search(pattern, text)
            if not match:
                return None
            start = match.end() - 1  # Position of opening (
            depth = 0
            arg_start = start + 1
            for i in range(start, len(text)):
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return text[arg_start:i]
            return None
        
        if has_cos:
            cos_arg = extract_trig_argument(expr_lower, 'cos')
            if cos_arg and 'x1' in cos_arg:
                # Check if x1 appears with multiplication, addition, or subtraction (not just cos(x1))
                # Also check that it's not just a simple pattern like "x1" alone
                cos_arg_clean = cos_arg.strip()
                if cos_arg_clean != 'x1':
                    # Check if it has operators (multiplication, addition, subtraction)
                    if '*' in cos_arg or '+' in cos_arg or '-' in cos_arg:
                        x1_in_complex_trig = True
        elif has_sin:
            sin_arg = extract_trig_argument(expr_lower, 'sin')
            if sin_arg and 'x1' in sin_arg:
                # Check if x1 appears with multiplication, addition, or subtraction (not just sin(x1))
                # Also check that it's not just a simple pattern like "x1" alone
                sin_arg_clean = sin_arg.strip()
                if sin_arg_clean != 'x1':
                    # Check if it has operators (multiplication, addition, subtraction)
                    if '*' in sin_arg or '+' in sin_arg or '-' in sin_arg:
                        x1_in_complex_trig = True
        
        # Allow constants both before and after the trigonometric function
        # Accept expressions like "C - A*cos(...)", "A*cos(...) - C", "A*cos(...)", etc.
        # The only thing we reject is formatting bugs like ")056"
        has_standalone_constants = False
        
        # Check for formatting bugs like ")056" - digits directly after closing parenthesis
        # This is a formatting issue, not a valid constant
        trig_func = 'cos' if has_cos else 'sin'
        trig_pattern = rf'{trig_func}\s*\('
        trig_match = re.search(trig_pattern, expr_lower)
        if trig_match:
            # Find the matching closing parenthesis
            start = trig_match.end() - 1  # Position of opening (
            depth = 0
            end = start
            for i in range(start, len(expr_lower)):
                if expr_lower[i] == '(':
                    depth += 1
                elif expr_lower[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            # Check what comes AFTER the closing parenthesis for formatting bugs
            after_trig = expr_lower[end:].strip()
            if after_trig:
                # Check for formatting bugs like ")056" - digits directly after closing parenthesis
                # Remove operators and whitespace, check if digits remain
                after_cleaned = re.sub(r'[+\-*/\s]', '', after_trig)
                if after_cleaned and re.search(r'^\d', after_cleaned):
                    # There are digits at the start (like "056") - formatting bug
                    # But allow if it's a proper constant like "+ 0.1355" or "- 0.1355"
                    if not re.match(r'^\s*[+-]\s*\d+\.?\d+', after_trig):
                        has_standalone_constants = True
        
        # Expression is valid if:
        # 1. It has x1
        # 2. It has exactly one trigonometric function (one instance)
        # 3. x1 only appears inside the trigonometric function (not as standalone term)
        # 4. x1 appears in a complex expression inside the trig function (not just sin(x1) or cos(x1))
        # 5. No standalone constants added/subtracted outside the trigonometric function
        is_valid = has_allowed_var and has_exactly_one_trig and not x1_outside_trig and x1_in_complex_trig and not has_standalone_constants
        return {'valid': is_valid, 'terms_present': terms_present}
                    
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



class VAE(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, latent_dim=1):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # **Encoder layers**
        self.input = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of latent space
        self.output_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent space
        
        # **Decoder layers**
        self.dec_input = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
        
        # **Store best weights**
        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_mu_weight = torch.clone(self.output_mu.weight.data)
        self.best_output_mu_bias = torch.clone(self.output_mu.bias.data)
        self.best_output_logvar_weight = torch.clone(self.output_logvar.weight.data)
        self.best_output_logvar_bias = torch.clone(self.output_logvar.bias.data)
        self.best_dec_input_weight = torch.clone(self.dec_input.weight.data)
        self.best_dec_input_bias = torch.clone(self.dec_input.bias.data)
        self.best_fc2_weight = torch.clone(self.fc2.weight.data)
        self.best_fc2_bias = torch.clone(self.fc2.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def encoder(self, x):
        h = torch.tanh(self.input(x))
        h = torch.tanh(self.fc1(h))
        mu = self.output_mu(h)
        logvar = self.output_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h = torch.tanh(self.dec_input(z))
        h = torch.tanh(self.fc2(h))
        return self.output(h)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def update_best(self):
        """ Save the current best model weights """
        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_mu_weight = torch.clone(self.output_mu.weight.data)
        self.best_output_mu_bias = torch.clone(self.output_mu.bias.data)
        self.best_output_logvar_weight = torch.clone(self.output_logvar.weight.data)
        self.best_output_logvar_bias = torch.clone(self.output_logvar.bias.data)
        self.best_dec_input_weight = torch.clone(self.dec_input.weight.data)
        self.best_dec_input_bias = torch.clone(self.dec_input.bias.data)
        self.best_fc2_weight = torch.clone(self.fc2.weight.data)
        self.best_fc2_bias = torch.clone(self.fc2.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):
        """ Restore the best model weights """
        self.input.weight.data = self.best_input_weight
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.output_mu.weight.data = self.best_output_mu_weight
        self.output_mu.bias.data = self.best_output_mu_bias
        self.output_logvar.weight.data = self.best_output_logvar_weight
        self.output_logvar.bias.data = self.best_output_logvar_bias
        self.dec_input.weight.data = self.best_dec_input_weight
        self.dec_input.bias.data = self.best_dec_input_bias
        self.fc2.weight.data = self.best_fc2_weight
        self.fc2.bias.data = self.best_fc2_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias


def train_FN_time_dependent(ODE_Solution: np.ndarray,
                             ZT_Solution: np.ndarray,
                             dim: int = 3,
                             device: str = 'cpu',
                             learning_rate: float = 0.001,
                             n_iter: int = 5000,
                             best_valid_err: float = 5.0,
                             save_dir: str = None,
                             num_time_points: int = None,
                             time_range: tuple = None,
                             dt: float = 0.01):
    """
    Train a multi-output neural network (dim→dim) for time-dependent case.
    This preserves correlation structure between dimensions.
    Loops through time steps and trains a separate model for each time step.
    
    Args:
        ODE_Solution: 3D array of shape (size, dim, time_steps) - ODE solutions
        ZT_Solution: 3D array of shape (size, dim, time_steps) - Wiener increments
        dim: Number of dimensions
        device: Device to use ('cpu' or 'cuda')
        learning_rate: Learning rate for optimizer
        n_iter: Number of training iterations
        best_valid_err: Best validation error threshold
        save_dir: Directory to save the model
        num_time_points: Number of time points to train on (None = all)
        time_range: Tuple (start_idx, end_idx) for specific time range
        dt: Time step size
    """
    from .ODEParser import FN_Net, select_time_points
    
    total_time_steps = ODE_Solution.shape[2]
    size = ODE_Solution.shape[0]
    
    # Select time points to train on
    if time_range is not None:
        # Use specific time range
        start_idx, end_idx = time_range
        time_indices = range(start_idx, min(end_idx, total_time_steps))
        selected_times = np.array([t * dt for t in time_indices])
        print(f"Training {dim}→{dim} network on time range {start_idx}-{min(end_idx, total_time_steps)} ({len(time_indices)} time points)")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    elif num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Training {dim}→{dim} network on {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
    
    for t_idx, t in enumerate(time_indices):
        print(f'Training {dim}→{dim} network for {t_idx+1}/{len(time_indices)} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        
        # Check if model already exists
        if save_dir is not None:
            FN_path = os.path.join(save_dir, f'FN_3to3_t{t}.pth')
            norm_path = os.path.join(save_dir, f'norm_params_3to3_t{t}.npy')
            
            if os.path.exists(FN_path) and os.path.exists(norm_path):
                print(f'[INFO] {dim}→{dim} model for time step {t} already exists. Skipping...')
                continue
        
        NTrain = int(size * 0.8)
        
        # Create dim→dim neural network
        FN_multi = FN_Net(dim, dim, 100).to(device)  # dim inputs, dim outputs
        FN_multi.zero_grad()
        optimizer = optim.Adam(FN_multi.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
        criterion = nn.MSELoss()
        
        # Prepare data for ALL dimensions at once
        # Input: All dim Wiener increments [W1, W2, ..., Wdim]
        xTrain_normal = torch.tensor(ZT_Solution[0:NTrain, :, t], dtype=torch.float32).to(device)  # (N, dim)
        # Output: All dim ODE solutions [dim1, dim2, ..., dimdim]
        yTrain_normal = torch.tensor(ODE_Solution[0:NTrain, :, t], dtype=torch.float32).to(device)  # (N, dim)
        
        # Validation data
        xValid_normal = torch.tensor(ZT_Solution[NTrain:size, :, t], dtype=torch.float32).to(device)  # (N, dim)
        yValid_normal = torch.tensor(ODE_Solution[NTrain:size, :, t], dtype=torch.float32).to(device)  # (N, dim)
        
        # Calculate normalization parameters for all dimensions
        y_mean = np.mean(ODE_Solution[0:NTrain, :, t], axis=0)  # (dim,)
        y_std = np.std(ODE_Solution[0:NTrain, :, t], axis=0)     # (dim,)
        
        # Normalize the data
        yTrain_normal = (yTrain_normal - torch.tensor(y_mean, dtype=torch.float32).to(device)) / torch.tensor(y_std, dtype=torch.float32).to(device)
        yValid_normal = (yValid_normal - torch.tensor(y_mean, dtype=torch.float32).to(device)) / torch.tensor(y_std, dtype=torch.float32).to(device)
        
        best_valid_loss = float('inf')
        patience_counter = 0
        patience_limit = 500  # Early stopping patience
        
        print(f'[INFO] Training {dim}→{dim} network for time step {t}...')
        print(f'[INFO] Input shape: {xTrain_normal.shape}, Output shape: {yTrain_normal.shape}')
        
        for it in range(n_iter):
            optimizer.zero_grad()
            
            # Forward pass: predict all dimensions together
            pred = FN_multi(xTrain_normal)  # (N, dim)
            loss = criterion(pred, yTrain_normal)
            loss.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_valid = FN_multi(xValid_normal)
                valid_loss = criterion(pred_valid, yValid_normal)
            
            # Learning rate scheduling
            scheduler.step(valid_loss.item())
            
            # Early stopping
            if valid_loss.item() < best_valid_loss:
                best_valid_loss = valid_loss.item()
                FN_multi.update_best()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f"Early stopping at iteration {it}")
                break
            
            if it % 500 == 0:
                print(f'[INFO] Epoch {it+1}/{n_iter}; Train Loss: {loss.item():.6f}; Valid Loss: {valid_loss.item():.6f}')
        
        # Load best model
        FN_multi.final_update()
        
        # Save the dim→dim network
        if save_dir is not None:
            FN_path = os.path.join(save_dir, f'FN_3to3_t{t}.pth')
            # Save model parameters in CPU format regardless of training device
            state_dict_cpu = {k: v.cpu() for k, v in FN_multi.state_dict().items()}
            torch.save(state_dict_cpu, FN_path)
            print(f'[SAVE] Saved {dim}→{dim} model to: {FN_path}')
            
            # Save normalization parameters for all dimensions
            norm_params = {'mean': y_mean, 'std': y_std}
            norm_path = os.path.join(save_dir, f'norm_params_3to3_t{t}.npy')
            np.save(norm_path, norm_params)
            print(f'[SAVE] Saved normalization params to: {norm_path}')
        else:
            print(f'[WARNING] save_dir is None, not saving {dim}→{dim} model for t{t}')
    
    print(f"[INFO] {dim}→{dim} neural network training completed for time-dependent case!")


def train_TF_CDM_time_dependent(ODE_Solution: np.ndarray,
                                ZT_Solution: np.ndarray,
                                current_state: np.ndarray,
                                dim: int = 3,
                                device: str = 'cpu',
                                learning_rate: float = 0.001,
                                n_iter: int = 5000,
                                best_valid_err: float = 5.0,
                                save_dir: str = None,
                                num_time_points: int = None,
                                time_range: tuple = None,
                                dt: float = 0.01):
    """
    Train TF-CDM models (2*dim→dim) for time-dependent case.
    TF-CDM takes (current_state, z) as input, so input_dim = 2*dim.
    Loops through time steps and trains a separate model for each time step.
    
    Args:
        ODE_Solution: 3D array of shape (size, dim, time_steps) - ODE solutions
        ZT_Solution: 3D array of shape (size, dim, time_steps) - Wiener increments
        current_state: 3D array of shape (size, dim, time_steps) - current states
        dim: Number of dimensions
        device: Device to use ('cpu' or 'cuda')
        learning_rate: Learning rate for optimizer
        n_iter: Number of training iterations
        best_valid_err: Best validation error threshold
        save_dir: Directory to save the model
        num_time_points: Number of time points to train on (None = all)
        time_range: Tuple (start_idx, end_idx) for specific time range
        dt: Time step size
    """
    from .ODEParser import FN_Net, select_time_points
    
    total_time_steps = ODE_Solution.shape[2]
    size = ODE_Solution.shape[0]
    
    # Select time points to train on
    if time_range is not None:
        start_idx, end_idx = time_range
        time_indices = range(start_idx, min(end_idx, total_time_steps))
        selected_times = np.array([t * dt for t in time_indices])
        print(f"Training TF-CDM (2*{dim}→{dim}) network on time range {start_idx}-{min(end_idx, total_time_steps)} ({len(time_indices)} time points)")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    elif num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Training TF-CDM (2*{dim}→{dim}) network on {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
    
    for t_idx, t in enumerate(time_indices):
        print(f'Training TF-CDM (2*{dim}→{dim}) network for {t_idx+1}/{len(time_indices)} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        
        # Check if model already exists
        if save_dir is not None:
            FN_path = os.path.join(save_dir, f'FN_TF_CDM_t{t}.pth')
            norm_path = os.path.join(save_dir, f'norm_params_TF_CDM_t{t}.npy')
            
            if os.path.exists(FN_path) and os.path.exists(norm_path):
                print(f'[INFO] TF-CDM model for time step {t} already exists. Skipping...')
                continue
        
        NTrain = int(size * 0.8)
        
        # Create 2*dim→dim neural network (TF-CDM takes (current_state, z) as input)
        FN_TF_CDM = FN_Net(2 * dim, dim, 100).to(device)
        FN_TF_CDM.zero_grad()
        optimizer = optim.Adam(FN_TF_CDM.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
        criterion = nn.MSELoss()
        
        # Prepare data: concatenate current_state and ZT_Solution
        # Input: (current_state, z) -> (N, 2*dim)
        current_state_t = current_state[0:NTrain, :, t]  # (N, dim)
        z_t = ZT_Solution[0:NTrain, :, t]  # (N, dim)
        xTrain_normal = np.hstack((current_state_t, z_t))  # (N, 2*dim)
        xTrain_normal = torch.tensor(xTrain_normal, dtype=torch.float32).to(device)
        
        # Output: ODE solutions
        yTrain_normal = torch.tensor(ODE_Solution[0:NTrain, :, t], dtype=torch.float32).to(device)  # (N, dim)
        
        # Validation data
        current_state_t_valid = current_state[NTrain:size, :, t]
        z_t_valid = ZT_Solution[NTrain:size, :, t]
        xValid_normal = np.hstack((current_state_t_valid, z_t_valid))
        xValid_normal = torch.tensor(xValid_normal, dtype=torch.float32).to(device)
        yValid_normal = torch.tensor(ODE_Solution[NTrain:size, :, t], dtype=torch.float32).to(device)
        
        # Calculate normalization parameters
        x_mean = np.mean(xTrain_normal.cpu().numpy(), axis=0)  # (2*dim,)
        x_std = np.std(xTrain_normal.cpu().numpy(), axis=0)  # (2*dim,)
        y_mean = np.mean(ODE_Solution[0:NTrain, :, t], axis=0)  # (dim,)
        y_std = np.std(ODE_Solution[0:NTrain, :, t], axis=0)  # (dim,)
        
        # Normalize the data
        xTrain_normal = (xTrain_normal - torch.tensor(x_mean, dtype=torch.float32).to(device)) / torch.tensor(x_std, dtype=torch.float32).to(device)
        xValid_normal = (xValid_normal - torch.tensor(x_mean, dtype=torch.float32).to(device)) / torch.tensor(x_std, dtype=torch.float32).to(device)
        yTrain_normal = (yTrain_normal - torch.tensor(y_mean, dtype=torch.float32).to(device)) / torch.tensor(y_std, dtype=torch.float32).to(device)
        yValid_normal = (yValid_normal - torch.tensor(y_mean, dtype=torch.float32).to(device)) / torch.tensor(y_std, dtype=torch.float32).to(device)
        
        best_valid_loss = float('inf')
        patience_counter = 0
        patience_limit = 500
        
        print(f'[INFO] Training TF-CDM (2*{dim}→{dim}) network for time step {t}...')
        print(f'[INFO] Input shape: {xTrain_normal.shape}, Output shape: {yTrain_normal.shape}')
        
        for it in range(n_iter):
            optimizer.zero_grad()
            
            pred = FN_TF_CDM(xTrain_normal)  # (N, dim)
            loss = criterion(pred, yTrain_normal)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred_valid = FN_TF_CDM(xValid_normal)
                valid_loss = criterion(pred_valid, yValid_normal)
            
            scheduler.step(valid_loss.item())
            
            if valid_loss.item() < best_valid_loss:
                best_valid_loss = valid_loss.item()
                FN_TF_CDM.update_best()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f"Early stopping at iteration {it}")
                break
            
            if it % 500 == 0:
                print(f'[INFO] Epoch {it+1}/{n_iter}; Train Loss: {loss.item():.6f}; Valid Loss: {valid_loss.item():.6f}')
        
        FN_TF_CDM.final_update()
        
        # Save model
        if save_dir is not None:
            FN_path = os.path.join(save_dir, f'FN_TF_CDM_t{t}.pth')
            state_dict_cpu = {k: v.cpu() for k, v in FN_TF_CDM.state_dict().items()}
            torch.save(state_dict_cpu, FN_path)
            print(f'[SAVE] Saved TF-CDM model to: {FN_path}')
            
            # Save normalization parameters
            norm_params = {'x_mean': x_mean, 'x_std': x_std, 'mean': y_mean, 'std': y_std}
            norm_path = os.path.join(save_dir, f'norm_params_TF_CDM_t{t}.npy')
            np.save(norm_path, norm_params)
            print(f'[SAVE] Saved normalization params to: {norm_path}')
        else:
            print(f'[WARNING] save_dir is None, not saving TF-CDM model for t{t}')
    
    print(f"[INFO] TF-CDM (2*{dim}→{dim}) neural network training completed for time-dependent case!")


def load_time_dependent_models(save_dir, dim, device='cpu', total_time_steps=None):
    """
    Load all time-dependent models (FN_3to3_t{t}.pth) and normalization parameters.
    
    Args:
        save_dir: Directory containing the models
        dim: Number of dimensions
        device: Device to load models on
        total_time_steps: Total number of time steps (if None, will scan directory)
    
    Returns:
        models_dict: Dictionary mapping time step t to (model, norm_params) tuple
    """
    from .ODEParser import FN_Net
    import glob
    
    models_dict = {}
    
    # Find all model files
    pattern = os.path.join(save_dir, 'FN_3to3_t*.pth')
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f'[WARNING] No time-dependent models found in {save_dir}')
        return models_dict
    
    # Extract time steps from filenames
    time_steps = []
    for f in model_files:
        try:
            # Extract time step from filename like "FN_3to3_t42.pth"
            basename = os.path.basename(f)
            t_str = basename.replace('FN_3to3_t', '').replace('.pth', '')
            time_steps.append(int(t_str))
        except:
            continue
    
    time_steps = sorted(time_steps)
    print(f'[INFO] Found {len(time_steps)} time-dependent models for time steps: {time_steps[:5]}...{time_steps[-5:] if len(time_steps) > 10 else time_steps}')
    
    # Load each model
    for t in time_steps:
        FN_path = os.path.join(save_dir, f'FN_3to3_t{t}.pth')
        norm_path = os.path.join(save_dir, f'norm_params_3to3_t{t}.npy')
        
        if not os.path.exists(FN_path) or not os.path.exists(norm_path):
            print(f'[WARNING] Missing model or norm file for time step {t}, skipping...')
            continue
        
        # Load model
        FN_multi = FN_Net(dim, dim, 100).to(device)
        state_dict = torch.load(FN_path, map_location=device)
        FN_multi.load_state_dict(state_dict)
        FN_multi.eval()
        
        # Load normalization parameters
        norm_params = np.load(norm_path, allow_pickle=True).item()
        
        models_dict[t] = (FN_multi, norm_params)
    
    print(f'[INFO] Successfully loaded {len(models_dict)} time-dependent models')
    return models_dict


def load_time_dependent_TF_CDM_models(save_dir, dim, device='cpu', total_time_steps=None):
    """
    Load all time-dependent TF-CDM models (FN_TF_CDM_t{t}.pth) and normalization parameters.
    
    Args:
        save_dir: Directory containing the models
        dim: Number of dimensions
        device: Device to load models on
        total_time_steps: Total number of time steps (if None, will scan directory)
    
    Returns:
        models_dict: Dictionary mapping time step t to (model, norm_params) tuple
    """
    from .ODEParser import FN_Net
    import glob
    
    models_dict = {}
    
    # Find all model files
    pattern = os.path.join(save_dir, 'FN_TF_CDM_t*.pth')
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f'[WARNING] No time-dependent TF-CDM models found in {save_dir}')
        return models_dict
    
    # Extract time steps from filenames
    time_steps = []
    for f in model_files:
        try:
            # Extract time step from filename like "FN_TF_CDM_t42.pth"
            basename = os.path.basename(f)
            t_str = basename.replace('FN_TF_CDM_t', '').replace('.pth', '')
            time_steps.append(int(t_str))
        except:
            continue
    
    time_steps = sorted(time_steps)
    print(f'[INFO] Found {len(time_steps)} time-dependent TF-CDM models for time steps: {time_steps[:5]}...{time_steps[-5:] if len(time_steps) > 10 else time_steps}')
    
    # Load each model
    for t in time_steps:
        FN_path = os.path.join(save_dir, f'FN_TF_CDM_t{t}.pth')
        norm_path = os.path.join(save_dir, f'norm_params_TF_CDM_t{t}.npy')
        
        if not os.path.exists(FN_path) or not os.path.exists(norm_path):
            print(f'[WARNING] Missing TF-CDM model or norm file for time step {t}, skipping...')
            continue
        
        # Load model (TF-CDM: 2*dim inputs, dim outputs)
        FN_TF_CDM = FN_Net(2 * dim, dim, 100).to(device)
        state_dict = torch.load(FN_path, map_location=device)
        FN_TF_CDM.load_state_dict(state_dict)
        FN_TF_CDM.eval()
        
        # Load normalization parameters
        norm_params = np.load(norm_path, allow_pickle=True).item()
        
        models_dict[t] = (FN_TF_CDM, norm_params)
    
    print(f'[INFO] Successfully loaded {len(models_dict)} time-dependent TF-CDM models')
    return models_dict


def train_VAE_time_dependent(residuals: np.ndarray,
                              dim: int = 3,
                              device: str = 'cpu',
                              learning_rate: float = 0.001,
                              n_iter: int = 5000,
                              save_dir: str = None,
                              num_time_points: int = None,
                              time_range: tuple = None,
                              dt: float = 0.01):
    """
    Train VAE models for time-dependent case.
    Loops through time steps and trains a separate VAE model for each time step.
    
    Args:
        residuals: 3D array of shape (size, dim, time_steps) - residuals scaled by DIFF_SCALE
        dim: Number of dimensions
        device: Device to use ('cpu' or 'cuda')
        learning_rate: Learning rate for optimizer
        n_iter: Number of training iterations
        save_dir: Directory to save the model
        num_time_points: Number of time points to train on (None = all)
        time_range: Tuple (start_idx, end_idx) for specific time range
        dt: Time step size
    """
    from .ODEParser import select_time_points
    import torch.nn.functional as F
    
    total_time_steps = residuals.shape[2]
    size = residuals.shape[0]
    
    # Select time points to train on
    if time_range is not None:
        start_idx, end_idx = time_range
        time_indices = range(start_idx, min(end_idx, total_time_steps))
        selected_times = np.array([t * dt for t in time_indices])
        print(f"Training VAE network on time range {start_idx}-{min(end_idx, total_time_steps)} ({len(time_indices)} time points)")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    elif num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Training VAE network on {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
    
    # Define VAE loss function
    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div
    
    for t_idx, t in enumerate(time_indices):
        print(f'Training VAE network for {t_idx+1}/{len(time_indices)} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        
        # Check if model already exists
        if save_dir is not None:
            VAE_path = os.path.join(save_dir, f'VAE_FEX_t{t}.pth')
            
            if os.path.exists(VAE_path):
                print(f'[INFO] VAE model for time step {t} already exists. Skipping...')
                continue
        
        NTrain = int(size * 0.8)
        
        # Create VAE model
        VAE_model = VAE(input_dim=dim, hidden_dim=50, latent_dim=dim).to(device)
        VAE_model.zero_grad()
        optimizer = optim.Adam(VAE_model.parameters(), lr=learning_rate, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
        
        # Prepare data for current time step
        # Input: residuals at time step t
        residuals_t = residuals[0:NTrain, :, t]  # (N, dim)
        residuals_tensor = torch.tensor(residuals_t, dtype=torch.float32).to(device)
        
        # Validation data
        residuals_valid_t = residuals[NTrain:size, :, t]
        residuals_valid_tensor = torch.tensor(residuals_valid_t, dtype=torch.float32).to(device)
        
        best_valid_loss = float('inf')
        patience_counter = 0
        patience_limit = 500
        
        print(f'[INFO] Training VAE network for time step {t}...')
        print(f'[INFO] Input shape: {residuals_tensor.shape}')
        
        for it in range(n_iter):
            optimizer.zero_grad()
            
            recon_x, mu, logvar = VAE_model(residuals_tensor)
            loss = vae_loss(recon_x, residuals_tensor, mu, logvar)
            loss.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                recon_x_valid, mu_valid, logvar_valid = VAE_model(residuals_valid_tensor)
                valid_loss = vae_loss(recon_x_valid, residuals_valid_tensor, mu_valid, logvar_valid)
            
            scheduler.step(valid_loss.item())
            
            if valid_loss.item() < best_valid_loss:
                best_valid_loss = valid_loss.item()
                VAE_model.update_best()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f"Early stopping at iteration {it}")
                break
            
            if it % 500 == 0:
                print(f'[INFO] Epoch {it+1}/{n_iter}; Train Loss: {loss.item():.6f}; Valid Loss: {valid_loss.item():.6f}')
        
        VAE_model.final_update()
        
        # Save model
        if save_dir is not None:
            VAE_path = os.path.join(save_dir, f'VAE_FEX_t{t}.pth')
            state_dict_cpu = {k: v.cpu() for k, v in VAE_model.state_dict().items()}
            torch.save(state_dict_cpu, VAE_path)
            print(f'[SAVE] Saved VAE model to: {VAE_path}')
        else:
            print(f'[WARNING] save_dir is None, not saving VAE model for t{t}')
    
    print(f"[INFO] VAE neural network training completed for time-dependent case!")


def train_FEX_NN_time_dependent(residuals: np.ndarray,
                                 current_state: np.ndarray,
                                 dim: int = 3,
                                 device: str = 'cpu',
                                 learning_rate: float = 0.001,
                                 n_iter: int = 5000,
                                 save_dir: str = None,
                                 num_time_points: int = None,
                                 time_range: tuple = None,
                                 dt: float = 0.01):
    """
    Train FEX-NN (CovarianceNet) models for time-dependent case.
    Loops through time steps and trains a separate FEX-NN model for each time step.
    
    Args:
        residuals: 3D array of shape (size, dim, time_steps) - residuals
        current_state: 3D array of shape (size, dim, time_steps) - current states
        dim: Number of dimensions
        device: Device to use ('cpu' or 'cuda')
        learning_rate: Learning rate for optimizer
        n_iter: Number of training iterations
        save_dir: Directory to save the model
        num_time_points: Number of time points to train on (None = all)
        time_range: Tuple (start_idx, end_idx) for specific time range
        dt: Time step size
    """
    from .ODEParser import CovarianceNet, select_time_points
    
    total_time_steps = residuals.shape[2]
    size = residuals.shape[0]
    
    # Select time points to train on
    if time_range is not None:
        start_idx, end_idx = time_range
        time_indices = range(start_idx, min(end_idx, total_time_steps))
        selected_times = np.array([t * dt for t in time_indices])
        print(f"Training FEX-NN network on time range {start_idx}-{min(end_idx, total_time_steps)} ({len(time_indices)} time points)")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    elif num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Training FEX-NN network on {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
    
    for t_idx, t in enumerate(time_indices):
        print(f'Training FEX-NN network for {t_idx+1}/{len(time_indices)} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        
        # Check if model already exists
        if save_dir is not None:
            FEX_NN_path = os.path.join(save_dir, f'FEX_NN_t{t}.pth')
            
            if os.path.exists(FEX_NN_path):
                print(f'[INFO] FEX-NN model for time step {t} already exists. Skipping...')
                continue
        
        NTrain = int(size * 0.8)
        
        # Prepare data for current time step
        # Input: current_state at time step t
        current_state_t = current_state[0:NTrain, :, t]  # (N, dim)
        current_state_tensor = torch.tensor(current_state_t, dtype=torch.float32).to(device)
        
        # Target: (r_t * r_t^T) / dt computed from residuals at time step t
        residuals_t = residuals[0:NTrain, :, t]  # (N, dim)
        
        # Compute target: (r_t * r_t^T) / dt for each sample
        if dim == 1:
            # 1D case: target is (r_t^2) / dt, shape (N, 1)
            target_cov = (residuals_t ** 2) / dt
            output_dim = 1
        else:
            # Multi-D case: compute outer product r_t * r_t^T for each sample
            # Shape: (N, dim, dim) -> flatten to (N, dim*dim)
            N = residuals_t.shape[0]
            target_cov = np.zeros((N, dim * dim))
            for i in range(N):
                r_t = residuals_t[i:i+1, :]  # (1, dim)
                r_outer = np.outer(r_t, r_t)  # (dim, dim)
                target_cov[i, :] = (r_outer / dt).flatten()
            output_dim = dim * dim
        
        target_cov_tensor = torch.tensor(target_cov, dtype=torch.float32).to(device)
        
        # Validation data
        current_state_valid_t = current_state[NTrain:size, :, t]
        current_state_valid_tensor = torch.tensor(current_state_valid_t, dtype=torch.float32).to(device)
        
        residuals_valid_t = residuals[NTrain:size, :, t]
        if dim == 1:
            target_cov_valid = (residuals_valid_t ** 2) / dt
        else:
            N_valid = residuals_valid_t.shape[0]
            target_cov_valid = np.zeros((N_valid, dim * dim))
            for i in range(N_valid):
                r_t = residuals_valid_t[i:i+1, :]
                r_outer = np.outer(r_t, r_t)
                target_cov_valid[i, :] = (r_outer / dt).flatten()
        target_cov_valid_tensor = torch.tensor(target_cov_valid, dtype=torch.float32).to(device)
        
        # Initialize model
        FEX_NN = CovarianceNet(input_dim=dim, output_dim=output_dim, hid_size=50).to(device)
        FEX_NN.zero_grad()
        optimizer_nn = torch.optim.Adam(FEX_NN.parameters(), lr=learning_rate, weight_decay=1e-6)
        criterion_nn = torch.nn.MSELoss()
        
        best_valid_err_nn = float('inf')
        
        print(f'[INFO] Training FEX-NN network for time step {t}...')
        print(f'[INFO] Input shape: {current_state_tensor.shape}, Target shape: {target_cov_tensor.shape}')
        
        for it in range(n_iter):
            optimizer_nn.zero_grad()
            
            # Forward pass
            pred_cov = FEX_NN(current_state_tensor)
            loss = criterion_nn(pred_cov, target_cov_tensor)
            loss.backward()
            optimizer_nn.step()
            
            # Validation
            with torch.no_grad():
                pred_cov_valid = FEX_NN(current_state_valid_tensor)
                valid_loss = criterion_nn(pred_cov_valid, target_cov_valid_tensor)
            
            if valid_loss < best_valid_err_nn:
                FEX_NN.update_best()
                best_valid_err_nn = valid_loss
            
            if it % 500 == 0:
                print(f'[INFO] Epoch {it+1}/{n_iter}; Train Loss: {loss.item():.6f}; Valid Loss: {valid_loss.item():.6f}')
        
        FEX_NN.final_update()
        
        # Save model
        if save_dir is not None:
            FEX_NN_path = os.path.join(save_dir, f'FEX_NN_t{t}.pth')
            state_dict_cpu = {k: v.cpu() for k, v in FEX_NN.state_dict().items()}
            torch.save(state_dict_cpu, FEX_NN_path)
            print(f'[SAVE] Saved FEX-NN model to: {FEX_NN_path}')
        else:
            print(f'[WARNING] save_dir is None, not saving FEX-NN model for t{t}')
    
    print(f"[INFO] FEX-NN neural network training completed for time-dependent case!")


def load_time_dependent_FEX_NN_models(save_dir, dim, device='cpu', total_time_steps=None):
    """
    Load all time-dependent FEX-NN models (FEX_NN_t{t}.pth).
    
    Args:
        save_dir: Directory containing FEX_NN_t{t}.pth files
        dim: Number of dimensions
        device: Device to use ('cpu' or 'cuda')
        total_time_steps: Total number of time steps (None = auto-detect)
    
    Returns:
        dict: Dictionary mapping time step index to FEX-NN model, or empty dict if none found
    """
    from .ODEParser import CovarianceNet
    import glob
    
    models_dict = {}
    
    if save_dir is None or not os.path.exists(save_dir):
        return models_dict
    
    # Find all FEX_NN_t*.pth files
    pattern = os.path.join(save_dir, 'FEX_NN_t*.pth')
    model_files = glob.glob(pattern)
    
    if not model_files:
        return models_dict
    
    # Extract time step indices from filenames
    for model_file in model_files:
        # Extract time step from filename: FEX_NN_t{t}.pth
        filename = os.path.basename(model_file)
        try:
            # Extract number after 't' and before '.pth'
            time_step_str = filename.replace('FEX_NN_t', '').replace('.pth', '')
            time_step = int(time_step_str)
            
            # Load model
            output_dim = dim * dim if dim > 1 else 1
            FEX_NN = CovarianceNet(input_dim=dim, output_dim=output_dim, hid_size=50).to(device)
            FEX_NN.load_state_dict(torch.load(model_file, map_location=device))
            FEX_NN.eval()
            
            models_dict[time_step] = FEX_NN
        except (ValueError, KeyError) as e:
            print(f"[WARNING] Could not parse time step from {filename}: {e}")
            continue
    
    if models_dict:
        print(f"[INFO] Loaded {len(models_dict)} time-dependent FEX-NN models")
    else:
        print("[INFO] No time-dependent FEX-NN models found")
    
    return models_dict


def load_time_dependent_VAE_models(save_dir, dim, device='cpu', total_time_steps=None):
    """
    Load all time-dependent VAE models (VAE_FEX_t{t}.pth).
    
    Args:
        save_dir: Directory containing the models
        dim: Number of dimensions
        device: Device to load models on
        total_time_steps: Total number of time steps (if None, will scan directory)
    
    Returns:
        models_dict: Dictionary mapping time step t to VAE model
    """
    import glob
    
    models_dict = {}
    
    # Find all model files
    pattern = os.path.join(save_dir, 'VAE_FEX_t*.pth')
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f'[WARNING] No time-dependent VAE models found in {save_dir}')
        return models_dict
    
    # Extract time steps from filenames
    time_steps = []
    for f in model_files:
        try:
            # Extract time step from filename like "VAE_FEX_t42.pth"
            basename = os.path.basename(f)
            t_str = basename.replace('VAE_FEX_t', '').replace('.pth', '')
            time_steps.append(int(t_str))
        except:
            continue
    
    time_steps = sorted(time_steps)
    print(f'[INFO] Found {len(time_steps)} time-dependent VAE models for time steps: {time_steps[:5]}...{time_steps[-5:] if len(time_steps) > 10 else time_steps}')
    
    # Load each model
    for t in time_steps:
        VAE_path = os.path.join(save_dir, f'VAE_FEX_t{t}.pth')
        
        if not os.path.exists(VAE_path):
            print(f'[WARNING] Missing VAE model file for time step {t}, skipping...')
            continue
        
        # Load model
        VAE_model = VAE(input_dim=dim, hidden_dim=50, latent_dim=dim).to(device)
        state_dict = torch.load(VAE_path, map_location=device)
        VAE_model.load_state_dict(state_dict)
        VAE_model.eval()
        
        models_dict[t] = VAE_model
    
    print(f'[INFO] Successfully loaded {len(models_dict)} time-dependent VAE models')
    return models_dict


def predict_time_dependent_stochastic(Winc_tensor, t, models_dict, device='cpu'):
    """
    Predict stochastic update using time-dependent model at time step t.
    
    Args:
        Winc_tensor: Wiener increments tensor of shape (N, dim)
        t: Current time step index
        models_dict: Dictionary from load_time_dependent_models
        device: Device to use
    
    Returns:
        stoch_update: Stochastic update numpy array of shape (N, dim), or None if model not found
    """
    if t not in models_dict:
        # Try to find nearest time step
        available_t = sorted(models_dict.keys())
        if not available_t:
            return None
        # Use nearest available time step
        nearest_t = min(available_t, key=lambda x: abs(x - t))
        print(f'[WARNING] Model for time step {t} not found, using nearest time step {nearest_t}')
        t = nearest_t
    
    FN_multi, norm_params = models_dict[t]
    
    with torch.no_grad():
        # Normalize input (Wiener increments)
        # For time-dependent, we typically don't normalize inputs, but check norm_params structure
        if 'x_mean' in norm_params and 'x_std' in norm_params:
            x_mean = torch.tensor(norm_params['x_mean'], dtype=torch.float32).to(device)
            x_std = torch.tensor(norm_params['x_std'], dtype=torch.float32).to(device)
            Winc_normalized = (Winc_tensor - x_mean) / x_std
        else:
            Winc_normalized = Winc_tensor
        
        # Forward pass
        pred_normalized = FN_multi(Winc_normalized)
        
        # Denormalize output
        y_mean = torch.tensor(norm_params['mean'], dtype=torch.float32).to(device)
        y_std = torch.tensor(norm_params['std'], dtype=torch.float32).to(device)
        pred = pred_normalized * y_std + y_mean
        
        # Convert to numpy
        stoch_update = pred.cpu().detach().numpy()
    
    return stoch_update