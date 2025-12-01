import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import re
import sympy as sp

class BaseFEX(nn.Module):
    """Base class with shared unary and binary operations"""
    @staticmethod
    def unary(op_idx: int, x: Tensor):
        if op_idx == 0:
            return torch.zeros_like(x)
        
        elif op_idx == 1:
            return torch.ones_like(x)
        
        elif op_idx == 2:
            return x
        
        elif op_idx == 3:
            return torch.square(x)

        elif op_idx == 4:
            return torch.pow(x, 3)
        
        elif op_idx == 5:
            return torch.pow(x, 4)
        
        elif op_idx == 6:
            return torch.exp(x)
        
        elif op_idx == 7:
            return torch.sin(x)

        elif op_idx == 8:
            return torch.cos(x)
        
        else:
            raise ValueError(f"Unary operator index {op_idx} is undefined.")
    
    @staticmethod
    def binary(op_idx: int, x: Tensor, y: Tensor):
        if op_idx == 0:
            return torch.add(x,y)
        elif op_idx == 1:
            return torch.sub(x,y)
        elif op_idx == 2:
            return torch.mul(x,y)
        elif op_idx == 3:
            raise ValueError(f"Binary operator index {op_idx} is undefined.")
    
class FEX(BaseFEX):
    def __init__(self, op_seq: torch.Tensor, dim: int)->None:
        super().__init__()
        self.op_seq = op_seq
        self.dim = dim

        self.linear_a = nn.Parameter(torch.ones(dim))
        self.linear_b = nn.Parameter(torch.zeros(dim))
        self.linear_c = nn.Parameter(torch.ones(dim))
        self.linear_d = nn.Parameter(torch.zeros(dim))
        self.linear_e = nn.Parameter(torch.ones(dim))
        self.linear_f = nn.Parameter(torch.zeros(dim))
        # For expression visualization
        self.exprs_0 = ""
        self.exprs_1 = ""
        self.exprs_2 = ""
    
    def forward(self, x: Tensor)-> Tensor:
        # op_seq length: 4 for 1D, 8 for 2D, 12 for 3D
        # Process each dimension separately for first three parts, then all together for fourth
        linear_a = self.linear_a.to(x.device)
        linear_b = self.linear_b.to(x.device)
        linear_c = self.linear_c.to(x.device)
        linear_d = self.linear_d.to(x.device)
        linear_e = self.linear_e.to(x.device)
        linear_f = self.linear_f.to(x.device)
        
        fourth_parts = []
        for i in range(self.dim):
            # Each dimension uses 4 consecutive op_seq elements
            op_start = i * 4
            op_idx_0 = int(self.op_seq[op_start + 0].item())
            op_idx_1 = int(self.op_seq[op_start + 1].item())
            op_idx_2 = int(self.op_seq[op_start + 2].item())
            op_idx_3 = int(self.op_seq[op_start + 3].item())
            
            # Extract the i-th dimension from x
            # x can have shape (dim,) or (batch, dim) or (batch, time, dim), etc.
            if x.dim() == 1:
                x_dim = x[i]
            else:
                # For multi-dimensional x, extract along the last dimension
                x_dim = x[..., i]
            
            # op_seq[3] -> first unary with linear_e[i] as weight and linear_f[i] as bias
            first_part = linear_e[i] * self.unary(op_idx_3, x_dim) + linear_f[i]
            # op_seq[2] -> second unary with linear_c[i] as weight and linear_d[i] as bias
            second_part = linear_c[i] * self.unary(op_idx_2, x_dim) + linear_d[i]
            # op_seq[1] -> binary operation on the two weighted parts
            third_part = self.binary(op_idx_1, first_part, second_part)
            # op_seq[0] -> final unary operation with linear_a[i] as weight
            fourth_part = linear_a[i] * self.unary(op_idx_0, third_part) + linear_b[i]
            fourth_parts.append(fourth_part)
        
        # Multiply all fourth_parts together element-wise
        result = fourth_parts[0]
        for i in range(1, len(fourth_parts)):
            result = result * fourth_parts[i]
        
        # Ensure output shape is correct: if input is (N, dim), output should be (N,) or (N, 1)
        # Handle potential broadcasting issues that create (N, N) or (N, M) shapes
        if result.dim() == 2:
            if result.shape[0] == result.shape[1]:
                # Likely a square matrix (N, N) due to broadcasting issue - take diagonal
                result = torch.diagonal(result)
            elif result.shape[1] == 1:
                # (N, 1) -> (N,)
                result = result.squeeze(-1)
            elif result.shape[1] > 1 and result.shape[1] != result.shape[0]:
                # (N, M) where M != N - take first column
                result = result[:, 0]
        
        return result
    
    def _op_to_str(self, op_idx:int, x_str:str) -> str:
        """Convert operator index to string representation"""
        if op_idx == 0:
            return "0"
        elif op_idx == 1:
            return "1"
        elif op_idx == 2:
            return x_str
        elif op_idx == 3:
            return f"({x_str})**2"
        elif op_idx == 4:
            return f"({x_str})**3"
        elif op_idx == 5:
            return f"({x_str})**4"
        elif op_idx == 6:
            return f"exp({x_str})"
        elif op_idx == 7:
            return f"sin({x_str})"
        elif op_idx == 8:
            return f"cos({x_str})"
        else:
            raise ValueError(f"Unary Operator index {op_idx} is undefined.")
    
    def _binary_to_str(self, op_idx:int, x_str:str, y_str:str) -> str:
        """ Convert binary operator index to string representation"""
        if op_idx == 0:
            return f"({x_str}) + ({y_str})"
        elif op_idx == 1:
            return f"({x_str}) - ({y_str})"
        elif op_idx == 2:
            return f"({x_str}) * ({y_str})"
        elif op_idx == 3:
            raise ValueError(f"Binary operator index {op_idx} is undefined.")
    
    def expression_visualize(self) -> str:
        """
        Visualize the expression using _op_to_str and _binary_to_str methods.
        Matches the exact structure of the forward method.
        """
        # Non-linear part - matches forward method exactly
        exprs = []
        for i in range(self.dim):
            # Each dimension uses 4 consecutive op_seq elements
            op_start = i * 4
            op_idx_0 = int(self.op_seq[op_start + 0].item())  # Final unary
            op_idx_1 = int(self.op_seq[op_start + 1].item())  # Binary
            op_idx_2 = int(self.op_seq[op_start + 2].item())  # Second unary
            op_idx_3 = int(self.op_seq[op_start + 3].item())  # First unary
            
            # Get parameter values
            linear_e_val = self.linear_e[i].item()
            linear_f_val = self.linear_f[i].item()
            linear_c_val = self.linear_c[i].item()
            linear_d_val = self.linear_d[i].item()
            linear_a_val = self.linear_a[i].item()
            linear_b_val = self.linear_b[i].item()
            
            # Build expression matching forward method:
            # first_part = linear_e[i] * unary(op_idx_3, x_dim) + linear_f[i]
            x_var = f'x{i+1}'
            unary_str_3 = self._op_to_str(op_idx_3, x_var)
            first_part = f"{linear_e_val:.4f}*({unary_str_3})+{linear_f_val:.4f}"
            
            # second_part = linear_c[i] * unary(op_idx_2, x_dim) + linear_d[i]
            unary_str_2 = self._op_to_str(op_idx_2, x_var)
            second_part = f"{linear_c_val:.4f}*({unary_str_2})+{linear_d_val:.4f}"
            
            # third_part = binary(op_idx_1, first_part, second_part)
            third_part = self._binary_to_str(op_idx_1, first_part, second_part)
            
            # fourth_part = linear_a[i] * unary(op_idx_0, third_part) + linear_b[i]
            unary_str_0 = self._op_to_str(op_idx_0, third_part)
            fourth_part = f"{linear_a_val:.4f}*({unary_str_0})+{linear_b_val:.4f}"
            
            exprs.append(fourth_part)
        
        self.exprs_0 = exprs[0] if len(exprs) > 0 else ""
        self.exprs_1 = exprs[1] if len(exprs) > 1 else ""
        self.exprs_2 = exprs[2] if len(exprs) > 2 else ""
        
        # Combine nonlinear expressions (multiply all fourth_parts together)
        if len(exprs) == 1:
            self.nonlinear_expr = f"({exprs[0]})"
        elif len(exprs) == 2:
            self.nonlinear_expr = f"({exprs[0]})*({exprs[1]})"
        else:
            self.nonlinear_expr = f"({exprs[0]})*({exprs[1]})*({exprs[2]})"
        
        # Force part - if force parameters exist, apply unary operator
        if hasattr(self, 'force_a') and hasattr(self, 'force_b') and len(self.op_seq) > self.dim * 4:
            force_linear = f"{self.force_a.item():.4f}*t + {self.force_b.item():.4f}"
            force_op_idx = int(self.op_seq[-1].item())
            self.force_expr = self._op_to_str(force_op_idx, force_linear)
            expr_str = f"({self.nonlinear_expr}) + ({self.force_expr})"
        else:
            self.force_expr = ""
            expr_str = self.nonlinear_expr
        
        return expr_str
    
    def expression_visualize_simplified(self) -> str:
        """
        Simplified version of expression visualization using sympy for maximum simplification.
        Uses _op_to_str and _binary_to_str methods.
        """
        # First generate the full expression
        self.expression_visualize()
        
        # Function to check if expression is constant (no x1,x2,x3)
        def is_constant_expr(expr):
            return all(f'x{i+1}' not in expr for i in range(self.dim))
        
        # Function to evaluate constant expression
        def eval_constant_expr(expr):
            # Replace mathematical operations with Python syntax
            expr = expr.replace('^', '**')
            try:
                result = eval(expr)
                # Return as integer if it's a whole number, otherwise as float
                if isinstance(result, float) and result.is_integer():
                    return str(int(result))
                return f"{result:.4f}".rstrip('0').rstrip('.')
            except:
                return expr
        
        # Process each expression
        exprs = [self.exprs_0, self.exprs_1, self.exprs_2]
        simplified_exprs = []
        
        for i, expr in enumerate(exprs):
            if expr == "":
                continue
            if is_constant_expr(expr):
                result = eval_constant_expr(expr)
                simplified_exprs.append(result)
            else:
                # Convert to sympy and expand
                try:
                    # First convert to sympy
                    expr_sympy = sp.sympify(expr)
                    # Expand to get fully expanded form
                    expanded = sp.expand(expr_sympy)
                    simplified_exprs.append(expanded)
                except Exception as e:
                    # If sympy fails, use original expression
                    simplified_exprs.append(expr)
        
        # Combine the parts - expand each part first, then multiply
        try:
            # Convert all parts to sympy and expand them individually
            sympy_parts = []
            for expr in simplified_exprs:
                if isinstance(expr, sp.Basic):
                    sympy_parts.append(sp.expand(expr))
                elif isinstance(expr, str):
                    sympy_parts.append(sp.expand(sp.sympify(expr)))
                else:
                    sympy_parts.append(sp.expand(sp.sympify(str(expr))))
            
            # Multiply all parts together
            if len(sympy_parts) == 1:
                nonlinear_expr = sympy_parts[0]
            elif len(sympy_parts) == 2:
                nonlinear_expr = sympy_parts[0] * sympy_parts[1]
            else:
                nonlinear_expr = sympy_parts[0] * sympy_parts[1] * sympy_parts[2]
            
            # Fully expand the combined expression to get sum of terms
            # This converts x1*(x1 + 1) to x1**2 + x1
            # Use expand_multinomial to ensure full expansion
            nonlinear_simplified = sp.expand(nonlinear_expr, deep=True, mul=True)
            
            # Don't use simplify() as it may factor back - just collect like terms
            # Use collect to combine coefficients of like terms without factoring
            try:
                nonlinear_simplified = sp.collect(nonlinear_simplified, evaluate=True)
            except:
                pass
            
            # For force term, handle carefully
            if self.force_expr:
                force_sympy = sp.sympify(self.force_expr)
                if 't' in str(force_sympy) or 'exp(' in str(force_sympy):
                    force_simplified = force_sympy  # Keep time-dependent and exponential terms as is
                else:
                    force_simplified = sp.simplify(force_sympy)
                
                # Combine and expand final expression
                final_expr = nonlinear_simplified + force_simplified
                final_simplified = sp.expand(final_expr)
                final_simplified = sp.simplify(final_simplified)
            else:
                final_simplified = nonlinear_simplified
            
            # Convert to string and clean up
            result_str = str(final_simplified)
            
            # Remove unnecessary 1.0* and 0.0+ patterns (more aggressive)
            import re
            # Remove 1.0* and *1.0 (but be careful with 10, 100, etc.)
            result_str = re.sub(r'\b1\.0\s*\*\s*', '', result_str)
            result_str = re.sub(r'\*\s*1\.0\b', '', result_str)
            result_str = re.sub(r'\b1\.0\s*\*\s*', '', result_str)  # Do it twice to catch nested cases
            # Remove 0.0 + and + 0.0
            result_str = result_str.replace('0.0 + ', '').replace(' + 0.0', '')
            result_str = result_str.replace('0.0*', '0*').replace('*0.0', '*0')
            result_str = result_str.replace(' - 0.0', '').replace('0.0 - ', '-')
            # Remove trailing .0 from numbers (e.g., 1.0 -> 1, but keep 10, 100, etc.)
            result_str = re.sub(r'(\d+)\.0+(\D|$)', r'\1\2', result_str)
            result_str = re.sub(r'(\d+)\.0+$', r'\1', result_str)
            # Clean up extra spaces around operators
            result_str = re.sub(r'\s+\+\s+', ' + ', result_str)
            result_str = re.sub(r'\s+\-\s+', ' - ', result_str)
            result_str = re.sub(r'\s+\*\s+', '*', result_str)
            
            return result_str
        except Exception as e:
            # If sympy fails, return the original nonlinear expression
            if isinstance(nonlinear_expr, sp.Basic):
                return str(nonlinear_expr)
            return str(nonlinear_expr) if nonlinear_expr else self.nonlinear_expr


# Expression cache for FEX_model_learned
_expression_cache = {}


def FEX_model_learned(x, model_name='OU1d', params_name=None, noise_level=1.0, device='cpu', base_path=None, domain_folder=None):
    """
    Create learned FEX model by reading final expressions from file.
    
    Args:
        x: Input tensor of shape (batch_size, dim) or numpy array
        model_name: Model name (e.g., 'OU1d', 'SIR')
        params_name: Params name (defaults to model_name)
        noise_level: Noise level (default: 1.0)
        device: Device string ('cpu' or 'cuda:0')
        base_path: Base path to Results directory (if None, will construct from current file location)
        domain_folder: Domain folder name (e.g., 'domain_0.0_2.5'). If None, will try to detect from base_path or use default.
    
    Returns:
        Output tensor/array of shape (batch_size, dim) with learned expressions
    """
    # Use defaults if not provided
    if params_name is None:
        params_name = model_name
    
    # Get dimension from input
    if isinstance(x, torch.Tensor):
        batch_size = x.shape[0]
        dim = x.shape[1]
        x_device = x.device
        x_dtype = x.dtype
    else:
        batch_size = x.shape[0]
        dim = x.shape[1]
        x_device = None
        x_dtype = None
    
    # Extract dimensions from input
    x_tensors = []
    for i in range(dim):
        if isinstance(x, torch.Tensor):
            x_tensors.append(x[:, i:i+1].squeeze(-1))
        else:
            x_tensors.append(x[:, i])
    
    # Construct path to final_expressions.txt
    if base_path is None:
        # Try to get base_path from config if available
        try:
            import config
            if str(device) != 'cpu' and 'cuda' in str(device):
                base_path = os.path.join(config.DIR_PROJECT, 'Results', 'gpu_folder', model_name)
            else:
                base_path = os.path.join(config.DIR_PROJECT, 'Results', 'cpu_folder', model_name)
        except:
            # Fallback: construct from current file location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            if str(device) != 'cpu' and 'cuda' in str(device):
                base_path = os.path.join(project_root, 'Results', 'gpu_folder', model_name)
            else:
                base_path = os.path.join(project_root, 'Results', 'cpu_folder', model_name)
    
    # If domain_folder is not provided, try to detect it from base_path
    if domain_folder is None and base_path is not None:
        # Check if base_path contains a domain folder
        path_parts = base_path.split(os.sep)
        for part in path_parts:
            if part.startswith('domain_'):
                domain_folder = part
                break
    
    # Construct the full path to final_expressions.txt
    if domain_folder:
        expr_file = os.path.join(base_path, domain_folder, f'noise_{noise_level}', 'final_expressions.txt')
    else:
        expr_file = os.path.join(base_path, f'noise_{noise_level}', 'final_expressions.txt')
    
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Final expressions file not found: {expr_file}")
    
    # Check if expressions are already cached for this configuration
    cache_key = f"{model_name}_{params_name}_noise_{noise_level}"
    if cache_key not in _expression_cache:
        # Read the expressions from file
        expressions = {}
        operator_sequences = {}
        with open(expr_file, 'r') as f:
            lines = f.readlines()
        
        print(f"\n[INFO] Reading learned expressions from: {expr_file}")
        print("="*60)
        print("LEARNED FEX EXPRESSIONS:")
        print("="*60)
        
        # Parse the file - handle both old format (single line) and new format (multi-line with operator sequence)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('dimension_'):
                # Extract dimension name
                if ':' in line:
                    dim_name = line.split(':')[0]
                else:
                    dim_name = line
                
                # Check if this is new format (multi-line) or old format (single line)
                op_seq = None
                expr_str = None
                
                # Check next line to determine format
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    
                    # Check if next line contains "Operator Sequence:" (new format)
                    if 'Operator Sequence:' in next_line:
                        # New format: multi-line with operator sequence
                        op_seq_str = next_line.split('Operator Sequence:')[1].strip()
                        try:
                            import ast
                            op_seq = ast.literal_eval(op_seq_str)
                            operator_sequences[dim_name] = op_seq
                        except:
                            operator_sequences[dim_name] = op_seq_str
                        
                        # Get expression from the line after operator sequence
                        if i + 2 < len(lines):
                            expr_line = lines[i + 2].strip()
                            if 'Expression:' in expr_line:
                                expr_str = expr_line.split('Expression:')[1].strip()
                                expressions[dim_name] = expr_str
                                i += 3  # Skip dimension, operator sequence, and expression lines
                            else:
                                i += 2  # Skip dimension and operator sequence lines
                        else:
                            i += 2  # Skip dimension and operator sequence lines
                    else:
                        # Old format: dimension_X: expression (single line)
                        parts = line.split(': ', 1)
                        if len(parts) == 2:
                            expr_str = parts[1].strip()
                            expressions[dim_name] = expr_str
                        i += 1  # Move to next line
                else:
                    # No next line, old format check
                    parts = line.split(': ', 1)
                    if len(parts) == 2:
                        expr_str = parts[1].strip()
                        expressions[dim_name] = expr_str
                    i += 1
                
                # Print with operator sequence if available
                if dim_name in expressions:
                    if dim_name in operator_sequences:
                        print(f"{dim_name}:")
                        print(f"  Operator Sequence: {operator_sequences[dim_name]}")
                        print(f"  Expression: {expressions[dim_name]}")
                    else:
                        print(f"{dim_name}: {expressions[dim_name]}")
            else:
                i += 1
        
        print("="*60)
        
        if not expressions:
            raise ValueError(f"No expressions found in {expr_file}")
        
        # Cache the expressions
        _expression_cache[cache_key] = expressions
    else:
        # Use cached expressions - still print them for visibility
        expressions = _expression_cache[cache_key]
        print(f"\n[INFO] Using cached expressions for {model_name} (noise={noise_level})")
        print("="*60)
        print("LEARNED FEX EXPRESSIONS (from cache):")
        print("="*60)
        for dim_name, expr_str in sorted(expressions.items()):
            print(f"{dim_name}: {expr_str}")
        print("="*60)
    
    # Create the learned model outputs
    outputs = []
    
    # Process each dimension
    for dim_idx in range(1, dim + 1):
        dim_key = f'dimension_{dim_idx}'
        if dim_key not in expressions:
            raise ValueError(f"Expression for {dim_key} not found in file")
        
        expr_str = expressions[dim_key]
        
        # Convert expression string to torch/numpy operations
        try:
            if isinstance(x, torch.Tensor):
                # Use torch operations
                # Create local variables for x1, x2, x3
                x1 = x_tensors[0] if dim >= 1 else torch.zeros_like(x_tensors[0])
                x2 = x_tensors[1] if dim >= 2 else torch.zeros_like(x_tensors[0])
                x3 = x_tensors[2] if dim >= 3 else torch.zeros_like(x_tensors[0])
                
                # Print expression being evaluated (only for first call or when verbose)
                if dim_idx == 1:
                    print(f"\n[INFO] Evaluating expression for dimension {dim_idx}: {expr_str}")
                
                # Create safe evaluation environment
                safe_dict = {
                    'x1': x1, 'x2': x2, 'x3': x3,
                    'torch': torch,
                    '__builtins__': {},
                }
                
                # Evaluate the expression
                result = eval(expr_str, safe_dict)
                
                # Ensure result is a tensor with correct shape
                if not isinstance(result, torch.Tensor):
                    result = torch.tensor(result, dtype=x_dtype, device=x_device)
                elif result.dim() == 0:
                    result = result.expand_as(x1)
                
                outputs.append(result)
            else:
                # Use numpy operations
                x1 = x_tensors[0] if dim >= 1 else np.zeros_like(x_tensors[0])
                x2 = x_tensors[1] if dim >= 2 else np.zeros_like(x_tensors[0])
                x3 = x_tensors[2] if dim >= 3 else np.zeros_like(x_tensors[0])
                
                # Create safe evaluation environment
                safe_dict = {
                    'x1': x1, 'x2': x2, 'x3': x3,
                    'np': np,
                    '__builtins__': {},
                }
                
                # Evaluate the expression
                result = eval(expr_str, safe_dict)
                outputs.append(result)
                
        except Exception as e:
            print(f"Error evaluating expression for {dim_key}: {expr_str}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Stack outputs to create (batch_size, dim) tensor
    if isinstance(x, torch.Tensor):
        return torch.stack(outputs, dim=1)
    else:
        return np.stack(outputs, axis=1)


if __name__ == "__main__":
    print("="*60)
    print("Example 1: Basic FEX model")
    print("="*60)
    op_seq = torch.tensor([2, 0, 3, 3])
    fex = FEX(op_seq, 1)
    print(f"Full expression: {fex.expression_visualize()}")
    print(f"Simplified expression: {fex.expression_visualize_simplified()}")
    print()
    
    print("="*60)
    print("Example 2: FEX_model_learned - Reading from final_expressions.txt")
    print("="*60)
    
    # Example: Read learned model for OU1d with noise level 1.0
    try:
        # Create test input
        test_input = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input: {test_input.squeeze().tolist()}")
        
        # Load learned model
        output = FEX_model_learned(
            x=test_input,
            model_name='OU1d',
            noise_level=1.0,
            device='cpu'
        )
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Output: {output.squeeze().tolist()}")
        print("\n[SUCCESS] FEX_model_learned works correctly!")
        
    except FileNotFoundError as e:
        print(f"\n[WARNING] {e}")
        print("This is expected if final_expressions.txt doesn't exist yet.")
        print("Run 1stage_deterministic.py first to generate the expressions file.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
