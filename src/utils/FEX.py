import torch
from torch import Tensor
import torch.nn as nn
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

if __name__ == "__main__":
    op_seq = torch.tensor([2, 0, 3, 3])
    fex = FEX(op_seq, 1)
    print(fex.expression_visualize())
    print(fex.expression_visualize_simplified())
