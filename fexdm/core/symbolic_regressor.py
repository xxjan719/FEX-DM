"""
Symbolic Regressor for discovering functional forms.

This module implements symbolic regression to discover drift and diffusion
functions from data using finite expression methods.
"""

import numpy as np
from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass
import itertools


@dataclass
class Expression:
    """Represents a symbolic expression."""
    
    formula: str
    complexity: int
    coefficients: np.ndarray
    function: Callable[[np.ndarray], np.ndarray]
    error: float = np.inf
    
    def __repr__(self) -> str:
        return f"Expression(formula='{self.formula}', error={self.error:.6f}, complexity={self.complexity})"


class SymbolicRegressor:
    """
    Symbolic regression using finite expression methods.
    
    This class discovers symbolic expressions for functions by testing
    a library of candidate basis functions and their combinations.
    """
    
    def __init__(
        self,
        basis_functions: Optional[List[str]] = None,
        max_complexity: int = 3,
        operators: Optional[List[str]] = None
    ):
        """
        Initialize the symbolic regressor.
        
        Parameters
        ----------
        basis_functions : list of str, optional
            List of basis function names (default: polynomial and trigonometric)
        max_complexity : int, optional
            Maximum complexity (number of terms) for expressions (default: 3)
        operators : list of str, optional
            Operators to use in expressions (default: ['+', '*'])
        """
        if basis_functions is None:
            basis_functions = ['1', 'x', 'x**2', 'x**3', 'sin(x)', 'cos(x)', 'exp(-x**2)']
        
        if operators is None:
            operators = ['+', '*']
        
        self.basis_functions = basis_functions
        self.max_complexity = max_complexity
        self.operators = operators
        self.best_expression_ = None
        self.all_expressions_ = []
    
    def _create_basis_function(self, formula: str) -> Callable:
        """
        Create a callable function from a string formula.
        
        SECURITY NOTE: This method uses eval() with a restricted namespace.
        Only use with trusted formula strings from the predefined basis_functions
        list. Do not pass user-supplied formulas directly without validation.
        """
        # Safe evaluation with only allowed functions and no builtins
        safe_dict = {
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
        }
        
        def func(x):
            safe_dict['x'] = x
            try:
                result = eval(formula, {"__builtins__": {}}, safe_dict)
                # Ensure result is an array with same shape as x
                if np.isscalar(result):
                    result = np.full_like(x, result, dtype=float)
                return result
            except (ValueError, NameError, ZeroDivisionError, TypeError, AttributeError):
                # Handle expected mathematical errors
                return np.zeros_like(x)
        
        return func
    
    def _evaluate_expression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        basis_combo: List[str]
    ) -> Expression:
        """
        Evaluate a candidate expression against data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target data
        basis_combo : list of str
            Combination of basis functions to evaluate
        
        Returns
        -------
        Expression
            The evaluated expression with fitted coefficients and error
        """
        # Build design matrix
        design_matrix = []
        for basis in basis_combo:
            func = self._create_basis_function(basis)
            design_matrix.append(func(X))
        
        design_matrix = np.column_stack(design_matrix)
        
        # Least squares fit
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
            if len(residuals) > 0:
                error = residuals[0] / len(y)
            else:
                error = np.mean((design_matrix @ coeffs - y) ** 2)
        except (np.linalg.LinAlgError, ValueError):
            # Handle linear algebra errors (singular matrix, etc.)
            coeffs = np.zeros(len(basis_combo))
            error = np.inf
        
        # Create formula string
        terms = []
        for i, (coeff, basis) in enumerate(zip(coeffs, basis_combo)):
            if np.abs(coeff) > 1e-10:
                if basis == '1':
                    terms.append(f"{coeff:.6f}")
                else:
                    terms.append(f"{coeff:.6f}*{basis}")
        
        formula = " + ".join(terms) if terms else "0"
        
        # Create callable function
        def combined_func(x):
            result = np.zeros_like(x)
            for coeff, basis in zip(coeffs, basis_combo):
                func = self._create_basis_function(basis)
                result += coeff * func(x)
            return result
        
        return Expression(
            formula=formula,
            complexity=len(basis_combo),
            coefficients=coeffs,
            function=combined_func,
            error=error
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'SymbolicRegressor':
        """
        Fit the symbolic regressor to data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data (1D or 2D)
        y : np.ndarray
            Target data
        verbose : bool, optional
            Whether to print progress (default: False)
        
        Returns
        -------
        self
            The fitted regressor
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        
        # Try all combinations of basis functions up to max_complexity
        all_expressions = []
        
        for complexity in range(1, self.max_complexity + 1):
            for basis_combo in itertools.combinations_with_replacement(
                self.basis_functions, complexity
            ):
                expr = self._evaluate_expression(X, y, list(basis_combo))
                all_expressions.append(expr)
                
                if verbose and expr.error < np.inf:
                    print(f"Tried: {expr.formula[:80]}, Error: {expr.error:.6e}")
        
        # Sort by error
        all_expressions.sort(key=lambda e: e.error)
        self.all_expressions_ = all_expressions
        
        # Select best expression
        if all_expressions:
            self.best_expression_ = all_expressions[0]
            if verbose:
                print(f"\nBest expression: {self.best_expression_.formula}")
                print(f"Error: {self.best_expression_.error:.6e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the best discovered expression.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.best_expression_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.best_expression_.function(X)
    
    def get_expression(self) -> Optional[Expression]:
        """Get the best discovered expression."""
        return self.best_expression_
    
    def __repr__(self) -> str:
        if self.best_expression_ is not None:
            return f"SymbolicRegressor(best='{self.best_expression_.formula[:50]}...')"
        return "SymbolicRegressor(not fitted)"
