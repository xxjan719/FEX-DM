import os, sys, warnings
import argparse
import subprocess

# Try to use modern importlib.metadata, fallback to pkg_resources if needed
try:
    from importlib.metadata import distributions, version
    USE_MODERN_METADATA = True
except ImportError:
    try:
        import pkg_resources
        USE_MODERN_METADATA = False
    except ImportError:
        raise ImportError("Neither importlib.metadata nor pkg_resources is available. Please install setuptools or use Python 3.8+")

# Try to import packaging for version parsing, fallback to simple comparison
try:
    from packaging.version import parse as parse_version
    HAS_PACKAGING = True
except ImportError:
    try:
        from pkg_resources import parse_version
        HAS_PACKAGING = True
    except ImportError:
        HAS_PACKAGING = False
        # Simple version comparison fallback
        def parse_version(v):
            """Simple version parser that returns a comparable tuple"""
            parts = []
            for part in v.split('.'):
                try:
                    parts.append(int(part))
                except ValueError:
                    # Handle non-numeric parts
                    parts.append(part)
            return tuple(parts)

warnings.filterwarnings("ignore")

class Config:
    """Singleton configuration class for FEX-DM"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_paths()
            self._setup_packages()
            Config._initialized = True
        
    def _setup_paths(self):
        """Setup all path configurations"""
        
        # Project directory structure
        self.DIR_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DIR_EXAMPLES = os.path.join(self.DIR_PROJECT, "examples")
        # Example directories structure
        self.DIR_SIR = os.path.join(self.DIR_EXAMPLES, "SIR")
        self.DIR_OU1d = os.path.join(self.DIR_EXAMPLES, "OU1d")
        
        # store file paths instead of loading numpy arrays immediately
        self.MODEL_CONFIG = {
            'SIR': {
                'name': 'FEX_dim_1',
                'op_sir_seq_path': os.path.join(self.DIR_SIR, 'op_sir_seq_1.npy'),
                'op_ou1d_seq_path': os.path.join(self.DIR_OU1d, 'op_ou1d_seq_1.npy'),
            },
            '2': {
                'name': 'FEX_dim_2',
                'op_lorenz_seq_path': os.path.join(self.DIR_SIR, 'op_lorenz_seq_2.npy'),
                'op_ou1d_seq_path': None,
            },
            '3': {
                'name': 'FEX_dim_3',
                'op_lorenz_seq_path': os.path.join(self.DIR_SIR, 'op_lorenz_seq_3.npy'),
                'op_ou1d_seq_path': None,
            },
        }
    
    def _setup_packages(self):
        """Setup package management configurations"""
        self.REQUIRED_PACKAGES = {
            'faiss-cpu': '1.8.1',
            'sympy': '1.13.1',
            'torch': '2.3.1',
            'torchvision': '0.18.1',
            'numpy': '1.26.4',
            'scipy':'1.13.0',
            'numba':'0.59.0',
            'matplotlib': '3.8.2',
            'scikit-learn':'1.5.2',
        }

        # Built-in modules that don't need installation
        self.BUILTIN_MODULES = {
            'typing', # Built into Python 3.5+
            'dataclasses', # Built into Python 3.7+
        }
    

    def check_and_install_packages(self):
        """Check and install required packages"""
        print("\nChecking and installing required packages...")
        
        # Get installed packages using modern or legacy API
        if USE_MODERN_METADATA:
            installed_packages = {}
            for dist in distributions():
                name = dist.metadata.get('Name', '')
                if name:
                    # Normalize package name (e.g., 'faiss-cpu' vs 'faiss_cpu')
                    normalized_name = name.lower().replace('_', '-')
                    try:
                        installed_packages[normalized_name] = dist.version
                    except Exception:
                        pass
        else:
            installed_packages = {pkg.key.lower(): pkg.version for pkg in pkg_resources.working_set}

        missing_packages = []
        outdated_packages = []

         # Check built-in modules first
        for module in self.BUILTIN_MODULES:
            try:
                __import__(module)
                print(f"[SUCCESS] {module} (built-in) is available")
            except ImportError:
                print(f"[WARNING] {module} (built-in) is not available in this Python version")

        # Check external packages
        for package, required_version in self.REQUIRED_PACKAGES.items():
            # Normalize package name for lookup
            normalized_package = package.lower().replace('_', '-')
            if normalized_package not in installed_packages:
                missing_packages.append(package)
            else:
                # check if version is sufficient
                try:
                    if USE_MODERN_METADATA:
                        installed_version = version(normalized_package)
                    else:
                        installed_version = pkg_resources.get_distribution(normalized_package).version
                    
                    if parse_version(installed_version) < parse_version(required_version):
                        outdated_packages.append((package, required_version, installed_version))
                except Exception:
                    # If we can't get version, treat as missing
                    missing_packages.append(package)

        # If packages are missing, check virtual environment first
        if missing_packages or outdated_packages:
            if not self._is_in_virtual_environment():
                raise ValueError("Please create a virtual environment first.")
        
        # Now install/upgrade packages
        for package, required_version in self.REQUIRED_PACKAGES.items():
            try:
                normalized_package = package.lower().replace('_', '-')
                # Check if package is installed
                if normalized_package not in installed_packages:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={required_version}"])
                    print(f"Successfully installed {package}")
                else:
                    # Check if version is sufficient
                    try:
                        if USE_MODERN_METADATA:
                            installed_version = version(normalized_package)
                        else:
                            installed_version = pkg_resources.get_distribution(normalized_package).version
                        
                        if parse_version(installed_version) < parse_version(required_version):
                            print(f"Upgrading {package} from {installed_version} to {required_version}...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", f"{package}>={required_version}"])
                            print(f"Successfully upgraded {package}")
                        else:
                            print(f"[SUCCESS] {package} {installed_version} is already installed")
                    except Exception:
                        # If version check fails, try to upgrade anyway
                        print(f"Upgrading {package}...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", f"{package}>={required_version}"])
                        print(f"Successfully upgraded {package}")
            except Exception as e:
                print(f"Error installing {package}: {str(e)}")
                raise

        print("\nAll required packages are installed and up to date!")
    
    def _is_in_virtual_environment(self):
        """Check if the script is running in a virtual environment"""
        return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    def create_virtual_environment(self, env_name='turbulentfex_env'):
        """Create a virtual environment for the project"""
        import subprocess
        import os
        
        env_path = os.path.join(self.DIR_PROJECT, env_name)
        
        if os.path.exists(env_path):
            print(f"Virtual environment '{env_name}' already exists at {env_path}")
            return env_path
        
        print(f"Creating virtual environment '{env_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", env_path])
            print(f"Successfully created virtual environment at {env_path}")
            return env_path
        except Exception as e:
            print(f"Error creating virtual environment: {str(e)}")
            raise
    
    def activate_virtual_environment(self, env_name='turbulentfex_env'):
        """Activate the virtual environment"""
        import subprocess
        import os
        
        env_path = os.path.join(self.DIR_PROJECT, env_name)
        
        if not os.path.exists(env_path):
            print(f"Virtual environment '{env_name}' does not exist. Creating it first...")
            self.create_virtual_environment(env_name)
        
        # Get the activation script path
        if os.name == 'nt':  # Windows
            activate_script = os.path.join(env_path, 'Scripts', 'activate.bat')
        else:  # Unix/Linux/macOS
            activate_script = os.path.join(env_path, 'bin', 'activate')
        
        if not os.path.exists(activate_script):
            print(f"Activation script not found at {activate_script}")
            return False
        
        print(f"To activate the virtual environment, run:")
        print(f"source {activate_script}")
        print(f"Or on Windows:")
        print(f"{activate_script}")
        
        return True
    
    def setup_environment(self, env_name='turbulentfex_env'):
        """Complete environment setup: create, activate, and install packages"""
        print("Setting up FEX-DM environment...")
        
        # Create virtual environment
        env_path = self.create_virtual_environment(env_name)
        
        # Check if we're in the virtual environment
        if not self._is_in_virtual_environment():
            print("Please activate the virtual environment first:")
            if os.name == 'nt':  # Windows
                print(f"{os.path.join(env_path, 'Scripts', 'activate.bat')}")
            else:  # Unix/Linux/macOS
                print(f"source {os.path.join(env_path, 'bin', 'activate')}")
            print("Then run this script again to install packages.")
            return False
        
        # Install packages
        self.check_and_install_packages()
        
        print("Environment setup complete!")
        return True
    
    def create_main_parser(self):
        """Create the main argument parser"""
        parser = argparse.ArgumentParser(
            description="FEX-DM for SDE simulation and model selection"
        )
        parser.add_argument(
            "--model",
            type=str,
            default = 'OU1d',
            choices = ['SIR', 'OU1d'],
            help = "Model to use for simulation and model selection"
        )
        parser.add_argument(
            "--SEED",
            type=int,
            default = 1234,
            help = "Random seed for reproducibility"
        )

        parser.add_argument(
            "--DEVICE", type = str,
            choices = ['cpu', 'cuda:0'],
            default = 'cuda:0',
            help = "Device to use for simulation and model selection"
        )

        parser.add_argument(
            '--CONTROLLER_LR',type = float,
            default = 2.0e-3,
            help = 'Learning rate for the controller'
        )

        parser.add_argument(
            '--CONTROLLER_INPUT_SIZE', type = int,
            default = 20,
            help = 'Input size for the controller'
        )

        parser.add_argument('--CONTROLLER_HIDDEN_SIZE', type=int, 
                            default=30,
                            help='Hidden size for controller')

        parser.add_argument('--CONTROLLER_TOP_SAMPLES_FRACTION', type=float, 
                            default=0.25,
                            help='Top samples fraction for controller')
        
        parser.add_argument('--CONTROLLER_QUANTILE_METHOD', type=str, 
                            choices = ['linear', 'quantile'],
                            default='linear',
                            help='Quantile method for controller')
        
        # FEX training settings
        parser.add_argument('--EXPLORATION_ITERS', type=int, 
                            default=20,
                            help='Number of exploration iterations')
        parser.add_argument('--NUM_TREES', type=int, 
                            default=200,
                            help='Number of trees in inner iteration')
        parser.add_argument('--TRAINING_DETER_SAMPLES',type=int, default=1000, help ='Number of experiment for testing fewer sample')
        
        parser.add_argument('--POOL_LIMIT', type=int,
                            default = 100,
                            help='Limit of models in the pool')


        # FEX training integration
        parser.add_argument('--INTEGRATOR_METHOD', type=str, 
                            choices = ['integration-based', 'derivative-based'],
                            default='integration-based',
                            help='Integration method')
        # FEX optimizer settings
        parser.add_argument('--FEX_LR_FIRST', type=float, 
                            default=8.0e-3,
                            help='Learning rate for FEX optimizer for the pre-training.')
        parser.add_argument('--FEX_LR_SECOND', type=float, 
                            default=5.0e-3,
                            help='Learning rate for FEX optimizer for the fine tuning.')
        
        parser.add_argument('--DIFF_SCALE',type=float,
                            default = 100,
                            help='Diffusion scale for FEX-DM')
        
        parser.add_argument('--ODESLOVER_TIME_STEPS',type=int,
                            default = 2000,
                            help='Number of time steps for ODE solver')
        
        parser.add_argument('--SHORT_SIZE',type=int,
                            default = 2048,
                            help='Short size for FEX-DM')
        

        
        parser.add_argument('--NN_SOLVER_LR',type=float,
                            default = 0.01,
                            help='Learning rate for NN solver')
        parser.add_argument('--NN_SOLVER_EPOCHS',type=int,
                            default = 2000,
                            help='Number of epochs for NN solver')
        
        parser.add_argument('--RESIDUAL_SAMPLES',type=int,
                            default = 10000,
                            help='Number of samples for DM training.')
        
        parser.add_argument('--TRAIN_WORKING_DIM',type=int,
                            default = 1,
                            help='Working dimension for DM training.')
        
        parser.add_argument('--NOISE_LEVEL',type=float,
                            default =4,
                            help='Noise level for MC simulation.')
        parser.add_argument('--TRAIN_SIZE',type=int,
                            default = 10000,
                            help='Number of samples for DM training.')
        # OU1d for  50000


        parser.add_argument('--DATA_SAVE_PATH',type=str,
                            default = None,
                            help='Path to save data.')
        parser.add_argument('--LOG_SAVE_PATH',type=str,
                            default = None,
                            help='Path to save log.')
        parser.add_argument('--FIGURE_SAVE_PATH',type=str,
                            default = None,
                            help='Path to save figure.')
        parser.add_argument('--TRAIN_EPOCHS_FIRST',type=int,
                            default = 20,
                            help='Number of first epochs for the first stage training.')
        parser.add_argument('--TRAIN_EPOCHS_SECOND',type=int,
                            default = 80000,
                            help='Number of fine tuning epochs for the first stage training.')
        
        parser.add_argument('--DOMAIN_START', type=float,
                            default=0.0,
                            help='Start point of the domain for initial condition generation')
        
        parser.add_argument('--DOMAIN_END', type=float,
                            default=2.5,
                            help='End point of the domain for initial condition generation')
        
        return parser


    def parse_args(self):
        """Parse command line arguments for main program"""
        main_parser = self.create_main_parser()
        return main_parser.parse_args()

# Create global instance
config = Config()

# Export commonly used attributes for backward compatibility
DIR_PROJECT = config.DIR_PROJECT
DIR_EXAMPLES = config.DIR_EXAMPLES
DIR_SIR = config.DIR_SIR
DIR_OU1d = config.DIR_OU1d


# Export functions
create_main_parser = config.create_main_parser
parse_args = config.parse_args
check_and_install_packages = config.check_and_install_packages
create_virtual_environment = config.create_virtual_environment
activate_virtual_environment = config.activate_virtual_environment
setup_environment = config.setup_environment



# Add project directory to sys.path
sys.path.append(DIR_PROJECT)


# =================================================================================
# CONVENIENT IMPORTS HELPER
# =================================================================================
def get_config():
    """ Get the global config instance - loads only once """
    return config



# Make it easy to import specific items
__all__ = [
    'config',
    'get_config',
    'create_main_parser',
    'parse_args',
    'check_and_install_packages',
    'create_virtual_environment',
    'activate_virtual_environment',
    'setup_environment',
    'DIR_PROJECT',
    'DIR_EXAMPLES',
    'DIR_SIR',
    'DIR_OU1d',
]

# =================================================================================
# MAIN EXECUTION FLOW
# =================================================================================
if __name__ == "__main__":
    # Example usage
    print("Configuration module loaded successfully!")
    
    # Example: Create and setup environment
    # config.create_virtual_environment('turbulentfex_env')
    # config.activate_virtual_environment('turbulentfex_env')
    # config.setup_environment('turbulentfex_env')
    # Get activation instructions
    if not os.path.exists(os.path.join(config.DIR_PROJECT, 'env')):
        config.create_virtual_environment('env')
    
    config.activate_virtual_environment('env')

    # Complete setup (requires manual activation first)
    config.setup_environment('env')
    # Uncomment to check packages
    # config.check_and_install_packages()
    