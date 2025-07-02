# setup_environment.py
"""
Environment setup and validation script for BoTorch 0.13.x + Ax ≥0.5
"""

import subprocess
import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        raise RuntimeError(f"Python 3.10+ required, got {version.major}.{version.minor}")
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")


def install_packages():
    """Install required packages."""
    packages = [
        "torch>=2.0.0",
        "torchvision",
        "torchaudio",
        "gpytorch>=1.11",
        "botorch==0.10.0",
        "ax-platform>=0.5.0",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib"
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def validate_installation():
    """Validate that all packages are correctly installed."""
    import torch
    import gpytorch
    import botorch
    import pandas as pd
    import numpy as np
    import sklearn

    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ GPyTorch {gpytorch.__version__}")
    print(f"✓ BoTorch {botorch.__version__}")
    print(f"✓ Pandas {pd.__version__}")
    print(f"✓ NumPy {np.__version__}")
    print(f"✓ Scikit-learn {sklearn.__version__}")

    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("! CUDA not available, using CPU")


def create_directory_structure():
    """Create project directory structure."""
    dirs = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "outputs/candidates",
        "outputs/pareto",
        "logs",
        "tests"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")


def create_sample_data():
    """Create sample CSV data matching your format."""
    import pandas as pd
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    n_samples = 50

    # Additive names matching your data
    additive_names = [
        'bsa', 'cacl2', 'dmso', 'edta', 'etoh', 'fe2', 'glycerol', 'imidazole',
        'mgcl2', 'mn2', 'paa', 'peg200k', 'peg20k', 'peg400', 'peg5k', 'pl127',
        'pva', 'sucrose', 'tw20', 'tw80', 'tx100', 'zn2'
    ]

    data = {}

    # Fixed columns from your data
    data['Trial'] = range(1, n_samples + 1)
    data['tmb'] = np.random.uniform(0.2, 0.8, n_samples)
    data['hrp'] = np.full(n_samples, 5.00E-05)  # Constant like in your data
    data['h2o2'] = np.full(n_samples, 0.12)  # Constant like in your data

    # Switch and level variables for each additive
    for name in additive_names:
        # Switch variables (binary)
        data[f'{name}_sw'] = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
        # Level variables (concentrations)
        data[f'{name}_lv'] = np.where(
            data[f'{name}_sw'] == 1,
            np.random.uniform(0.01, 1.0, n_samples),
            0.0
        )

    # Objective variables
    data['Imax'] = np.random.uniform(0.1, 2.0, n_samples)
    data['Lag_time'] = np.random.uniform(10, 300, n_samples)
    data['efficiency'] = np.random.uniform(60, 95, n_samples)

    # Create DataFrame and save
    df = pd.DataFrame(data)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/sample_data.csv", index=False)
    print(f"✓ Created sample data with {n_samples} samples")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Shape: {df.shape}")


if __name__ == "__main__":
    print("Setting up Chemical Formulation Optimization Environment...")
    print("=" * 60)

    try:
        check_python_version()
        install_packages()
        validate_installation()
        create_directory_structure()
        create_sample_data()

        print("=" * 60)
        print("✓ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your CSV data in data/raw/")
        print("2. Run: python run_pipeline.py")
        print("3. Check outputs in outputs/candidates/")

    except Exception as e:
        print(f"✗ Setup failed: {e}")
        sys.exit(1)

# ================================================================
# run_pipeline.py
"""
Main pipeline for chemical formulation optimization using BoTorch and Ax
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# BoTorch and Ax imports
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

# Ax imports
from ax.service.ax_client import AxClient
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import RangeParameter, ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.runners.synthetic import SyntheticRunner


class ChemicalFormulationOptimizer:
    """
    Multi-objective optimization for chemical formulations using BoTorch and Ax.
    """

    def __init__(self, data_dir: Path, models_dir: Path, outputs_dir: Path,
                 device: str = "auto"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.outputs_dir = Path(outputs_dir)

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize attributes
        self.data = None
        self.features = None
        self.objectives = None
        self.models = {}

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, csv_path: Optional[str] = None):
        """Load and preprocess data."""
        if csv_path is None:
            csv_path = self.data_dir / "raw" / "sample_data.csv"

        self.logger.info(f"Loading data from {csv_path}")
        self.data = pd.read_csv(csv_path)

        # Identify feature columns (excluding objectives and trial info)
        objective_cols = ['Imax', 'Lag_time', 'efficiency']
        exclude_cols = ['Trial'] + objective_cols

        self.features = [col for col in self.data.columns if col not in exclude_cols]
        self.objectives = objective_cols

        self.logger.info(f"Loaded {len(self.data)} samples with {len(self.features)} features")
        self.logger.info(f"Features: {self.features[:10]}...")  # Show first 10
        self.logger.info(f"Objectives: {self.objectives}")

        return self.data

    def preprocess_data(self):
        """Preprocess data for optimization."""
        self.logger.info("Preprocessing data...")

        # Convert to tensors
        X = torch.tensor(self.data[self.features].values, dtype=torch.float32, device=self.device)
        Y = torch.tensor(self.data[self.objectives].values, dtype=torch.float32, device=self.device)

        # Handle any missing values
        X = torch.nan_to_num(X, nan=0.0)
        Y = torch.nan_to_num(Y, nan=0.0)

        self.logger.info(f"Data shapes - X: {X.shape}, Y: {Y.shape}")

        return X, Y

    def train_surrogate_models(self, X, Y):
        """Train Gaussian Process surrogate models for each objective."""
        self.logger.info("Training surrogate models...")

        self.models = {}

        for i, obj_name in enumerate(self.objectives):
            self.logger.info(f"Training model for {obj_name}")

            # Extract single objective
            y = Y[:, i:i + 1]

            # Create GP model
            model = SingleTaskGP(
                X, y,
                outcome_transform=Standardize(m=1)
            )

            # Create marginal log likelihood
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Fit model
            fit_gpytorch_mll(mll)

            self.models[obj_name] = model
            self.logger.info(f"✓ Model trained for {obj_name}")

        # Save models
        model_path = self.models_dir / "surrogate_models.pkl"
        joblib.dump(self.models, model_path)
        self.logger.info(f"Models saved to {model_path}")

    def setup_ax_optimization(self):
        """Setup Ax optimization client."""
        self.logger.info("Setting up Ax optimization...")

        # Create search space
        parameters = []

        # Add parameters for each feature
        for feature in self.features:
            if feature.endswith('_sw'):  # Switch variables (binary)
                parameters.append(
                    ChoiceParameter(
                        name=feature,
                        parameter_type=ParameterType.INT,
                        values=[0, 1]
                    )
                )
            else:  # Continuous variables
                # Determine bounds from data
                min_val = float(self.data[feature].min())
                max_val = float(self.data[feature].max())
                if min_val == max_val:  # Constant feature
                    max_val = min_val + 0.1

                parameters.append(
                    RangeParameter(
                        name=feature,
                        parameter_type=ParameterType.FLOAT,
                        lower=min_val,
                        upper=max_val
                    )
                )

        # Create Ax client
        self.ax_client = AxClient()

        # Create experiment
        self.ax_client.create_experiment(
            name="chemical_formulation_optimization",
            parameters=parameters,
            objectives={
                "Imax": ObjectiveProperties(minimize=False),  # Maximize
                "efficiency": ObjectiveProperties(minimize=False),  # Maximize
                "Lag_time": ObjectiveProperties(minimize=True)  # Minimize
            }
        )

        self.logger.info("✓ Ax optimization setup complete")

    def generate_candidates(self, n_candidates: int = 10):
        """Generate optimization candidates using acquisition functions."""
        self.logger.info(f"Generating {n_candidates} optimization candidates...")

        candidates = []

        # Get data bounds for optimization
        X_train = torch.tensor(self.data[self.features].values, dtype=torch.float32, device=self.device)
        bounds = torch.stack([X_train.min(dim=0)[0], X_train.max(dim=0)[0]])

        for obj_name, model in self.models.items():
            self.logger.info(f"Optimizing for {obj_name}")

            # Create acquisition function
            if obj_name in ['Imax', 'efficiency']:  # Maximize
                acq_func = ExpectedImprovement(model, best_f=model.train_targets.max())
            else:  # Minimize (Lag_time)
                acq_func = ExpectedImprovement(model, best_f=model.train_targets.min(), maximize=False)

            # Optimize acquisition function
            candidate, acq_value = optimize_acqf(
                acq_func,
                bounds=bounds,
                q=n_candidates,
                num_restarts=20,
                raw_samples=512
            )

            # Convert back to DataFrame format
            candidate_df = pd.DataFrame(
                candidate.detach().cpu().numpy(),
                columns=self.features
            )
            candidate_df['objective'] = obj_name
            candidate_df['acquisition_value'] = acq_value.detach().cpu().numpy()

            candidates.append(candidate_df)

        # Combine all candidates
        all_candidates = pd.concat(candidates, ignore_index=True)

        # Save candidates
        candidates_path = self.outputs_dir / "candidates" / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        candidates_path.parent.mkdir(exist_ok=True)
        all_candidates.to_csv(candidates_path, index=False)

        self.logger.info(f"✓ Generated {len(all_candidates)} candidates saved to {candidates_path}")

        return all_candidates

    def pareto_front_analysis(self):
        """Perform Pareto front analysis for multi-objective optimization."""
        self.logger.info("Performing Pareto front analysis...")

        # Get predictions for all training data
        X_train = torch.tensor(self.data[self.features].values, dtype=torch.float32, device=self.device)

        predictions = {}
        for obj_name, model in self.models.items():
            with torch.no_grad():
                posterior = model.posterior(X_train)
                mean = posterior.mean.cpu().numpy().flatten()
                predictions[obj_name] = mean

        # Create results DataFrame
        results_df = self.data.copy()
        for obj_name, pred in predictions.items():
            results_df[f'{obj_name}_pred'] = pred

        # Find Pareto front (simplified 2D case for Imax vs Lag_time)
        pareto_mask = self.find_pareto_front(
            results_df[['Imax', 'Lag_time']].values,
            maximize=[True, False]  # Maximize Imax, minimize Lag_time
        )

        pareto_solutions = results_df[pareto_mask].copy()
        pareto_solutions['is_pareto'] = True

        # Save Pareto solutions
        pareto_path = self.outputs_dir / "pareto" / f"pareto_front_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pareto_path.parent.mkdir(exist_ok=True)
        pareto_solutions.to_csv(pareto_path, index=False)

        self.logger.info(f"✓ Found {len(pareto_solutions)} Pareto optimal solutions")
        self.logger.info(f"Pareto front saved to {pareto_path}")

        return pareto_solutions

    def find_pareto_front(self, costs, maximize=None):
        """Find Pareto front from cost matrix."""
        if maximize is None:
            maximize = [True] * costs.shape[1]

        # Convert maximization to minimization
        costs_adj = costs.copy()
        for i, should_max in enumerate(maximize):
            if should_max:
                costs_adj[:, i] = -costs_adj[:, i]

        # Find Pareto front
        is_efficient = np.ones(costs.shape[0], dtype=bool)

        for i, c in enumerate(costs_adj):
            if is_efficient[i]:
                # Remove all points that are dominated by c
                is_efficient[is_efficient] = np.any(costs_adj[is_efficient] < c, axis=1)
                is_efficient[i] = True  # Keep c itself

        return is_efficient

    def run_optimization(self, csv_path: Optional[str] = None, n_candidates: int = 10):
        """Run the complete optimization pipeline."""
        self.logger.info("Starting chemical formulation optimization pipeline")

        try:
            # Load and preprocess data
            self.load_data(csv_path)
            X, Y = self.preprocess_data()

            # Train surrogate models
            self.train_surrogate_models(X, Y)

            # Generate optimization candidates
            candidates = self.generate_candidates(n_candidates)

            # Perform Pareto analysis
            pareto_solutions = self.pareto_front_analysis()

            # Summary report
            self.generate_summary_report(candidates, pareto_solutions)

            self.logger.info("✓ Optimization pipeline completed successfully!")

            return {
                'candidates': candidates,
                'pareto_solutions': pareto_solutions,
                'models': self.models
            }

        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise

    def generate_summary_report(self, candidates, pareto_solutions):
        """Generate summary report of optimization results."""
        self.logger.info("Generating summary report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'n_samples': len(self.data),
                'n_features': len(self.features),
                'objectives': self.objectives
            },
            'optimization_results': {
                'n_candidates': len(candidates),
                'n_pareto_solutions': len(pareto_solutions),
                'pareto_objectives': {
                    'Imax_range': [float(pareto_solutions['Imax'].min()),
                                   float(pareto_solutions['Imax'].max())],
                    'Lag_time_range': [float(pareto_solutions['Lag_time'].min()),
                                       float(pareto_solutions['Lag_time'].max())],
                    'efficiency_range': [float(pareto_solutions['efficiency'].min()),
                                         float(pareto_solutions['efficiency'].max())]
                }
            }
        }

        # Save report
        import json
        report_path = self.outputs_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"✓ Summary report saved to {report_path}")

        # Print key findings
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Training data: {len(self.data)} samples")
        print(f"Generated candidates: {len(candidates)}")
        print(f"Pareto optimal solutions: {len(pareto_solutions)}")
        print(f"\nBest Pareto solutions:")
        print(f"  Highest Imax: {pareto_solutions['Imax'].max():.3f}")
        print(f"  Lowest Lag_time: {pareto_solutions['Lag_time'].min():.1f}")
        print(f"  Highest efficiency: {pareto_solutions['efficiency'].max():.1f}%")
        print("=" * 60)


# ================================================================
# test_pipeline.py
"""
Unit tests for the optimization pipeline
"""

import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


class TestChemicalOptimization(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.optimizer = ChemicalFormulationOptimizer(
            data_dir=Path(self.test_dir) / "data",
            models_dir=Path(self.test_dir) / "models",
            outputs_dir=Path(self.test_dir) / "outputs",
            device="cpu"  # Force CPU for testing
        )

        # Create test data
        self.create_test_csv()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def create_test_csv(self):
        """Create test CSV data."""
        np.random.seed(42)

        # Small test dataset
        n_samples = 20
        data = {}

        # Add some binary switch variables
        data['additive1_sw'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        data['additive2_sw'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

        # Add some continuous level variables
        data['additive1_lv'] = np.random.uniform(0, 1, n_samples)
        data['additive2_lv'] = np.random.uniform(0, 1, n_samples)
        data['tmb'] = np.random.uniform(0.2, 0.8, n_samples)

        # Objectives
        data['Imax'] = np.random.uniform(0.5, 2.0, n_samples)
        data['Lag_time'] = np.random.uniform(50, 200, n_samples)
        data['efficiency'] = np.random.uniform(70, 90, n_samples)

        df = pd.DataFrame(data)

        # Save to test directory
        test_data_dir = Path(self.test_dir) / "data" / "raw"
        test_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(test_data_dir / "test_data.csv", index=False)

        self.test_csv_path = test_data_dir / "test_data.csv"

    def test_data_loading(self):
        """Test data loading functionality."""
        self.optimizer.load_data(self.test_csv_path)

        self.assertIsNotNone(self.optimizer.data)
        self.assertEqual(len(self.optimizer.data), 20)
        self.assertIn('Imax', self.optimizer.objectives)
        self.assertIn('Lag_time', self.optimizer.objectives)
        self.assertIn('efficiency', self.optimizer.objectives)

    def test_preprocessing(self):
        """Test data preprocessing."""
        self.optimizer.load_data(self.test_csv_path)
        X, Y = self.optimizer.preprocess_data()

        self.assertEqual(X.shape[0], 20)  # 20 samples
        self.assertEqual(Y.shape[1], 3)  # 3 objectives
        self.assertTrue(torch.is_tensor(X))
        self.assertTrue(torch.is_tensor(Y))

    def test_model_training(self):
        """Test surrogate model training."""
        self.optimizer.load_data(self.test_csv_path)
        X, Y = self.optimizer.preprocess_data()
        self.optimizer.train_surrogate_models(X, Y)

        self.assertEqual(len(self.optimizer.models), 3)
        self.assertIn('Imax', self.optimizer.models)
        self.assertIn('Lag_time', self.optimizer.models)
        self.assertIn('efficiency', self.optimizer.models)

    def test_pareto_front(self):
        """Test Pareto front calculation."""
        # Simple test case
        costs = np.array([
            [1, 10],  # Point A
            [2, 8],  # Point B (dominated by C)
            [2, 6],  # Point C
            [3, 4],  # Point D
            [4, 2]  # Point E
        ])

        pareto_mask = self.optimizer.find_pareto_front(costs, maximize=[True, False])

        # Points A, C, D, E should be on Pareto front
        expected_pareto = [True, False, True, True, True]
        np.testing.assert_array_equal(pareto_mask, expected_pareto)

    def test_full_pipeline(self):
        """Test the complete optimization pipeline."""
        try:
            results = self.optimizer.run_optimization(
                csv_path=self.test_csv_path,
                n_candidates=5
            )

            self.assertIn('candidates', results)
            self.assertIn('pareto_solutions', results)
            self.assertIn('models', results)

            # Check that we got some results
            self.assertGreater(len(results['candidates']), 0)
            self.assertGreater(len(results['pareto_solutions']), 0)

        except Exception as e:
            self.fail(f"Full pipeline test failed: {str(e)}")


if __name__ == "__main__":
    # Add missing imports for the main pipeline
    import sys

    sys.path.append('.')

    # Fix import issues
    try:
        from ax.core.parameter import ParameterType
        from ax.core.objective import ObjectiveProperties
    except ImportError:
        # Fallback definitions if imports fail
        class ParameterType:
            INT = "int"
            FLOAT = "float"


        class ObjectiveProperties:
            def __init__(self, minimize=True):
                self.minimize = minimize

    # Run the main pipeline if this script is executed directly
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # Run optimization pipeline
        optimizer = ChemicalFormulationOptimizer(
            data_dir=Path("data"),
            models_dir=Path("models"),
            outputs_dir=Path("outputs")
        )

        try:
            results = optimizer.run_optimization(n_candidates=10)
            print("\n🎉 Optimization completed successfully!")
            print("Check the outputs/ directory for results.")
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            sys.exit(1)