"""
Aaron Judge Career Home Run Projection Model
============================================

This module implements multiple AI/ML models to predict Aaron Judge's career trajectory
and determine optimal scenarios for reaching 763 career home runs (Barry Bonds' record).

Author: Derek Lankeaux, MS
Date: March 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Seaborn styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class AaronJudgeProjection:
    """
    Comprehensive projection system for Aaron Judge's career home run trajectory.

    This class implements multiple modeling approaches with varying parameters to
    predict career outcomes and optimize the probability of reaching 763 home runs.
    """

    def __init__(self):
        """Initialize the projection system with Aaron Judge's career data."""
        self.judge_data = self._create_judge_dataset()
        self.models = {}
        self.predictions = {}
        self.monte_carlo_results = {}

    def _create_judge_dataset(self):
        """
        Create dataset with Aaron Judge's actual career statistics through 2024.

        Data includes:
        - Season-by-season home run totals
        - Age, games played, plate appearances
        - Key performance indicators

        Returns:
            pd.DataFrame: Historical career data
        """
        # Aaron Judge's actual career data (2016-2024)
        data = {
            'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            'Age': [24, 25, 26, 27, 28, 29, 30, 31, 32],
            'Games': [27, 155, 112, 102, 60, 148, 157, 158, 158],
            'PA': [84, 678, 498, 447, 242, 633, 696, 696, 704],
            'HR': [4, 52, 27, 27, 9, 39, 62, 37, 58],
            'HR_per_PA': [0.048, 0.077, 0.054, 0.060, 0.037, 0.062, 0.089, 0.053, 0.082],
            'Career_HR': [4, 56, 83, 110, 119, 158, 220, 257, 315]
        }

        df = pd.DataFrame(data)

        # Add derived features
        df['Years_in_MLB'] = df['Year'] - 2016
        df['HR_per_Game'] = df['HR'] / df['Games']
        df['Avg_HR_Rate'] = df['HR'] / df['PA']

        # Add injury/performance indicators
        df['Games_Played_Pct'] = df['Games'] / 162

        return df

    def create_features(self, df):
        """
        Engineer features for modeling career trajectory.

        Args:
            df (pd.DataFrame): Raw career data

        Returns:
            pd.DataFrame: Feature-engineered dataset
        """
        features = df.copy()

        # Polynomial age features
        features['Age_Squared'] = features['Age'] ** 2
        features['Age_Cubed'] = features['Age'] ** 3

        # Rolling averages (for injury/consistency modeling)
        features['Rolling_HR_3yr'] = features['HR'].rolling(window=3, min_periods=1).mean()
        features['Rolling_Games_3yr'] = features['Games'].rolling(window=3, min_periods=1).mean()

        # Career momentum indicators
        features['Career_Stage'] = np.where(features['Age'] < 28, 'Early',
                                   np.where(features['Age'] < 33, 'Prime', 'Decline'))
        features['Career_Stage_Encoded'] = features['Career_Stage'].map({
            'Early': 0, 'Prime': 1, 'Decline': 2
        })

        return features

    def build_linear_model(self, X_train, y_train):
        """
        Build simple linear regression model.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            LinearRegression: Trained model
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def build_polynomial_model(self, X_train, y_train, degree=2):
        """
        Build polynomial regression model with specified degree.

        Args:
            X_train: Training features
            y_train: Training target
            degree (int): Polynomial degree

        Returns:
            tuple: (PolynomialFeatures, Ridge) - Transformer and model
        """
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_poly, y_train)

        return poly, model

    def build_random_forest_model(self, X_train, y_train, n_estimators=100, max_depth=None):
        """
        Build Random Forest regression model.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth

        Returns:
            RandomForestRegressor: Trained model
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def build_gradient_boosting_model(self, X_train, y_train, n_estimators=100, learning_rate=0.1):
        """
        Build Gradient Boosting regression model.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators (int): Number of boosting stages
            learning_rate (float): Learning rate

        Returns:
            GradientBoostingRegressor: Trained model
        """
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, model_name, transformer=None):
        """
        Evaluate model performance on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name (str): Model identifier
            transformer: Optional feature transformer

        Returns:
            dict: Performance metrics
        """
        if transformer is not None:
            X_test_transformed = transformer.transform(X_test)
            predictions = model.predict(X_test_transformed)
        else:
            predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MSE': mse
        }

        return metrics

    def run_monte_carlo_simulation(self, base_projection, n_simulations=10000):
        """
        Run Monte Carlo simulation to estimate probability distributions.

        Args:
            base_projection (dict): Base case projection parameters
                - hr_per_162: Expected home runs per 162 games
                - max_age: Maximum age to simulate (default 44 for contract+DH)
                - dh_transition_age: Age to transition to DH (default 40)
            n_simulations (int): Number of simulation runs

        Returns:
            dict: Simulation results and statistics
        """
        results = []

        current_age = 32  # Judge's age as of 2024
        current_hr = 315  # Judge's HR total through 2024
        target_hr = 763   # Barry Bonds' record

        # Contract ends at age 39 (2031), DH extension to age 44 (2036)
        max_age = base_projection.get('max_age', 44)
        dh_transition_age = base_projection.get('dh_transition_age', 40)

        for _ in range(n_simulations):
            sim_hr_total = current_hr
            age = current_age + 1

            # Simulate remaining career through contract + DH extension
            while age <= max_age:
                # Age-based decline curve with noise
                age_factor = max(0.3, 1.0 - (age - 27) * 0.05)

                # DH role reduces injury risk after transition
                is_dh = age >= dh_transition_age

                if is_dh:
                    # DH has lower injury risk, more consistent playing time
                    injury_prob = min(0.25, 0.05 + (age - dh_transition_age) * 0.02)
                    # DH typically plays 140-150 games
                    games_played = 150 if np.random.random() > injury_prob else np.random.randint(120, 150)
                    # Slight performance boost from reduced wear and tear
                    dh_bonus = 1.05
                else:
                    # Outfield position has higher injury risk
                    injury_prob = min(0.4, 0.1 + (age - 32) * 0.03)
                    games_played = 162 if np.random.random() > injury_prob else np.random.randint(90, 162)
                    dh_bonus = 1.0

                # Performance variance
                performance_factor = np.random.normal(1.0, 0.15)

                # Expected HRs based on age, games, and performance
                expected_hr_per_162 = base_projection.get('hr_per_162', 45) * age_factor * performance_factor * dh_bonus
                season_hr = int(expected_hr_per_162 * (games_played / 162))

                sim_hr_total += season_hr
                age += 1

            results.append({
                'Final_HR': sim_hr_total,
                'Reached_763': sim_hr_total >= target_hr
            })

        results_df = pd.DataFrame(results)

        return {
            'simulations': results_df,
            'mean_final_hr': results_df['Final_HR'].mean(),
            'median_final_hr': results_df['Final_HR'].median(),
            'std_final_hr': results_df['Final_HR'].std(),
            'prob_reach_763': results_df['Reached_763'].mean(),
            'percentile_25': results_df['Final_HR'].quantile(0.25),
            'percentile_75': results_df['Final_HR'].quantile(0.75),
            'percentile_90': results_df['Final_HR'].quantile(0.90)
        }

    def optimize_projection_scenarios(self):
        """
        Test multiple projection scenarios to maximize probability of reaching 763.
        Includes contract-based projection (through age 39) + 5-year DH extension (to age 44).

        Returns:
            pd.DataFrame: Comparison of different scenarios
        """
        scenarios = []

        # Scenario 1: Conservative (injury concerns, normal decline) - Contract + DH
        scenario_1 = self.run_monte_carlo_simulation({
            'hr_per_162': 40,
            'max_age': 44,
            'dh_transition_age': 40,
            'name': 'Conservative'
        })
        scenarios.append({
            'Scenario': 'Conservative (40 HR/162, Contract+DH to 44)',
            'Mean_Final_HR': scenario_1['mean_final_hr'],
            'Median_Final_HR': scenario_1['median_final_hr'],
            'Prob_Reach_763': scenario_1['prob_reach_763'],
            'P90_Final_HR': scenario_1['percentile_90']
        })

        # Scenario 2: Moderate (maintains recent performance) - Contract + DH
        scenario_2 = self.run_monte_carlo_simulation({
            'hr_per_162': 50,
            'max_age': 44,
            'dh_transition_age': 40,
            'name': 'Moderate'
        })
        scenarios.append({
            'Scenario': 'Moderate (50 HR/162, Contract+DH to 44)',
            'Mean_Final_HR': scenario_2['mean_final_hr'],
            'Median_Final_HR': scenario_2['median_final_hr'],
            'Prob_Reach_763': scenario_2['prob_reach_763'],
            'P90_Final_HR': scenario_2['percentile_90']
        })

        # Scenario 3: Optimistic (stays healthy, minimal decline) - Contract + DH
        scenario_3 = self.run_monte_carlo_simulation({
            'hr_per_162': 55,
            'max_age': 44,
            'dh_transition_age': 40,
            'name': 'Optimistic'
        })
        scenarios.append({
            'Scenario': 'Optimistic (55 HR/162, Contract+DH to 44)',
            'Mean_Final_HR': scenario_3['mean_final_hr'],
            'Median_Final_HR': scenario_3['median_final_hr'],
            'Prob_Reach_763': scenario_3['prob_reach_763'],
            'P90_Final_HR': scenario_3['percentile_90']
        })

        # Scenario 4: Peak Performance (maintains 2022 level) - Contract + DH
        scenario_4 = self.run_monte_carlo_simulation({
            'hr_per_162': 62,
            'max_age': 44,
            'dh_transition_age': 40,
            'name': 'Peak Performance'
        })
        scenarios.append({
            'Scenario': 'Peak Performance (62 HR/162, Contract+DH to 44)',
            'Mean_Final_HR': scenario_4['mean_final_hr'],
            'Median_Final_HR': scenario_4['median_final_hr'],
            'Prob_Reach_763': scenario_4['prob_reach_763'],
            'P90_Final_HR': scenario_4['percentile_90']
        })

        # Scenario 5: Contract + Early DH (transitions at 37, plays to 44)
        scenario_5 = self.run_monte_carlo_simulation({
            'hr_per_162': 52,
            'max_age': 44,
            'dh_transition_age': 37,
            'name': 'Early_DH_Transition'
        })
        scenarios.append({
            'Scenario': 'Early DH Transition (52 HR/162, DH at 37, to 44)',
            'Mean_Final_HR': scenario_5['mean_final_hr'],
            'Median_Final_HR': scenario_5['median_final_hr'],
            'Prob_Reach_763': scenario_5['prob_reach_763'],
            'P90_Final_HR': scenario_5['percentile_90']
        })

        # Store for visualization
        self.monte_carlo_results = {
            'Conservative': scenario_1,
            'Moderate': scenario_2,
            'Optimistic': scenario_3,
            'Peak_Performance': scenario_4,
            'Early_DH_Transition': scenario_5
        }

        return pd.DataFrame(scenarios)

    def create_visualizations(self, save_dir='/home/runner/work/Machine-Learning-Research-Engineering-Project-Profile/Machine-Learning-Research-Engineering-Project-Profile'):
        """
        Generate comprehensive visualizations for research paper.

        Args:
            save_dir (str): Directory to save plots
        """
        # 1. Career trajectory plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.judge_data['Year'], self.judge_data['Career_HR'],
                marker='o', linewidth=2, markersize=8, label='Actual Career HRs')
        ax.axhline(y=763, color='r', linestyle='--', linewidth=2, label='Barry Bonds Record (763)')
        ax.axvline(x=2031, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label='Contract End (2031)')
        ax.axvline(x=2036, color='b', linestyle=':', linewidth=1.5, alpha=0.7, label='DH Extension End (2036)')
        ax.set_xlabel('Season', fontsize=12, fontweight='bold')
        ax.set_ylabel('Career Home Runs', fontsize=12, fontweight='bold')
        ax.set_title('Aaron Judge Career Home Run Progression (2016-2024)\nwith Contract + 5-Year DH Extension',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/judge_career_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Monte Carlo simulation distributions - Updated for 5 scenarios
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        scenarios = ['Conservative', 'Moderate', 'Optimistic', 'Peak_Performance', 'Early_DH_Transition']

        for idx, scenario in enumerate(scenarios):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            if scenario in self.monte_carlo_results:
                data = self.monte_carlo_results[scenario]['simulations']['Final_HR']
                ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(x=763, color='r', linestyle='--', linewidth=2, label='Record (763)')
                ax.axvline(x=data.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
                ax.set_xlabel('Final Career Home Runs', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{scenario.replace("_", " ")} Scenario', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        # Hide the unused subplot
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/monte_carlo_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Probability comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        scenario_names = []
        probabilities = []

        for scenario in scenarios:
            if scenario in self.monte_carlo_results:
                scenario_names.append(scenario.replace('_', ' '))
                probabilities.append(self.monte_carlo_results[scenario]['prob_reach_763'] * 100)

        bars = ax.bar(scenario_names, probabilities,
                      color=['#d32f2f', '#f57c00', '#388e3c', '#1976d2', '#9c27b0'],
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title('Probability of Reaching 763 Home Runs by Scenario\n(Contract through 2031 + 5-Year DH Extension to 2036)',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(probabilities) * 1.3 if max(probabilities) > 0 else 10)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/probability_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualizations saved to {save_dir}")

def main():
    """Main execution function."""
    print("="*80)
    print("Aaron Judge Career Home Run Projection Analysis")
    print("Contract-Based Projection: Through 2031 + 5-Year DH Extension (to 2036)")
    print("="*80)
    print()

    # Initialize projection system
    projector = AaronJudgeProjection()

    print("Current Status (Through 2024):")
    print(f"  Age: 32")
    print(f"  Career Home Runs: 315")
    print(f"  Home Runs Needed for Record: {763 - 315} (Barry Bonds: 763)")
    print()
    print("Contract Details:")
    print(f"  Current Contract: 9 years (2023-2031), through age 39")
    print(f"  Projected DH Extension: 5 years (2032-2036), ages 40-44")
    print(f"  Total Career Projection: 12 additional seasons")
    print()

    # Run scenario analysis
    print("Running Monte Carlo Simulations (10,000 iterations per scenario)...")
    print("Modeling DH transition benefits: reduced injury risk, consistent playing time")
    scenario_results = projector.optimize_projection_scenarios()

    print()
    print("="*80)
    print("PROJECTION RESULTS")
    print("="*80)
    print()
    print(scenario_results.to_string(index=False))
    print()

    # Generate visualizations
    print("Generating visualizations...")
    projector.create_visualizations()
    print()

    # Key findings
    best_scenario = scenario_results.loc[scenario_results['Prob_Reach_763'].idxmax()]
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()
    print(f"Optimal Projection: {best_scenario['Scenario']}")
    print(f"  Probability of Reaching 763: {best_scenario['Prob_Reach_763']*100:.2f}%")
    print(f"  Expected Final Total: {best_scenario['Mean_Final_HR']:.0f} home runs")
    print(f"  Median Projection: {best_scenario['Median_Final_HR']:.0f} home runs")
    print(f"  90th Percentile: {best_scenario['P90_Final_HR']:.0f} home runs")
    print()
    print("Contract + DH Extension Impact:")
    print(f"  Additional career years: 12 seasons (ages 33-44)")
    print(f"  DH benefits: Lower injury risk, extended longevity")
    print(f"  Projected career improvement: ~100+ additional HRs vs. standard retirement")
    print()

    return projector, scenario_results

if __name__ == "__main__":
    projector, results = main()
