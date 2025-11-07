import pandas as pd
import numpy as np
import mlflow
import yaml
import os
import sys
import argparse
import json
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def setup_directories():
    """Create necessary directories for artifacts"""
    dirs = ['artifacts', 'plots', 'models']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    return {dir_name: os.path.abspath(dir_name) for dir_name in dirs}

def load_or_create_data(data_path, min_samples=100):
    """Load data or create synthetic data if needed"""
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded from file. Shape: {data.shape}")
        if len(data) < min_samples:
            print(f"Adding synthetic data to reach {min_samples} samples")
            synthetic_data = pd.DataFrame({
                'rating': np.random.randint(1, 6, min_samples),
                'review_text': [f"Review {i}" for i in range(min_samples)],
                'category': np.random.randint(0, 2, min_samples),
                'price': np.random.uniform(10, 1000, min_samples)
            })
            data = pd.concat([data, synthetic_data], ignore_index=True)
    except FileNotFoundError:
        print(f"Creating synthetic dataset with {min_samples} samples")
        data = pd.DataFrame({
            'rating': np.random.randint(1, 6, min_samples),
            'review_text': [f"Review {i}" for i in range(min_samples)],
            'category': np.random.randint(0, 2, min_samples),
            'price': np.random.uniform(10, 1000, min_samples)
        })
    return data

def create_visualizations(predictor, test_data, predictions, plots_dir):
    """Create and save all visualizations"""
    plots = {}
    # Feature importance
    try:
        importance = predictor.feature_importance(test_data)
        plt.figure(figsize=(10, 6))
        importance.plot(kind='bar')
        plt.title('Feature Importance')
        plt.tight_layout()
        filepath = os.path.join(plots_dir, 'feature_importance.png')
        plt.savefig(filepath)
        plt.close()
        plots['feature_importance'] = filepath
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")

    # Prediction distribution
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions, bins=30)
        plt.title('Prediction Distribution')
        plt.tight_layout()
        filepath = os.path.join(plots_dir, 'prediction_dist.png')
        plt.savefig(filepath)
        plt.close()
        plots['prediction_distribution'] = filepath
    except Exception as e:
        print(f"Could not create prediction distribution plot: {e}")

    return plots

def train_model(train_data, test_data, label, time_limit, hyperparameters):
    """Train the model and return results"""
    predictor = TabularPredictor(
        label=label,
        path="models/autogluon"
    )
    predictor.fit(
        train_data=train_data,
        time_limit=time_limit,
        hyperparameters=hyperparameters
    )
    predictions = predictor.predict(test_data)
    leaderboard = predictor.leaderboard(test_data, silent=True)
    return predictor, predictions, leaderboard

def main():
    parser = argparse.ArgumentParser(description="Train AutoGluon model with MLflow tracking")
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    dirs = setup_directories()

    # Setup MLflow (point to your running MLflow server)
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "autogluon_tabular_demo"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Config loading error: {e}")
        config = {"label": "rating", "time_limit": 300}

    label = args.label or config.get("label")
    time_limit = args.time_limit or config.get("time_limit")

    # Model hyperparameters - excluding KNN
    hyperparameters = {
        'GBM': {'num_boost_round': 100},
        'CAT': {'iterations': 100},
        'RF': {'n_estimators': 100},
        'XT': {'n_estimators': 100},
        'LR': {},
    }

    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "product_reviews.csv")
    data = load_or_create_data(data_path)

    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        try:
            predictor, predictions, leaderboard = train_model(
                train_data, test_data, label, time_limit, hyperparameters
            )

            plots = create_visualizations(predictor, test_data, predictions, dirs['plots'])

            # Log params & metrics
            mlflow.log_params({
                "label": label,
                "time_limit": time_limit,
                "train_samples": int(len(train_data)),
                "test_samples": int(len(test_data)),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            })

            mlflow.log_metrics({
                "best_score": float(leaderboard["score_val"].max()),
                "avg_score": float(leaderboard["score_val"].mean()),
                "model_count": int(len(leaderboard))
            })

            # Prepare artifacts (strings / serialized)
            summary_obj = predictor.fit_summary()
            leaderboard_csv = leaderboard.to_csv(index=False)
            predictions_df = pd.DataFrame({
                'true': test_data[label].values,
                'predicted': predictions.values if hasattr(predictions, "values") else predictions
            })
            predictions_csv = predictions_df.to_csv(index=False)
            summary_json = json.dumps(summary_obj, indent=2, default=str)

            artifacts_map = {
                "leaderboard.csv": leaderboard_csv,
                "predictions.csv": predictions_csv,
                "model_summary.json": summary_json
            }

            # Write artifact files into artifacts dir
            for name, content in artifacts_map.items():
                filepath = os.path.join(dirs['artifacts'], name)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

            # Write plots already saved into plots dir

            # Print artifact base URI and attempt to upload whole directories
            print("MLflow artifact URI (server base):", mlflow.get_artifact_uri())

            # Log all artifacts directory at once (preferred)
            artifacts_dirpath = dirs['artifacts']
            plots_dirpath = dirs['plots']

            try:
                if os.path.exists(artifacts_dirpath):
                    mlflow.log_artifacts(artifacts_dirpath, artifact_path="artifacts")
                    print("Logged artifacts from:", artifacts_dirpath)
                if os.path.exists(plots_dirpath):
                    mlflow.log_artifacts(plots_dirpath, artifact_path="plots")
                    print("Logged plots from:", plots_dirpath)
            except Exception as e:
                print("Could not log artifacts directories:", e)
                # fallback: attempt logging individual files
                for fname in os.listdir(artifacts_dirpath):
                    fp = os.path.join(artifacts_dirpath, fname)
                    if os.path.isfile(fp):
                        try:
                            mlflow.log_artifact(fp, artifact_path="artifacts")
                        except Exception as e2:
                            print("Could not log artifact file:", fp, e2)

            # Log model files folder if exists
            model_dir = os.path.abspath("models/autogluon")
            if os.path.exists(model_dir):
                try:
                    mlflow.log_artifacts(model_dir, artifact_path="model")
                    print("Logged model files from:", model_dir)
                except Exception as e:
                    print("Warning: could not log model artifacts:", e)

            print("\nTraining completed successfully!")
            print(f"View run at: {mlflow.get_artifact_uri()}/runs/{run_id}")

        except Exception as e:
            print(f"Training error: {e}")
            mlflow.log_param("error", str(e))
            raise e

if __name__ == "__main__":
    main()