import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
import warnings
warnings.filterwarnings("ignore")

def evaluate_and_register_models(X_test, y_test, model_dir="models"):
    mlflow.set_experiment("insurance_model_evaluation")
    client = MlflowClient()

    metrics_list = []
    best_r2 = -np.inf
    best_model_version = None
    best_registered_name = None

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    for model_file in os.listdir(model_dir):
        if not model_file.endswith(".pkl") or "scaler" in model_file.lower():
            continue

        model_name = model_file.replace(".pkl", "")
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)

        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metrics({
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2
            })

            # Save prediction plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual Charges")
            ax.set_ylabel("Predicted Charges")
            ax.set_title(f"Actual vs Predicted ({model_name})")
            plot_path = f"{model_dir}/{model_name}_pred_plot.png"
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)

            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            # Register model correctly using `name=...` instead of deprecated `artifact_path`
            signature = infer_signature(X_test, y_pred)
            input_example = X_test.iloc[:1]

            try:
                mlflow.set_tag("model_name", model_name)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=f"insurance_model_{model_name}",  # new recommended argument
                    registered_model_name=f"insurance_model_{model_name}",
                    signature=signature,
                    input_example=input_example
                )
            except MlflowException as e:
                print(f"⚠️ Failed to register model {model_name}: {e}")
                continue

            # Retrieve latest model version and track best
            try:
                latest_versions = client.get_latest_versions(f"insurance_model_{model_name}", stages=["None"])
                if latest_versions:
                    version = latest_versions[-1].version
                    print(f"✅ Registered: insurance_model_{model_name} v{version} | R²: {r2:.4f}")

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_version = version
                        best_registered_name = f"insurance_model_{model_name}"
            except MlflowException as e:
                print(f"⚠️ Could not retrieve model version for {model_name}: {e}")

            metrics_list.append({
                "model": model_name,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse
            })

    # Save model comparison
    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list)
        df_metrics.sort_values(by="R2", ascending=False, inplace=True)
        metrics_csv_path = os.path.join(model_dir, "model_comparison_metrics.csv")
        df_metrics.to_csv(metrics_csv_path, index=False)
        mlflow.log_artifact(metrics_csv_path)
        print(f"📄 Saved model comparison to {metrics_csv_path}")

    # Promote best model
    if best_registered_name and best_model_version:
        try:
            client.set_registered_model_alias(
                name=best_registered_name,
                version=best_model_version,
                alias="champion"
            )
            print(f"\n🏆 Best model: {best_registered_name} v{best_model_version} set as 'champion' | R²: {best_r2:.4f}")
        except MlflowException as e:
            print(f"⚠️ Failed to set alias for best model: {e}")
