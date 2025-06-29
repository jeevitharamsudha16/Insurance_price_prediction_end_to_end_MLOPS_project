import os
import joblib
import mlflow
import mlflow.sklearn
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")

# ✅ Set MLflow tracking URI from environment variable
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ✅ Set experiment name
mlflow.set_experiment("insurance_model_training")

# ✅ Enable sklearn auto-logging
mlflow.sklearn.autolog()


def train_and_save_models(X_train, y_train, model_dir="models"):
    """
    Trains multiple regression models, performs hyperparameter tuning (where applicable),
    logs everything to MLflow, and saves the trained models locally as .pkl files.

    Args:
    - X_train (DataFrame): Training features
    - y_train (Series): Training target values
    - model_dir (str): Directory to save .pkl model files
    """
    os.makedirs(model_dir, exist_ok=True)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "decision_tree": DecisionTreeRegressor(),
        "svm": SVR(),
        "knn": KNeighborsRegressor(),
        "xgboost": XGBRegressor(verbosity=0)
    }

    param_grids = {
        "random_forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
        "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        "decision_tree": {"max_depth": [None, 10, 20]},
        "svm": {"C": [1, 10], "kernel": ["rbf", "linear"]},
        "knn": {"n_neighbors": [3, 5, 7]},
        "xgboost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_training"):
            print(f"🚀 Training: {name}")

            if name in param_grids:
                grid = GridSearchCV(model, param_grids[name], cv=3, scoring="r2", n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                print(f"✅ Best params: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)
                best_model = model

            # Save the model locally
            model_path = os.path.join(model_dir, f"{name}.pkl")
            joblib.dump(best_model, model_path)

            # Log the model artifact to MLflow
            mlflow.log_artifact(model_path)

            print(f"✅ Trained and saved: {name} → {model_path}")
