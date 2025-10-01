import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking.client import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_course.config import ProjectConfig, Tags


class FeatureLookUpModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.weather_stations_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_median_temperature"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()
        self.run_id = None

    def create_feature_table(self) -> None:
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name} (
        id STRING NOT NULL, t00 DOUBLE, t01 DOUBLE, t02 DOUBLE, t03 DOUBLE, t04 DOUBLE, t05 DOUBLE, t06 DOUBLE, t07 DOUBLE, t08 DOUBLE, t09 DOUBLE, t10 DOUBLE, t11 DOUBLE, t12 DOUBLE, t13 DOUBLE, t14 DOUBLE, t15 DOUBLE, t16 DOUBLE, t17 DOUBLE, t18 DOUBLE, t19 DOUBLE, t20 DOUBLE, t21 DOUBLE, t22 DOUBLE, t23 DOUBLE, frost_next_day INT
        );
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT weather_data_pk PRIMARY KEY(id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT id, t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, frost_next_day FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT id, t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, frost_next_day FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")


    def define_feature_function(self) -> None:
        """Define a function to calculate the median temparature of the day.

        This function calculates the median temperature across all hourly readings.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(
            t00 DOUBLE, t01 DOUBLE, t02 DOUBLE, t03 DOUBLE, t04 DOUBLE, t05 DOUBLE,
            t06 DOUBLE, t07 DOUBLE, t08 DOUBLE, t09 DOUBLE, t10 DOUBLE, t11 DOUBLE,
            t12 DOUBLE, t13 DOUBLE, t14 DOUBLE, t15 DOUBLE, t16 DOUBLE, t17 DOUBLE,
            t18 DOUBLE, t19 DOUBLE, t20 DOUBLE, t21 DOUBLE, t22 DOUBLE, t23 DOUBLE
        )
        RETURNS FLOAT
        LANGUAGE PYTHON AS
        $$
        import numpy as np
        temps = [t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11,
                t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23]
        return float(np.median([t for t in temps if t is not None]))
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
        # NOTE: (?) The train_set is not transformed to pandas dataframe because we will create training set using FeatureEngineeringClient
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("id", self.train_set["id"].cast("string"))
        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="median_temp",
                    input_bindings={f"t{str(i).zfill(2)}": f"t{str(i).zfill(2)}" for i in range(24)},
                ),
            ],
        )

        self.training_df = self.training_set.load_df().toPandas()
        # Calculate median temperature for test set
        self.test_set["median_temp"] = self.test_set[["t00", "t01", "t02", "t03", "t04", "t05", "t06", "t07", "t08", "t09", "t10", "t11", "t12", "t13", "t14", "t15", "t16", "t17", "t18", "t19", "t20", "t21", "t22", "t23"]].median(axis=1)

        # Define features for training
        self.X_train = self.training_df[self.num_features + ["median_temp"]]
        self.y_train = self.training_df[self.target]

        # Define features for testing
        self.X_test = self.test_set[self.num_features + ["median_temp"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow."""
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), self.num_features)], remainder="drop"
        )

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            logger.info(f"✅ Run ID: {self.run_id}")
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            y_prob = pipeline.predict_proba(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_prob[:, 1])

            logger.info(f"📊 Accuracy: {accuracy:.4f}")
            logger.info(f"📊 Precision: {precision:.4f}")
            logger.info(f"📊 Recall: {recall:.4f}")
            logger.info(f"📊 F1 Score: {f1:.4f}")
            logger.info(f"📊 ROC AUC: {auc:.4f}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "RandomForest Classifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)  # NOTE: This should be y_pred?
            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="randomforest-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )


    def register_model(self) -> str:
        """Register the model in MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/randomforest-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.randomforest_pipeline_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.randomforest_pipeline_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.randomforest_pipeline_model_fe@latest-model"
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions

    def update_feature_table(self) -> None:
        """Update the weather_stations_features table with the latest records from train and test sets.

        Executes SQL queries to insert new records based on timestamp.
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.catalog_name}.{self.schema_name}.train_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Id, t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23
            FROM {self.catalog_name}.{self.schema_name}.train_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.catalog_name}.{self.schema_name}.test_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Id, t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23
            FROM {self.catalog_name}.{self.schema_name}.test_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("Weather stations features table updated successfully.")

    def model_improved(self, test_set: DataFrame) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using accuracy metrics.
        :param test_set: DataFrame containing the test data.
        :return: True if the current model performs better, False otherwise.
        """
        X_test = test_set.drop(self.target)

        # Load predictions from latest registered model
        predictions_latest = self.load_latest_model_and_predict(X_test).withColumnRenamed(
            "prediction", "prediction_latest"
        )

        # Load predictions from current model
        current_model_uri = f"runs:/{self.run_id}/randomforest-pipeline-model-fe"
        predictions_current = self.fe.score_batch(
            model_uri=current_model_uri, df=X_test
        ).withColumnRenamed("prediction", "prediction_current")

        # Select only the ID and target columns from test set
        test_set = test_set.select("id", self.target)

        logger.info("Predictions are ready.")

        # Join the DataFrames on the 'id' column
        df = test_set.join(predictions_current, on="id").join(predictions_latest, on="id")

        # Calculate accuracy for each model
        df = df.withColumn(
            "correct_current",
            F.when(F.col(self.target) == F.col("prediction_current"), 1).otherwise(0)
        )
        df = df.withColumn(
            "correct_latest",
            F.when(F.col(self.target) == F.col("prediction_latest"), 1).otherwise(0)
        )

        # Calculate the accuracy for each model
        accuracy_current = df.agg(F.mean("correct_current")).collect()[0][0]
        accuracy_latest = df.agg(F.mean("correct_latest")).collect()[0][0]

        # Compare models based on accuracy
        logger.info(f"Accuracy for Current Model: {accuracy_current}")
        logger.info(f"Accuracy for Latest Model: {accuracy_latest}")

        if accuracy_current > accuracy_latest:
            logger.info("Current Model performs better.")
            return True
        else:
            logger.info("Current Model performs worse or the same.")
            return False
