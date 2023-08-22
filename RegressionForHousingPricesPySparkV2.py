from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler, Imputer, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from itertools import chain

# Create Spark session
spark = SparkSession.builder.appName("housing").getOrCreate()

# Load data from AWS S3
housing_df = spark.read.csv("s3://awsbucket-achen33/housing.csv", header=True, inferSchema=True)

# Assuming the following mapping for 'ocean_proximity' (you can change this accordingly)
myMapCat = {
    'NEAR BAY': 1,
    'INLAND': 2,
    '<1H OCEAN': 3,
    'NEAR OCEAN': 4,
    'ISLAND': 5
}

map_expr = F.create_map([F.lit(x) for x in chain(*myMapCat.items())])
housing_df = housing_df.withColumn("ocean_proximity", map_expr[housing_df["ocean_proximity"]])

# Handle Missing Values
imputer = Imputer(strategy="median", inputCols=["total_bedrooms"], outputCols=["total_bedrooms"])
housing_df = imputer.fit(housing_df).transform(housing_df)

# Feature Scaling
features = [col_name for col_name in housing_df.columns if col_name != "median_house_value" and col_name != "ocean_proximity"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Train-Test Split
train_data, test_data = housing_df.randomSplit([0.7, 0.3], seed=0)

# Linear Regression
lin_reg = LinearRegression(featuresCol="scaled_features", labelCol="median_house_value")
pipeline_lin = Pipeline(stages=[assembler, scaler, lin_reg])
model_lin = pipeline_lin.fit(train_data)
predictions_lin = model_lin.transform(test_data)

# Decision Tree Regression
dt_reg = DecisionTreeRegressor(featuresCol="scaled_features", labelCol="median_house_value")
pipeline_dt = Pipeline(stages=[assembler, scaler, dt_reg])
model_dt = pipeline_dt.fit(train_data)
predictions_dt = model_dt.transform(test_data)

# RandomForest
rf_reg = RandomForestRegressor(featuresCol="scaled_features", labelCol="median_house_value", numTrees=100)
pipeline_rf = Pipeline(stages=[assembler, scaler, rf_reg])
model_rf = pipeline_rf.fit(train_data)
predictions_rf = model_rf.transform(test_data)

evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction")

# Evaluate Linear Regression
r2_lin = evaluator.evaluate(predictions_lin, {evaluator.metricName: "r2"})
rmse_lin = evaluator.evaluate(predictions_lin, {evaluator.metricName: "rmse"})
print("Linear Regression - R-squared:", r2_lin)
print("Linear Regression - Root Mean Squared Error:", rmse_lin)

# Evaluate Decision Tree
r2_dt = evaluator.evaluate(predictions_dt, {evaluator.metricName: "r2"})
rmse_dt = evaluator.evaluate(predictions_dt, {evaluator.metricName: "rmse"})
print("Decision Tree - R-squared:", r2_dt)
print("Decision Tree - Root Mean Squared Error:", rmse_dt)

# Evaluate Random Forest
r2_rf = evaluator.evaluate(predictions_rf, {evaluator.metricName: "r2"})
rmse_rf = evaluator.evaluate(predictions_rf, {evaluator.metricName: "rmse"})
print("Random Forest - R-squared:", r2_rf)
print("Random Forest - Root Mean Squared Error:", rmse_rf)

# Close Spark session
spark.stop()
