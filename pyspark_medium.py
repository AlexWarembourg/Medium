from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import *

from features_utils import * 

schema = StructType([
StructField("DATE", DateType()),
StructField("STORE", IntegerType()),
StructField("NUMBERS_OF_TICKETS", IntegerType()),
StructField("QTY", IntegerType()),
StructField("CA", IntegerType()),
StructField("FORMAT", StringType())])

df = spark.read.csv("gs://alex_plannif_ks_poland/poland_ks.csv", header = 'true', schema=schema)
#================== step before ==================

df = (df
    .withColumn('yearday', F.dayofyear(F.col("DATE")))
    .withColumn('year', F.year(F.col('DATE')))
    )

self_join = (df
    .groupby("STORE", "year")
    .agg(
        F.collect_list("QTY").alias("qties"),
         F.collect_list("yearday").alias("days")
         )
    .withColumn("qties_vectorized", toSparseVector(F.col("days"), F.col("qties")))
    .withColumn("year_join", F.col("year") + 1)
     .select(
      F.col("STORE").alias("p_id_store"), F.col("year").alias("year2"), F.col("qties_vectorized").alias("qties_vectorized"),
       F.col("days").alias("dayss"),  F.col("qties").alias("qties"),  F.col("year_join").alias("year_join")
        )
     )


df= (df
    .join(self_join
        , ([self_join.p_id_store == df.STORE, self_join.year_join ==  df.year]),
        how = "left"
        )
        .withColumn("qty_reference", getReference(F.col("yearday"), F.col("qties_vectorized")))
        .withColumn('qty_reference', F.col("qty_reference").cast(DoubleType()))
        .withColumn('qty_reference', F.when(F.col("qty_reference").isNull(), F.lit(0)).otherwise(F.col("qty_reference")))
    )


#================ start pipeline ========================



dex = DayExtractor(inputCol='DATE')
mex = MonthExtractor(inputCol='DATE')
yex = YearExtractor(inputCol='DATE')
wdex = WeekDayExtractor(inputCol='DATE')
wex = WeekendExtractor()
vex = VivaldiExtractor(inputCol = "month")
mqex = MonthQuarterExtractor(inputCol = "day")
mbex = MonthBeginExtractor()
meex = MonthEndExtractor()
yqex = YearQuarterExtractor()
ydex = YearDayExtractor(inputCol='DATE')

numeric_col= ["qty_reference"]


imputer = Imputer(
    inputCols=numeric_col, 
    outputCols=["{}_imputed".format(c) for c in numeric_col]
)

categorical_col= ["SITE_FORMAT", "season"]

indexers = [StringIndexer(
    inputCol=c, outputCol="{0}_indexedd".format(c), 
    handleInvalid = 'skip'
    ) for c in categorical_col]

encoders = [OneHotEncoder(
    dropLast=True,inputCol=indexer.getOutputCol(),
    outputCol="{0}_encodedd".format(indexer.getOutputCol())
    ) for indexer in indexers]

assembler = VectorAssembler(inputCols= [encoder.getOutputCol() for encoder in encoders] + \
 [x +'_imputed' for x in numeric_col] + ['day', 'month', 'weekday', 'weekend', 'monthend', 'monthbegin', 'monthquarter', 'yearquarter'],
  outputCol="Features")

pca = PCA(k=5, inputCol="Features", outputCol="pcaFeatures")

pipeline = Pipeline(stages = [dex, mex, yex, wdex, wex, meex, vex, mbex, mqex, yqex, ydex] + \
    [imputer] + \
    indexers + \
    encoders +  \
    [assembler]+ \
    [pca])

model= pipeline.fit(df)
final_dataset = model.transform(df)

target = 'QTY'

gbt = GBTRegressor(featuresCol = 'Features', labelCol=target)
dt = DecisionTreeRegressor(featuresCol = 'Features', labelCol=target)
lr = LinearRegression(featuresCol = 'Features', labelCol=target)


X_train = (final_dataset.filter(F.col('DATE').between("2017-01-02", "2018-06-01"))
    .withColumn(target, F.log1p(F.col(target)))
    )

X_test = (final_dataset.filter(F.col('DATE') > "2018-06-01")
    .withColumn(target, F.log1p(F.col(target))))


fitted = gbt.fit(X_train)

yhat = (fitted.transform(X_test)
    .withColumn("prediction", F.expm1(F.col("prediction")))
    .withColumn(target, F.expm1(F.col(target)))
    )

eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="rmse")

rmse = eval_.evaluate(yhat)

print('rmse is %.2f' %rmse)

mae = eval_.evaluate(yhat, {eval_.metricName: "mae"})
print('mae is %.2f' %mae)

r2 = eval_.evaluate(yhat, {eval_.metricName: "r2"})
print('r2 is %.2f' %r2)


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [5, 8, 10, 12])
             .addGrid(gbt.maxBins, [64])
             .build())

cv = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=eval_,
                          numFolds=3)  
cvModel = cv.fit(X_train)


yhat = (cvModel.transform(X_test)
    .withColumn("prediction", F.expm1(F.col("prediction")))
    .withColumn(target, F.expm1(F.col(target)))
    )

eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="rmse")

rmse = eval_.evaluate(yhat)

print('rmse is %.2f' %rmse)

mae = eval_.evaluate(yhat, {eval_.metricName: "mae"})
print('mae is %.2f' %mae)

r2 = eval_.evaluate(yhat, {eval_.metricName: "r2"})
print('r2 is %.2f' %r2)


fi = fitted.featureImportances.toArray()

import pandas as pd 

features = [encoder.getOutputCol() for encoder in encoders] + \
 [x +'_imputed' for x in numeric_col] + ['day', 'month', 'weekday', 'weekend', 'monthend', 'monthbegin', 'monthquarter', 'yearquarter']

feat_imp = (pd.DataFrame(dict(zip(features, fi)), range(1))
  .T.rename(columns={0:'Score'})
  .sort_values("Score", ascending =False)
  )
