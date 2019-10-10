from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql import HiveContext, Window, DataFrameWriter, Row
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LinearSVC
import sys
from operator import add
from functools import reduce, wraps
import logging, time
from functools import wraps

from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import *
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np


schema = StructType([
StructField("ID_DAY", DateType()),
StructField("SID_STORE", IntegerType()),
StructField("NB_TICKET", IntegerType()),
StructField("F_TOTAL_QTY", IntegerType()),
StructField("F_VAL_SALES_AMT_TCK", DoubleType()),
StructField("SITE_FORMAT", StringType())])

df = spark.read.csv("gs://my_buck/poland_ks", header = 'true', schema=schema)
#json_to_df = spark.read.json("gs://my_buck/poland_ks_json")
#df = spark.read.format("com.databricks.spark.avro").load("gs://my_buck/poland_ks_avro", header = 'true')

'''

df = df.filter(F.col("F_TOTAL_QTY") > 1000)

df = df.filter(
(F.col("F_TOTAL_QTY") > 0)
& (F.col("ID_DAY").notNull())
)

df = df.filter(
F.col("ID_DAY").between("2018-12-01", "2019-01-01") & (
F.col("SITE_FORMAT").isin(["PRO", "SUP"]))
)


regex_exp = "[0-9]"
df = df.withColumn('dummie_col', F.when(F.col('F_TOTAL_QTY').between(500, 1000), F.lit('My favorite number is')).otherwise(F.lit("777")))
df = df.filter(F.col("dummie_col").rlike(regex_exp))

'''
df = (df
 .withColumn('dayofyear', F.dayofyear(F.col("ID_DAY")))
 .withColumn('Month', F.month(F.col('ID_DAY')))
 .withColumn('ratio_ticket_qty', F.col('F_TOTAL_QTY')/F.col('NB_TICKET'))
 )

# Hive timestamp is interpreted as UNIX timestamp in seconds*


grouped_windows = Window.partitionBy(F.col('SID_STORE'), F.col('Month'))

rolling_windows = (Window.orderBy(F.col("dayofyear").cast(IntegerType())).rangeBetween(-7, 0))

df = (df
 .withColumn('rolling_average', F.avg("F_TOTAL_QTY").over(rolling_windows))
 .withColumn('monthly_qty', F.avg('F_TOTAL_QTY').over(grouped_windows))
 )

month_qty = df.groupBy('SID_STORE', 'Month').agg(F.avg('F_TOTAL_QTY').alias('monthly_qty_by_join'))
df = df.join(month_qty, how = "left", on = ["SID_STORE", "Month"])


#=========================== preprocess ================================
'''df = df.withColumn('F_TOTAL_QTY', F.when(F.col("F_TOTAL_QTY") < 200, F.lit(None)).otherwise(F.col("F_TOTAL_QTY")))
## filter numeric cols
num_cols = [col_type[0] for col_type in filter(lambda dtype: dtype[1] in {"bigint", "double", "int"}, df.dtypes)]
### Compute a dict with <col_name, median_value>
median_dict = dict()
for c in num_cols:
   median_dict[c] = df.stat.approxQuantile(c, [0.5], 0.001)[0]

df = df.na.fill(median_dict)'''

#=========================================================================


df = df.withColumn("day_to_xmas", F.when((F.col("ID_DAY").between("2018-12-01", "2018-12-31")) | (F.col('ID_DAY').between("2019-12-01", "2019-12-31")),
    F.lit('xmas_is_coming')).otherwise(F.datediff(F.col("ID_DAY"), F.lit('2018-12-01').cast(DateType())))
)


@udf(VectorUDT())
def toSparseVector(index, values):
    day_list_index, qty_list_values = zip(*sorted(zip(index, values)))
    #367 for bisextile year (1 to 366 +1)
    return Vectors.sparse(366, day_list_index, qty_list_values)


@udf(FloatType())
def getReference(currentDay, referenceDay):
    #get the reference values
    try :
        if currentDay in referenceDay.indices :
            return float(referenceDay.values[np.where(referenceDay.indices == currentDay)])
    except AttributeError :
        #case when indices is None it'll return no indices
        return None
    #define 2 windows to be sure to catch at decent amount of reference
    windows = [7,14,21]
    #loop over windows if the previous year reference doesnt exist
    for window in windows:
        #catch every day between the span
        in_range_values = referenceDay.values[np.where((referenceDay.indices >= currentDay - window) \
                                                       & (referenceDay.indices <= currentDay + window+1))]
        #check how many values there's in range
        common_count = len(in_range_values)
        #if there's more than 1 do the mean
        if common_count > 0 :
            try :
                return float((sum(in_range_values) / common_count))
            #except none value error
            except AttributeError :
                in_range_values = [x for x in in_range_values if ~np.isnan(x)]
                return float((sum(in_range_values) / common_count))


df = (df
    .withColumn('year', F.year(F.col("ID_DAY")))
    .withColumn('yearday', F.dayofyear(F.col("ID_DAY")))
    )


self_join = (df
    .groupby("SID_STORE", "year")
    .agg(
        F.collect_list("F_TOTAL_QTY").alias("qties"),
         F.collect_list("yearday").alias("days")
         )
    .withColumn("qties_vectorized", toSparseVector(F.col("days"), F.col("qties")))
    .withColumn("year_join", F.col("year") + 1)
     .select(
      F.col("SID_STORE").alias("p_id_store"), F.col("year").alias("year2"), F.col("qties_vectorized").alias("qties_vectorized"),
       F.col("days").alias("dayss"),  F.col("qties").alias("qties"),  F.col("year_join").alias("year_join")
        )
     )


df= (df
    .join(self_join
        , ([self_join.p_id_store == df.SID_STORE, self_join.year_join ==  df.year]),
        how = "left"
        )
        .withColumn("qty_reference", getReference(F.col("yearday"), F.col("qties_vectorized")))
        )


df = (df
    .withColumn("dayofweek", F.date_format("ID_DAY", "E"))
    .withColumn('week', F.date_format("ID_DAY", "W"))
    .withColumn('weekofyear', F.weekofyear(F.col("ID_DAY")))
    )

features = ['SID_STORE', 'NB_TICKET', 'F_TOTAL_QTY', 'F_VAL_SALES_AMT_TCK',
'SITE_FORMAT', 'dayofyear', 'Month', 'yearday',"dayofweek", "week", "weekofyear", 'qty_reference']

df = df.select(*features)

#====================================

'''def extend_dataframe(dataset) :
    """
    some feature engineering
    """
    dex = DayExtractor(inputCol='dt_ticket')
    mex = MonthExtractor(inputCol='dt_ticket')
    yex = YearExtractor(inputCol='dt_ticket')
    wdex = WeekDayExtractor(inputCol='dt_ticket')
    wex = WeekendExtractor()
    vex = VivaldiExtractor(inputCol = "month")
    mqex = MonthQuarterExtractor(inputCol = "day")
    mbex = MonthBeginExtractor()
    meex = MonthEndExtractor()
    yqex = YearQuarterExtractor()
    ydex = YearDayExtractor(inputCol='dt_ticket')
    FeaturesPipeline =  Pipeline(stages=[dex, mex, yex, wdex, wex, meex, vex, mbex, mqex, yqex, ydex])
    return FeaturesPipeline


'''
"""
BEFORE RUNNING PIP CHECK NULL
"""

df = (df
    .withColumn('qty_reference', F.col("qty_reference").cast(IntegerType()))
    .withColumn('qty_reference', F.when(F.col("qty_reference").isNull(), F.lit(0)).otherwise(F.col("qty_reference")))
    )

from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import *


others_col= ['Month', "qty_reference", 'weekofyear', "dayofyear"]
categorical_col= ["SITE_FORMAT", "dayofweek", "SID_STORE",'week']

indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexedd".format(c), handleInvalid = 'error') for c in categorical_col]

encoders = [OneHotEncoder(dropLast=True,inputCol=indexer.getOutputCol(),
    outputCol="{0}_encodedd".format(indexer.getOutputCol())) for indexer in indexers]

assembler = VectorAssembler(inputCols= [encoder.getOutputCol() for encoder in encoders] +others_col, outputCol="Features")

pca = PCA(k=10, inputCol="Features", outputCol="pcaFeatures")

pipeline = Pipeline(stages = indexers + encoders + [assembler] + [pca])

'''fitted_df =pip.fit(df)
df = fitted_df.transform(df)'''

model= pipeline.fit(df)

final_dataset = model.transform(df)


X_train = final_dataset.filter(F.col('ID_DAY').between("2017-01-02", "2018-06-01"))
X_test = final_dataset.filter(F.col('ID_DAY') > "2018-06-01")


target = 'F_TOTAL_QTY'

def algorithm(target) :
    rf = RandomForestRegressor(featuresCol = 'Features', labelCol=target)
    gbt = GBTRegressor(featuresCol = 'Features', labelCol=target)
    dt = DecisionTreeRegressor(featuresCol = 'Features', labelCol=target)
    lr = LinearRegression(featuresCol = 'Features', labelCol=target)
    glr = GeneralizedLinearRegression(family="gaussian", link="identity", featuresCol = 'Features', labelCol=target)
    model = [gbt, dt, lr, glr, rf]
    return rf, gbt, dt, lr, glr, model

X_train = X_train.withColumn(target, F.log1p(F.col(target)))

X_test = X_test.withColumn(target, F.log1p(F.col(target)))

rf, gbt, dt, lr, glr, model = algorithm(target)

fitted = gbt.fit(X_train)

yhat = (fitted.transform(X_test)
    .withColumn("prediction", F.expm1(F.col("prediction")))
    .withColumn(target, F.expm1(F.col(target)))
    .withColumn('fiability', 1 - F.abs(F.col(target) - F.col("prediction"))/F.col(target))
    .withColumn('fiability', F.when(F.col("fiability") <0, 0).otherwise(F.col("fiability")))
    )

print(yhat.select(F.sum(F.col(target)*F.col("fiability"))/F.sum(F.col(target))).show())

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
             .addGrid(gbt.maxBins, [32, 64])
             .build())

cv = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=eval_,
                          numFolds=3)  
cvModel = cv.fit(X_train)


yhat = (cvModel.transform(X_test)
    .withColumn("prediction", F.expm1(F.col("prediction")))
    .withColumn(target, F.expm1(F.col(target)))
    .withColumn('fiability', 1 - F.abs(F.col(target) - F.col("prediction"))/F.col(target))
    .withColumn('fiability', F.when(F.col("fiability") <0, 0).otherwise(F.col("fiability")))
    )

print(yhat.select(F.sum(F.col(target)*F.col("fiability"))/F.sum(F.col(target))).show())

eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="rmse")

rmse = eval_.evaluate(yhat)

print('rmse is %.2f' %rmse)

mae = eval_.evaluate(yhat, {eval_.metricName: "mae"})
print('mae is %.2f' %mae)

r2 = eval_.evaluate(yhat, {eval_.metricName: "r2"})
print('r2 is %.2f' %r2)


fi = fitted.stages[-1].featureImportances.toArray()

import pandas as pd 

feat_imp = (pd.DataFrame(dict(zip(features, fi)), range(1))
  .T.rename(columns={0:'Score'})
  .sort_values("Score", ascending =False)
  )