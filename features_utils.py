from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql import Window
import sys
from pyspark.sql.functions import udf
from functools import reduce
from pyspark.ml.pipeline import Transformer
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
#if spark version < 2.2 

get_weekday = F.udf(lambda ts: int(ts.strftime('%w')), IntegerType())


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



class VivaldiExtractor(Transformer):
    def __init__(self, inputCol='month', outputCol='season'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("VivaldiExtractor"))

    def copy(extra):
        defaultCopy(extra)
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
              raise Exception('seasonExtractor input type %s did not match input type IntegerType' % field.dataType)
                
    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, 
                             F.when((df[self.inputCol] >= 3), "Spring")
                             .otherwise(F.when((df[self.inputCol] >= 5), "Summer")
                             .otherwise(F.when((df[self.inputCol] <= 9 ), "fall")
                             .otherwise("Winter"))))



class MonthQuarterExtractor(Transformer):
    def __init__(self, inputCol='day', outputCol='monthquarter'):
        self.inputCol = inputCol
        self.outputCol = outputCol


    def this():
        this(Identifiable.randomUID("MonthQuarterExtractor"))

    def copy(extra):
        defaultCopy(extra)
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
              raise Exception('monthQuarterExtractor input type %s did not match input type IntegerType' % field.dataType)
                
    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when((df[self.inputCol] <= 8), 0)
                               .otherwise(F.when((df[self.inputCol] <= 16), 1)
                                .otherwise(F.when((df[self.inputCol] <= 24), 2)
                                 .otherwise(3))))


class DayExtractor(Transformer):
    def __init__(self, inputCol, outputCol='day'):
        self.inputCol = inputCol
        self.outputCol = outputCol


    def this():
        this(Identifiable.randomUID("DayExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
            raise Exception('DayExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.dayofmonth(df[self.inputCol]))


class MonthExtractor(Transformer):
    def __init__(self, inputCol, outputCol='month'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("MonthExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
            raise Exception('MonthExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.month(df[self.inputCol]))

class YearExtractor(Transformer):
    def __init__(self, inputCol, outputCol='year'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("YearExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
                raise Exception('YearExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.year(df[self.inputCol]))


class WeekDayExtractor(Transformer):
    def __init__(self, inputCol, outputCol='weekday'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("WeekDayExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
              raise Exception('WeekDayExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, get_weekday(F.col(self.inputCol)))


class  WeekendExtractor(Transformer):
    def __init__(self, inputCol='weekday', outputCol='weekend'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("WeekendExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
              raise Exception('WeekendExtractor input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when(((df[self.inputCol] == 5) | (df[self.inputCol] == 6)), 1).otherwise(0))

class MonthBeginExtractor(Transformer):
    def __init__(self, inputCol='day', outputCol='monthbegin'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("MonthBeginExtractor"))

    def copy(extra):
        defaultCopy(extra)
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
            raise Exception('MonthBeginExtractor input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when((df[self.inputCol] <= 7), 1).otherwise(0))
        
class MonthEndExtractor(Transformer):
    def __init__(self, inputCol='day', outputCol='monthend'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def this():
        this(Identifiable.randomUID("MonthEndExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
            raise Exception('MonthEndExtractor input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when((df[self.inputCol] >= 24), 1).otherwise(0))
    
    
class YearQuarterExtractor(Transformer):
    def __init__(self, inputCol='month', outputCol='yearquarter'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("YearQuarterExtractor"))

    def copy(extra):
        defaultCopy(extra)
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
              raise Exception('YearQuarterExtractor input type %s did not match input type IntegerType' % field.dataType)
                
    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when((df[self.inputCol] <= 3), 0)
                               .otherwise(F.when((df[self.inputCol] <= 6), 1)
                                .otherwise(F.when((df[self.inputCol] <= 9), 2)
                                 .otherwise(3))))
    
    
class YearDayExtractor(Transformer):
    def __init__(self, inputCol, outputCol='yearday'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("YearDayExtractor"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
              raise Exception('YearDayExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.dayofyear(F.col(self.inputCol)))


class LogTarget(Transformer):
    def __init__(self, inputCol="F_TOTAL_QTY", outputCol='F_TOTAL_QTY'):
        self.inputCol = inputCol
        self.outputCol = outputCol

    def this():
        this(Identifiable.randomUID("LogTarget"))

    def copy(extra):
        defaultCopy(extra)

    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
              raise Exception('LogTarget input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.log1p(F.col(self.inputCol)))