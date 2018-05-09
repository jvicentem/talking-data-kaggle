import findspark
findspark.init() #Por defecto findspark mira en la variable de entorno del sistema SPARK_HOME

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql import Window as W

def start_spark():
  spark = (SparkSession.builder
          .master('spark://192.168.1.107:7077')
          .appName('kaggle_challenge')
          .config(key = 'spark.driver.cores', value = '4')
          .config(key = 'spark.driver.memory', value = '15G')
          .config(key = 'spark.executor.memory', value = '3456M') 
          .config(key = 'spark.executor.cores', value = '1')
          .config(key = 'spark.shuffle.service.enabled', value = 'true')
          .config(key = 'spark.dynamicAllocation.enabled', value = 'true')
          .config(key = 'spark.dynamicAllocation.minExecutors', value = '2')
          .config(key = 'spark.dynamicAllocation.maxExecutors', value = '6')
          .config(key = 'spark.network.timeout', value = '300s')
          .config(key = 'spark.driver.maxResultSize', value = '14G')
          .getOrCreate()
         )
  #obtenemos el sparkcontext a partir del sparksession
  sc = spark.sparkContext

  return {'spark': spark, 'sc': sc}
  
def do_transformations(train_df, spark):
  train_df = train_df.select('ip', 'app', 'device' ,'os', 'channel', 'click_time', 
                F.date_format('click_time', 'E').alias('click_time_wday'), 
                'attributed_time', F.date_format('attributed_time', 'E').alias('attributed_time_wday'), 
                'is_attributed')

  train_df = train_df.withColumn('click_time_hour', F.date_format('click_time', 'H'))       

  #train_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
  #train_df.show(n = 1, truncate = False)     

  w1 = W.partitionBy('ip', 'device', 'os').orderBy('click_time')

  train_df = train_df.withColumn('prev_value', F.lag('is_attributed').over(w1))

  train_df = train_df.withColumn('times_attributed', F.when(F.isnull( F.sum('prev_value').over(w1) ), 0 ).otherwise( F.sum('prev_value').over(w1) )  )

  w2 = W.partitionBy('ip', 'device', 'os', 'times_attributed').orderBy('click_time')

  train_df = train_df.withColumn('n_previous_clicks', F.row_number().over(w2) - 1)

  train_df = (train_df.withColumn('group_first', F.first('click_time').over(w2))
                      .withColumn('click_time_diff', F.abs( F.col('click_time').cast('long') - F.col('group_first').cast('long') ) / 60 )
             )

  train_df = train_df.drop('group_first').drop('times_attributed').drop('prev_value')

  #train_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
  #train_df.show(n = 1, truncate = False)     

  # custom scores
  aux = (train_df.select('device', 'is_attributed')
                 .groupBy('device')
                 .pivot('is_attributed')
                 .count())

  aux = (aux.withColumn('prcnt_0', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['0'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .withColumn('prcnt_1', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['1'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .na.fill(0)
            .select('device', 'prcnt_1', '1')
            .withColumn('prcnt_1', F.col('prcnt_1') / 100)
            .withColumn('custom_score', F.col('1') * F.col('prcnt_1'))
        )

  # aux = aux.join(train_df.select('device', 'is_attributed'), ['device'], 'inner')

  # aux_device_cat = ( aux.select('device', 'custom_score').withColumn('device_cat', F.when( F.col('custom_score') > 50, 'A' ) )
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') >= 9) & (F.col('custom_score') <= 50), 'B' ) )
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') > 7) & (F.col('custom_score') <= 9), 'C' ) )
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') > 5.24) & (F.col('custom_score') <= 7), 'D' ) )
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') > 4.12) & (F.col('custom_score') <= 5.24), 'E' ))
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') > 2.87) & (F.col('custom_score') <= 4.12), 'F' ))
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') > 1.92) & (F.col('custom_score') <= 2.87), 'G' ))
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') >= 1.2) & (F.col('custom_score') <= 1.92), 'H' ))
  #                   .withColumn('device_cat', F.when( (F.col('custom_score') > 0.5) & (F.col('custom_score') < 1.2), 'I' ).otherwise('J'))                  
  #              )

  train_df = train_df.join(aux.select('device', 'custom_score').withColumnRenamed('custom_score', 'device_custom_score'), ['device'], 'inner')

  #train_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
  #train_df.show(n = 1, truncate = False)       

  aux = (train_df.select('os', 'is_attributed')
                 .groupBy('os')
                 .pivot('is_attributed')
                 .count())

  aux = (aux.withColumn('prcnt_0', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['0'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .withColumn('prcnt_1', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['1'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .na.fill(0)
            .select('os', 'prcnt_1', '1')
            .withColumn('prcnt_1', F.col('prcnt_1') / 100)
            .withColumn('custom_score', F.col('1') * F.col('prcnt_1'))
        )

  # aux_os_cat = ( aux.select('os', 'custom_score').withColumn('os_cat', F.when( F.col('custom_score') > 105, 'A' ) )
  #                   .withColumn('os_cat', F.when( (F.col('custom_score') > 60) & (F.col('custom_score') <= 105), 'B' ) )
  #                   .withColumn('os_cat', F.when( (F.col('custom_score') > 35) & (F.col('custom_score') <= 60), 'C' ) )
  #                   .withColumn('os_cat', F.when( (F.col('custom_score') >= 10) & (F.col('custom_score') <= 35), 'D' ) )
  #                   .withColumn('os_cat', F.when( (F.col('custom_score') >= 1) & (F.col('custom_score') < 10), 'E' ).otherwise('F') )                  
  #              )

  train_df = train_df.join(aux.select('os', 'custom_score').withColumnRenamed('custom_score', 'os_custom_score'), ['os'], 'inner')

  #train_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
  #train_df.show(n = 1, truncate = False)     

  aux = (train_df.select('app', 'is_attributed')
                 .groupBy('app')
                 .pivot('is_attributed')
                 .count())

  aux = (aux.withColumn('prcnt_0', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['0'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .withColumn('prcnt_1', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['1'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .na.fill(0)
            .select('app', 'prcnt_1', '1')
            .withColumn('prcnt_1', F.col('prcnt_1') / 100)
            .withColumn('custom_score', F.col('1') * F.col('prcnt_1'))
        )

  # aux_app_cat = ( aux.select('app', 'custom_score').withColumn('app_cat', F.when( F.col('custom_score') >= 481, 'A' ) )
  #                   .withColumn('app_cat', F.when( (F.col('custom_score') >= 111) & (F.col('custom_score') <= 214), 'B' ) )
  #                   .withColumn('app_cat', F.when( (F.col('custom_score') > 23) & (F.col('custom_score') <= 51), 'C' ) )
  #                   .withColumn('app_cat', F.when( (F.col('custom_score') >= 4) & (F.col('custom_score') <= 13), 'D' ) )
  #                   .withColumn('app_cat', F.when( (F.col('custom_score') >= 0.6) & (F.col('custom_score') < 3), 'E' ).otherwise('F') )                  
  #              )

  train_df = train_df.join(aux.select('app', 'custom_score').withColumnRenamed('custom_score', 'app_custom_score'), ['app'], 'inner')

  #train_df.persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)
  #train_df.show(n = 1, truncate = False)       

  aux = (train_df.select('channel', 'is_attributed')
                 .groupBy('channel')
                 .pivot('is_attributed')
                 .count())

  aux = (aux.withColumn('prcnt_0', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['0'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .withColumn('prcnt_1', F.when(~F.isnull('0') & ~F.isnull('1'), (aux['1'] / (aux['0'] + aux['1'])) * 100).otherwise(None))
            .na.fill(0)
            .select('channel', 'prcnt_1', '1')
            .withColumn('prcnt_1', F.col('prcnt_1') / 100)
            .withColumn('custom_score', F.col('1') * F.col('prcnt_1'))
        )

  # aux_channel_cat = ( aux.select('channel', 'custom_score').withColumn('channel_cat', F.when( F.col('custom_score') >= 4279, 'A' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('custom_score') >= 1319) & (F.col('custom_score') <= 2356), 'B' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('custom_score') > 368) & (F.col('custom_score') <= 627), 'C' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('custom_score') >= 52) & (F.col('custom_score') <= 202), 'D' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('custom_score') >= 3) & (F.col('custom_score') < 26), 'E' ).otherwise('F') )                  
  #                  )

  return train_df.join(aux.select('channel', 'custom_score').withColumnRenamed('custom_score', 'channel_custom_score'), ['channel'], 'inner')

# Para submission las siguientes funciones

def prepare_submission_file(submission_df, spark):
  os_score_df = spark.read.csv('file:/C:/os_score_pd.csv', 
                 header = True, 
                 mode = 'DROPMALFORMED', 
                 schema = StructType([StructField('os', StringType()), 
                                      StructField('prcnt_1', DoubleType()), 
                                      StructField('1', IntegerType()),
                                      StructField('custom_score', DoubleType())
                                     ]),
                 timestampFormat='yyyy-MM-dd HH:mm:ss'
                )  

  app_score_df = spark.read.csv('file:/C:/app_score_weight.csv', 
                 header = True, 
                 mode = 'DROPMALFORMED', 
                 schema = StructType([StructField('app', StringType()), 
                                      StructField('custom_score', DoubleType()), 
                                      StructField('count', IntegerType())
                                     ]),
                 timestampFormat='yyyy-MM-dd HH:mm:ss'
                )  

  channel_score_df = spark.read.csv('file:/C:/channel_score_weight.csv', 
                 header = True, 
                 mode = 'DROPMALFORMED', 
                 schema = StructType([StructField('channel', StringType()), 
                                      StructField('custom_score', DoubleType()), 
                                      StructField('count', IntegerType())
                                     ]),
                 timestampFormat='yyyy-MM-dd HH:mm:ss'
                )                    

  devices_score_df = spark.read.csv('file:/C:/devices_score_pd.csv', 
                 header = True, 
                 mode = 'DROPMALFORMED', 
                 schema = StructType([StructField('device', StringType()), 
                                      StructField('prcnt_1', DoubleType()), 
                                      StructField('1', IntegerType()),
                                      StructField('custom_score', DoubleType())
                                     ]),
                 timestampFormat='yyyy-MM-dd HH:mm:ss'
                )           

  submission_df = submission_df.select('click_id', 'ip', 'app', 'device' ,'os', 'channel', 'click_time', 
              F.date_format('click_time', 'E').alias('click_time_wday'))  

  submission_df = submission_df.withColumn('click_time_hour', F.date_format('click_time', 'H'))       

  w1 = W.partitionBy('ip', 'device', 'os').orderBy('click_time')

  submission_df = submission_df.withColumn('times_attributed', F.sum(F.lit(1)).over(w1) )

  w2 = W.partitionBy('ip', 'device', 'os', 'times_attributed').orderBy('click_time')

  submission_df = submission_df.withColumn('n_previous_clicks', F.row_number().over(w2) - 1)

  submission_df = (submission_df.withColumn('group_first', F.first('click_time').over(w2))
                      .withColumn('click_time_diff', F.abs( F.col('click_time').cast('long') - F.col('group_first').cast('long') ) / 60 )
                  )

  submission_df = submission_df.drop('group_first').drop('times_attributed')

  submission_df = submission_df.join(os_score_df.select('os', 'custom_score').withColumnRenamed('custom_score', 'os_custom_score'), ['os'], 'inner').drop('os')

  #submission_df.show()

  submission_df = submission_df.join(app_score_df.select('app', 'custom_score').withColumnRenamed('custom_score', 'app_custom_score'), ['app'], 'inner').drop('app')

  #submission_df.show()

  submission_df = submission_df.join(channel_score_df.select('channel', 'custom_score').withColumnRenamed('custom_score', 'channel_custom_score'), ['channel'], 'inner').drop('channel')

  #submission_df.show()

  submission_df = submission_df.join(devices_score_df.select('device', 'custom_score').withColumnRenamed('custom_score', 'device_custom_score'), ['device'], 'inner').drop('device')

  #submission_df.show()

  # submission_df = ( submission_df.withColumn('channel_cat', F.when( F.col('channel_custom_score') >= 4279, 'A' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('channel_custom_score') >= 1319) & (F.col('channel_custom_score') <= 2356), 'B' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('channel_custom_score') > 368) & (F.col('channel_custom_score') <= 627), 'C' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('channel_custom_score') >= 52) & (F.col('channel_custom_score') <= 202), 'D' ) )
  #                   .withColumn('channel_cat', F.when( (F.col('channel_custom_score') >= 3) & (F.col('channel_custom_score') < 26), 'E' ).otherwise('F') )                  
  #                 )  

  # submission_df = ( submission_df.withColumn('app_cat', F.when( F.col('app_custom_score') >= 481, 'A' ) )
  #                   .withColumn('app_cat', F.when( (F.col('app_custom_score') >= 111) & (F.col('app_custom_score') <= 214), 'B' ) )
  #                   .withColumn('app_cat', F.when( (F.col('app_custom_score') > 23) & (F.col('app_custom_score') <= 51), 'C' ) )
  #                   .withColumn('app_cat', F.when( (F.col('app_custom_score') >= 4) & (F.col('app_custom_score') <= 13), 'D' ) )
  #                   .withColumn('app_cat', F.when( (F.col('app_custom_score') >= 0.6) & (F.col('app_custom_score') < 3), 'E' ).otherwise('F') )                  
  #                 )

  # submission_df = ( submission_df.withColumn('os_cat', F.when( F.col('os_custom_score') > 105, 'A' ) )
  #                   .withColumn('os_cat', F.when( (F.col('os_custom_score') > 60) & (F.col('os_custom_score') <= 105), 'B' ) )
  #                   .withColumn('os_cat', F.when( (F.col('os_custom_score') > 35) & (F.col('os_custom_score') <= 60), 'C' ) )
  #                   .withColumn('os_cat', F.when( (F.col('os_custom_score') >= 10) & (F.col('os_custom_score') <= 35), 'D' ) )
  #                   .withColumn('os_cat', F.when( (F.col('os_custom_score') >= 1) & (F.col('os_custom_score') < 10), 'E' ).otherwise('F') )                  
  #                 )

  # submission_df = ( submission_df.withColumn('device_cat', F.when( F.col('device_custom_score') > 50, 'A' ) )
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') >= 9) & (F.col('device_custom_score') <= 50), 'B' ) )
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') > 7) & (F.col('device_custom_score') <= 9), 'C' ) )
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') > 5.24) & (F.col('device_custom_score') <= 7), 'D' ) )
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') > 4.12) & (F.col('device_custom_score') <= 5.24), 'E' ))
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') > 2.87) & (F.col('device_custom_score') <= 4.12), 'F' ))
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') > 1.92) & (F.col('device_custom_score') <= 2.87), 'G' ))
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') >= 1.2) & (F.col('device_custom_score') <= 1.92), 'H' ))
  #                   .withColumn('device_cat', F.when( (F.col('device_custom_score') > 0.5) & (F.col('device_custom_score') < 1.2), 'I' ).otherwise('J'))                  
  #                 )

  return submission_df

def only_wday_and_hours(submission_df, spark):
  submission_df = submission_df.select('click_id', 'click_time', F.date_format('click_time', 'E').alias('click_time_wday'))  

  submission_df = submission_df.withColumn('click_time_hour', F.date_format('click_time', 'H'))     

  return submission_df.drop('click_time')

def only_n_previous_clicks(submission_df, spark):
  w1 = W.partitionBy('ip', 'device', 'os').orderBy('click_time')

  submission_df = submission_df.withColumn('times_attributed', F.sum(F.lit(1)).over(w1) )

  w2 = W.partitionBy('ip', 'device', 'os', 'times_attributed').orderBy('click_time')

  submission_df = submission_df.withColumn('n_previous_clicks', F.row_number().over(w2) - 1)

  return submission_df.select('click_id', 'n_previous_clicks')

def only_clicktimediff(submission_df, spark):
  w1 = W.partitionBy('ip', 'device', 'os').orderBy('click_time')

  submission_df = submission_df.withColumn('times_attributed', F.sum(F.lit(1)).over(w1) )

  w2 = W.partitionBy('ip', 'device', 'os', 'times_attributed').orderBy('click_time')

  submission_df = (submission_df.withColumn('group_first', F.first('click_time').over(w2))
                      .withColumn('click_time_diff', F.abs( F.col('click_time').cast('long') - F.col('group_first').cast('long') ) / 60 )
                  )

  return submission_df.select('click_id', 'click_time_diff')  