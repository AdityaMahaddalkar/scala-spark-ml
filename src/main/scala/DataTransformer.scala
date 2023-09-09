import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}

class DataTransformer(spark: SparkSession) {

  private final val TARGET_COL: String = "label"

  private def readFile(filePath: String): DataFrame = {
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(filePath)
  }

  def transform(filePath: String): DataFrame = {

    var df = readFile(filePath)

    df = df.withColumn("Class", col("Class").cast(DoubleType))
      .withColumnRenamed("Class", TARGET_COL)

    val featureCols = df.columns.filter(col => col != TARGET_COL)

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(assembler))

    pipeline.fit(df).transform(df)

  }
}
