import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("MySparkApp")
      .master("local[*]")
      .getOrCreate()

    val transformer = new DataTransformer(spark)

    val df = transformer.transform("data/creditcard.csv.gz")

    spark.stop()
  }
}
