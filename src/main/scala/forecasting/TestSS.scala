package forecasting

import org.apache.spark.sql.SparkSession
import java.sql.Timestamp

/**
 * @author ay
 * @create 2021-05-09 21:25
 */
object TestSS {
  def main(args: Array[String]): Unit = {
      val spark = SparkSession
        .builder()
        .master("local[*]")
        .appName("BasicOperation")
        .getOrCreate()
      import spark.implicits._

      import org.apache.spark.sql.functions._
      val line = spark.readStream
        .format("socket")
        .option("host", "localhost")
        .option("port", 9999)
        .option("includeTimestamp", value = true)
        .load()  //给产生的数据自动添加时间戳
        .as[(String,Timestamp)]
        .flatMap{
          case (words,ts)=>words.split(" ").map((_,ts))
        }
        .toDF("word","ts")
        .groupBy(
          window($"ts","20 seconds","10 seconds"),
          $"word").count()

     line.writeStream
        .format("console")
        .outputMode("complete")
        .option("truncate",value = false)
        .start()
        .awaitTermination()

    }
}
