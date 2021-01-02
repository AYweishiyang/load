package fun.shiyang.forecasting;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;
import scala.collection.script.Start;
import shapeless.the;

import java.util.Arrays;

/**
 * @author ay
 * @create 2021-01-02 15:38
 */
public class TestStructureSteaming {
    public static void main(String[] args) throws StreamingQueryException {

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaStructuredNetworkWordCount")
                .config("spark.master","local[*]")
                .getOrCreate();
        // Create DataFrame representing the stream of input lines from connection to localhost:9999
        Dataset<Row> lines = spark
                .readStream()
                .format("socket")
                .option("host", "localhost")
                .option("port", 9999)
                .load();
        // Split the lines into words
        Dataset<String> words = lines
                .as(Encoders.STRING())
                .flatMap((FlatMapFunction<String, String>) x -> Arrays.asList(x.split(" ")).iterator(), Encoders.STRING());
//         Generate running word count
        Dataset<Row> wordCounts = words.groupBy("value").count();

//         Start running the query that prints the running counts to the console
//        StreamingQuery query = wordCounts.writeStream()
//                .outputMode("complete")
//                .format("console")
//                .start();

//        CrossValidatorModel crossValidatorModel = new CrossValidatorModel
        StreamingQuery query = wordCounts
                .selectExpr("CAST(value AS STRING)","CAST(count AS STRING)")
                .writeStream()
                .outputMode("complete")
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("topic", "test")
                .option("checkpointLocation","./haha")
                .start();

        query.awaitTermination();
    }

}
