package fun.shiyang.forecasting;

import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;

/**
 * @author ay
 * @create 2021-01-02 20:20
 */
public class TestKafka {
    public static void main(String[] args) throws StreamingQueryException {
        CrossValidatorModel loadModel = CrossValidatorModel.read().load("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\loadForecasting.model");

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaStructuredNetworkWordCount")
                .config("spark.master","local[*]")
                .getOrCreate();
        // Subscribe to 1 topic
        Dataset<Row> dataset = spark
                .readStream()
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "test")
                .load();

        Dataset<Row> prediction = loadModel.transform(dataset);
        
//        df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)");
        StreamingQuery query = dataset
                .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)","topic","partition","offset","timestamp","timestampType")
                .writeStream()
                .outputMode("append")
                .format("console")
                .start();

        query.awaitTermination();
    }
}
