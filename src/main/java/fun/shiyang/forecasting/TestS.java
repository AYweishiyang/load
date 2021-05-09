package fun.shiyang.forecasting;

import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder;
import org.apache.spark.sql.expressions.javalang.typed;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;
import scala.collection.Seq;

import java.util.Arrays;


/**
 * @author ay
 * @create 2021-05-09 18:13
 */


public class TestS {
    public static void main(String[] args) throws StreamingQueryException {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaStructuredNetworkWordCount")
                .config("spark.master","local[*]")
                .getOrCreate();

        Dataset<Row> lines = spark
                .readStream()
                .format("socket")
                .option("host", "localhost")
                .option("port", 9999)
                .option("includeTimestamp", true)
                .load();

        Dataset<String> stringDataset = lines
                .select("value")
                .as(Encoders.STRING())
                .flatMap((FlatMapFunction<String, String>) x -> Arrays.asList(x.split(",")).iterator(), Encoders.STRING());

        Seq<Seq<String>> rows = stringDataset.getRows(1, 0);
        
//
//        Dataset<Row> data = spark
//                .read()
//                .format("csv")
//                .option("header", "true")
//                .option("multiLine", true)
//                .option("inferSchema", true)
//                .load("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\final-data.csv");
//
//        data.createOrReplaceTempView("load");
//
//        Dataset<Row> sql = spark.sql("SELECT *,temp as value FROM load limit 1");
//
//        sql.show();
//
//        Dataset<Row> value = lines.as(Encoders.bean(DeviceData.class)).groupBy(
//                functions.window(lines.col("timestamp"),"20 seconds","10 seconds"),
//                lines.col("value")).count();
//
//        Dataset<Row> join = value.join(sql, "value");

        StreamingQuery query = stringDataset.writeStream()
        .outputMode("update")
        .format("console")
        .start();
        query.awaitTermination();

    }

}
