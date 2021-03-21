package fun.shiyang.forecasting;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * 使用随机森林回归
 * @author ay
 * @create 2020-12-13 11:41
 */
public class RFNoCvHour {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession
                .builder()
                .appName("RF")
                .config("spark.master","local[*]")
//                .config("spark.eventLog.enabled", "false")
                .getOrCreate();
        Dataset<Row> data = spark
                .read()
                .format("csv")
                .option("header", "true")
                .option("multiLine", true)
                .option("inferSchema", true)
                .load("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\final-data.csv");
//                .load("/final-data.csv");
//        data.createOrReplaceTempView("load");
//        data.orderBy("load").show();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "condition1",
                        "wind1",
                        "windspeed1",
                        "precip1",
                        "temp1",
                        "dew1",
                        "humi1",
                        //              "windspeed",
                        //              "precip",
                        "hour",
                        "dayofweek",
                        //              "quarter",
                        //               "month",
                        //              "year",
                        "dayofyear",
                        "dayofmonth",
                        "weekofyear",
                        "t_m24",
                        "t_m48" ,
                        "MA_14",
                        "MA_28",
                        "MA_60",
                        "MA_90",
                        "MMAX_14",
                        "MMIN_14",
                        "MSTD_14",
                        "MSLOPE_14",
                        "MACC_14",
                        "MMAX_28",
                        "MMIN_28",
                        "MSTD_28",
                        "MSLOPE_28",
                        "MACC_28",
                        "MMAX_60",
                        "MMIN_60",
                        "MSTD_60",
                        "MSLOPE_60",
                        "MACC_60",
                        "MMAX_90",
                        "MMIN_90",
                        "MSTD_90",
                        "MSLOPE_90",
                        "MACC_90",
                        "lag_1",
                        "lag_7",
                        "lag_14",
                        "lag_30"
                        //#           'tdif',
                })
                .setOutputCol("features");

        Dataset<Row> output = vectorAssembler.transform(data);
        output.show(10);

        output.createOrReplaceTempView("all");
        Dataset<Row> trainingData = spark.sql("select * from all where (timestamp < '2019-05-01 00:00:00')");


        Dataset<Row> predictionFeature1 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:00:00') and (timestamp < '2019-05-08 01:00:00')");
        Dataset<Row> predictionFeature2 = spark.sql("select * from all where (timestamp >= '2019-05-08 01:00:00') and (timestamp < '2019-05-08 02:00:00')");
        Dataset<Row> predictionFeature3 = spark.sql("select * from all where (timestamp >= '2019-05-08 02:00:00') and (timestamp < '2019-05-08 03:00:00')");
        Dataset<Row> predictionFeature4 = spark.sql("select * from all where (timestamp >= '2019-05-08 03:00:00') and (timestamp < '2019-05-08 04:00:00')");
        Dataset<Row> predictionFeature5 = spark.sql("select * from all where (timestamp >= '2019-05-08 04:00:00') and (timestamp < '2019-05-08 05:00:00')");
        Dataset<Row> predictionFeature6 = spark.sql("select * from all where (timestamp >= '2019-05-08 05:00:00') and (timestamp < '2019-05-08 06:00:00')");
        Dataset<Row> predictionFeature7 = spark.sql("select * from all where (timestamp >= '2019-05-08 06:00:00') and (timestamp < '2019-05-08 07:00:00')");
        Dataset<Row> predictionFeature8 = spark.sql("select * from all where (timestamp >= '2019-05-08 07:00:00') and (timestamp < '2019-05-08 08:00:00')");
        Dataset<Row> predictionFeature9 = spark.sql("select * from all where (timestamp >= '2019-05-08 08:00:00') and (timestamp < '2019-05-08 09:00:00')");
        Dataset<Row> predictionFeature10 = spark.sql("select * from all where (timestamp >= '2019-05-08 09:00:00') and (timestamp < '2019-05-08 10:00:00')");

        //one hour
        Dataset<Row> predictionFeature11 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:10:00') and (timestamp < '2019-05-08 11:00:00')");
        Dataset<Row> predictionFeature12 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:11:00') and (timestamp < '2019-05-08 12:00:00')");
        Dataset<Row> predictionFeature13 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:12:00') and (timestamp < '2019-05-08 13:00:00')");
        Dataset<Row> predictionFeature14 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:13:00') and (timestamp < '2019-05-08 14:00:00')");
        Dataset<Row> predictionFeature15 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:14:00') and (timestamp < '2019-05-08 15:00:00')");
        Dataset<Row> predictionFeature16 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:15:00') and (timestamp < '2019-05-08 16:00:00')");
        Dataset<Row> predictionFeature17 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:16:00') and (timestamp < '2019-05-08 17:00:00')");
        Dataset<Row> predictionFeature18 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:17:00') and (timestamp < '2019-05-08 18:00:00')");
        Dataset<Row> predictionFeature19 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:18:00') and (timestamp < '2019-05-08 19:00:00')");
        Dataset<Row> predictionFeature20 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:19:00') and (timestamp < '2019-05-08 20:00:00')");
        Dataset<Row> predictionFeature21 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:20:00') and (timestamp < '2019-05-08 21:00:00')");
        Dataset<Row> predictionFeature22 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:21:00') and (timestamp < '2019-05-08 22:00:00')");
        Dataset<Row> predictionFeature23 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:22:00') and (timestamp < '2019-05-08 23:00:00')");
        Dataset<Row> predictionFeature24 = spark.sql("select * from all where (timestamp >= '2019-05-08 00:23:00') and (timestamp < '2019-05-09 00:00:00')");


        List<Dataset<Row>> predictionFeatureList= new ArrayList<>();
//
        predictionFeatureList.add(predictionFeature1);
        predictionFeatureList.add(predictionFeature2);
        predictionFeatureList.add(predictionFeature3);
        predictionFeatureList.add(predictionFeature4);
        predictionFeatureList.add(predictionFeature5);
        predictionFeatureList.add(predictionFeature6);
        predictionFeatureList.add(predictionFeature7);
        predictionFeatureList.add(predictionFeature8);
        predictionFeatureList.add(predictionFeature9);
        predictionFeatureList.add(predictionFeature10);
        predictionFeatureList.add(predictionFeature11);
        predictionFeatureList.add(predictionFeature12);
        predictionFeatureList.add(predictionFeature13);
        predictionFeatureList.add(predictionFeature14);
        predictionFeatureList.add(predictionFeature15);
        predictionFeatureList.add(predictionFeature16);
        predictionFeatureList.add(predictionFeature17);
        predictionFeatureList.add(predictionFeature18);
        predictionFeatureList.add(predictionFeature19);
        predictionFeatureList.add(predictionFeature20);
        predictionFeatureList.add(predictionFeature21);
        predictionFeatureList.add(predictionFeature22);
        predictionFeatureList.add(predictionFeature23);
        predictionFeatureList.add(predictionFeature24);


        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("load")
                .setFeaturesCol("features");

        rf.setMaxDepth(7)
                .setSubsamplingRate(0.7)
                .setNumTrees(100);


        long start = System.currentTimeMillis();
        RandomForestRegressionModel randomForestRegressionModel = rf.fit(trainingData);
        long end = System.currentTimeMillis();
        System.out.println("train-time= " + (end-start)/1000.0 + " s");

        Vector vector = randomForestRegressionModel.featureImportances();

        double[] doubles = vector.toArray();

        System.out.println(vector.toString());

        RegressionEvaluator regressionEvaluator1 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        output.createOrReplaceTempView("all");

        AtomicInteger index = new AtomicInteger(1);
        //预测结果
        List<Dataset<Row>> predictions = predictionFeatureList.stream().map(item -> {
            Dataset<Row> transform = randomForestRegressionModel.transform(item);

            List<Row> predic = transform.select("prediction").collectAsList();
            List<Row> load = transform.select("load").collectAsList();
            List<Row> timestamp = transform.select("timestamp").collectAsList();

            List<String> timeList = timestamp.stream().map(row -> {
                return row.getTimestamp(0).toString();
            }).collect(Collectors.toList());

            List<Double> predicList = predic.stream().map(row -> {
                return row.getDouble(0);
            }).collect(Collectors.toList());

            List<Double> loadList = load.stream().map(row -> {
                return row.getDouble(0);
            }).collect(Collectors.toList());
            System.out.println("#time" + index + " = " + timeList);
            System.out.println("rf_predict" + index + " = "+ predicList);
            System.out.println("load"  + index + " = " + loadList);

            double sum = 0D;
            for (int i = 0; i < loadList.size(); i++) {
                double temp = Math.abs((loadList.get(i) - predicList.get(i)) / loadList.get(i));
                sum += temp;
            }
            double mape = sum / loadList.size()*100;
            System.out.println("print(" + mape + ")");
            System.out.println("######################");
            index.getAndIncrement();
            return transform;
        }).collect(Collectors.toList());

        predictions.forEach(item -> {
            double rmse = regressionEvaluator1.evaluate(item);
            System.out.println("(rmse) on test data = " + rmse);
            Dataset<Row> prediction = item.select("prediction");

        });
        spark.stop();
    }
}

