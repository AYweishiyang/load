package fun.shiyang.forecasting;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 使用随机森林回归
 * @author ay
 * @create 2020-12-13 11:41
 */
public class RFNoCv {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession
                .builder()
                .appName("RF")
                .config("spark.master","local[*]")
                .config("spark.eventLog.enabled", "false")
                .getOrCreate();
        Dataset<Row> data = spark
                .read()
                .format("csv")
                .option("header", "true")
                .option("multiLine", true)
                .option("inferSchema", true)
                .load("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\final-data.csv");
//        data.createOrReplaceTempView("load");
//        data.orderBy("load").show();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{
//                        "temp",
                        "temp1",
                        "dew1",
//                       "condition1",
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
        Dataset<Row> trainingData = spark.sql("select * from all where (timestamp < '2019-03-01 00:00:00')");

        Dataset<Row> predictionFeature1 = spark.sql("select * from all where (timestamp >= '2019-05-01 00:00:00') and (timestamp < '2019-05-02 00:00:00')");
        Dataset<Row> predictionFeature2 = spark.sql("select * from all where (timestamp >= '2019-05-02 00:00:00') and (timestamp < '2019-05-03 00:00:00')");
        Dataset<Row> predictionFeature3 = spark.sql("select * from all where (timestamp >= '2019-05-03 00:00:00') and (timestamp < '2019-05-04 00:00:00')");
        Dataset<Row> predictionFeature4 = spark.sql("select * from all where (timestamp >= '2019-05-04 00:00:00') and (timestamp < '2019-05-05 00:00:00')");
        Dataset<Row> predictionFeature5 = spark.sql("select * from all where (timestamp >= '2019-05-05 00:00:00') and (timestamp < '2019-05-06 00:00:00')");

        List<Dataset<Row>> predictionFeatureList= new ArrayList<>();

        predictionFeatureList.add(predictionFeature1);
        predictionFeatureList.add(predictionFeature2);
        predictionFeatureList.add(predictionFeature3);
        predictionFeatureList.add(predictionFeature4);
        predictionFeatureList.add(predictionFeature5);

        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("load")
                .setFeaturesCol("features");

        RandomForestRegressionModel randomForestRegressionModel = rf.fit(trainingData);

        Vector vector = randomForestRegressionModel.featureImportances();

        double[] doubles = vector.toArray();

//        new ArrayList<>()
//
//        for (double d:doubles) {
//
//        }
//
//        Collections.sort(list);


        System.out.println(vector.toString());

        spark.stop();
    }
}
