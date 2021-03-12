package fun.shiyang.forecasting;

import com.microsoft.ml.spark.lightgbm.LightGBMRegressor;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Option;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 使用随机森林回归
 * @author ay
 * @create 2020-12-13 11:41
 */
public class RF {
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
//                .load("")
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
        Dataset<Row> trainingData = spark.sql("select * from all where (timestamp < '2019-01-01 00:00:00')");

        Dataset<Row> predictionFeature1 = spark.sql("select * from all where (timestamp >= '2019-05-01 00:00:00') and (timestamp < '2019-05-02 00:00:00')");
        Dataset<Row> predictionFeature2 = spark.sql("select * from all where (timestamp >= '2019-05-02 00:00:00') and (timestamp < '2019-05-03 00:00:00')");
        Dataset<Row> predictionFeature3 = spark.sql("select * from all where (timestamp >= '2019-05-03 00:00:00') and (timestamp < '2019-05-04 00:00:00')");
        Dataset<Row> predictionFeature4 = spark.sql("select * from all where (timestamp >= '2019-05-04 00:00:00') and (timestamp < '2019-05-05 00:00:00')");
        Dataset<Row> predictionFeature5 = spark.sql("select * from all where (timestamp >= '2019-05-05 00:00:00') and (timestamp < '2019-05-06 00:00:00')");
        Dataset<Row> predictionFeature6 = spark.sql("select * from all where (timestamp >= '2019-05-06 00:00:00') and (timestamp < '2019-05-07 00:00:00')");
        Dataset<Row> predictionFeature7 = spark.sql("select * from all where (timestamp >= '2019-06-07 00:00:00') and (timestamp < '2019-06-08 00:00:00')");
        Dataset<Row> predictionFeature8 = spark.sql("select * from all where (timestamp >= '2019-06-08 00:00:00') and (timestamp < '2019-06-09 00:00:00')");
        Dataset<Row> predictionFeature9 = spark.sql("select * from all where (timestamp >= '2019-06-09 00:00:00') and (timestamp < '2019-06-10 00:00:00')");

        List<Dataset<Row>> predictionFeatureList= new ArrayList<>();

        predictionFeatureList.add(predictionFeature1);
        predictionFeatureList.add(predictionFeature2);
        predictionFeatureList.add(predictionFeature3);
        predictionFeatureList.add(predictionFeature4);
        predictionFeatureList.add(predictionFeature5);
        predictionFeatureList.add(predictionFeature6);
        predictionFeatureList.add(predictionFeature7);
        predictionFeatureList.add(predictionFeature8);
        predictionFeatureList.add(predictionFeature9);

        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("load")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{rf});

        ParamMap[] paramGridBuilder = new ParamGridBuilder()
//                .addGrid(rf.numTrees(), new int[]{80,100,150})
                .addGrid(rf.maxDepth(),new int[]{3,4,5})
//                .addGrid(rf.subsamplingRate(),new double[]{0.7,0.9,1.0})
//                .addGrid(rf.maxBins(),new int[]{16,24,32})
                .build();

        RegressionEvaluator regressionEvaluator1 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        RegressionEvaluator regressionEvaluator2 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("mse");
        RegressionEvaluator regressionEvaluator3 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("r2");
        RegressionEvaluator regressionEvaluator4 = new RegressionEvaluator()
                .setLabelCol("load")
                .setPredictionCol("prediction")
                .setMetricName("mae");

        // Train model. This also runs the indexer.

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(regressionEvaluator1)
                .setEvaluator(regressionEvaluator2)
                .setEvaluator(regressionEvaluator3)
                .setEvaluator(regressionEvaluator4)
                .setEstimatorParamMaps(paramGridBuilder)
                .setNumFolds(5)
                .setParallelism(12);
        long start = System.currentTimeMillis();
        CrossValidatorModel crossValidatorModel = crossValidator.fit(trainingData);

        long end = System.currentTimeMillis();

        output.createOrReplaceTempView("all");

        List<Dataset<Row>> predictions = predictionFeatureList.stream().map(item -> {
            Dataset<Row> transform = crossValidatorModel.transform(item);
            return transform;
        }).collect(Collectors.toList());


//        crossValidatorModel.write().overwrite().save("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\loadForecasting.model");
//
//        CrossValidatorModel loadModel = CrossValidatorModel.read().load("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\loadForecasting.model");

        predictions.forEach(item -> {
            double rmse = regressionEvaluator1.evaluate(item);
            double mse = regressionEvaluator2.evaluate(item);
            double r2 = regressionEvaluator3.evaluate(item);
            double mae = regressionEvaluator4.evaluate(item);
            System.out.println("(rmse) on test data = " + rmse);
//            System.out.println("(mse) on test data = " + mse);
//            System.out.println("(r2) on test data = " + r2);
//            System.out.println("(mae) on test data = " + mae);
//            System.out.println("----------------------------------------");
            Dataset<Row> prediction = item.select("prediction");


        });

//        Dataset<Row> loadPrediction = loadModel.transform(predictionFeature);

        PipelineModel bestModel = (PipelineModel)crossValidatorModel.bestModel();

        System.out.println("bestModel.stages().length=" + bestModel.stages()[0].extractParamMap().toString());


        System.out.println("train-time= " + (end-start)/1000.0 + " s");
        spark.stop();
    }
}
