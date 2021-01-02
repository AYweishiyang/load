package fun.shiyang.forecasting;

import com.microsoft.ml.spark.lightgbm.LightGBMRegressor;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

/**
 * @author ay
 * @create 2020-12-13 11:41
 */
public class LoadModel {
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
                //       "condition1",
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
        Dataset<Row> predictionFeature = spark.sql("select * from all where (timestamp >= '2019-04-01 00:00:00') and (timestamp < '2019-05-01 00:00:00')");

//        Dataset<Row> testData = spark.sql("select * from all where (timestamp >= '2019-06-01 00:00:00') and (timestamp < '2019-07-01 00:00:00')");

//        Dataset<Row>[] splits = selectData.randomSplit(new double[] {0.7, 0.3});
//        Dataset<Row> trainingData = splits[0];
//        Dataset<Row> testData = splits[1];

        LightGBMRegressor lightGBMRegressor = new LightGBMRegressor()
                .setLabelCol("load")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{lightGBMRegressor});

        ParamMap[] paramGridBuilder = new ParamGridBuilder()
                .addGrid(lightGBMRegressor.maxDepth(), new int[]{3,4})
                .addGrid(lightGBMRegressor.numLeaves(), new int[]{15,31})
                .addGrid(lightGBMRegressor.learningRate(),new double[]{0.1})
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
                .setNumFolds(3)
                .setParallelism(10);
        long start = System.currentTimeMillis();
        CrossValidatorModel crossValidatorModel = crossValidator.fit(trainingData);

        long end = System.currentTimeMillis();

        output.createOrReplaceTempView("all");

        Dataset<Row> prediction = crossValidatorModel.transform(predictionFeature);

        crossValidatorModel.write().overwrite().save("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\loadForecasting.model");

        CrossValidatorModel loadModel = CrossValidatorModel.read().load("file:///E:\\mi\\jupyter\\energy_forecasting_notebooks\\loadForecasting.model");

        prediction.select("prediction", "load", "features").show(5);

        double rmse = regressionEvaluator1.evaluate(prediction);
        double mse = regressionEvaluator2.evaluate(prediction);
        double r2 = regressionEvaluator3.evaluate(prediction);
        double mae = regressionEvaluator4.evaluate(prediction);
        System.out.println("(rmse) on test data = " + rmse);
        System.out.println("(mse) on test data = " + mse);
        System.out.println("(r2) on test data = " + r2);
        System.out.println("(mae) on test data = " + mae);
        System.out.println("----------------------------------------");
        Dataset<Row> loadPrediction = loadModel.transform(predictionFeature);

        double rmse1 = regressionEvaluator1.evaluate(loadPrediction);
        double mse1 = regressionEvaluator2.evaluate(loadPrediction);
        double r21 = regressionEvaluator3.evaluate(loadPrediction);
        double mae1 = regressionEvaluator4.evaluate(loadPrediction);
        System.out.println("(rmse1) on test data = " + rmse1);
        System.out.println("(mse1) on test data = " + mse1);
        System.out.println("(r21) on test data = " + r21);
        System.out.println("(mae1) on test data = " + mae1);

        PipelineModel bestModel = (PipelineModel)crossValidatorModel.bestModel();
        Transformer[] stages = ((PipelineModel) crossValidatorModel.bestModel()).stages();

        System.out.println("bestModel.stages().length=" + bestModel.stages()[0].extractParamMap().toString());

        System.out.println("train-time= " + (end-start)/1000.0 + " s");
        spark.stop();
    }
}
