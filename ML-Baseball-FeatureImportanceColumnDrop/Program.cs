using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace ML_Baseball_FeatureImportanceColumnDrop
{
    class Program
    {
        // Set up path locations
        private static string appFolder = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
        private static string _fullDataPath => Path.Combine(appFolder, "Data", "BaseballHOFFull.csv");
        private static string _performanceMetricsTrainTestModels => Path.Combine(appFolder, @"ModelPerformanceMetrics", "PerformanceMetricsTrainTestModels.csv");
        private static string _modelPerformanceMetricsFile => Path.Combine(appFolder, "Metrics", "ModelPerformanceMetrics.csv");

        // Thread-safe ML Context
        private static MLContext _mlContext;
        // Set seed to static value for re-producable model results (or DateTime for pseudo-random)
        private static int _seed = 100;

        private static int numberOfModelsToBuildForEeachIteration = 50;

        // CONFIGURATION ARRAYS

        // List of feature columns used for training
        // Useage: Comment out (or uncomment) feature names in order to explicitly select features for model training
        private static string[] featureColumns = new string[] {
            "YearsPlayed", "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "BattingAverage", "SluggingPct", "AllStarAppearances", "MVPs", "TripleCrowns", "GoldGloves",
            "MajorLeaguePlayerOfTheYearAwards", "TB", "TotalPlayerAwards" };

        // List of supervised learning labels
        // Useage: At least one must be left
        private static string[] labelColumns = new string[] { "OnHallOfFameBallot", "InductedToHallOfFame" };
        private static List<ModelPerformanceMetrics> _modelPerformanceMetrics = new List<ModelPerformanceMetrics>();

        static void Main(string[] args)
        {
            Console.Title = "Baseball Feature Importance with Column Dropout - Training Model Job";
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("Starting Baseball Predictions - Training Model Job w/ Feature Importance Column Dropout");
            Console.WriteLine("Using ML.NET - Version 1.4");
            Console.WriteLine();
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("This job will build a series of models that will predict both:");
            Console.WriteLine("1) Whether a baseball batter would make it on the HOF Ballot (OnHallOfFameBallot)");
            Console.WriteLine("2) Whether a baseball batter would be inducted to the HOF (InductedToHallOfFame).");
            Console.WriteLine("Based on an MLB batter's summarized career batting statistics.\n");
            Console.WriteLine("Note: The goal is to build a 'good enough' set of models & showcase the feature column drop.");
            Console.WriteLine("Note: For better models advanced historical scaling and features should be performed.");
            Console.ResetColor();
            Console.WriteLine();

            // Set the seed explicitly for reproducability (models will be built with consistent results)
            _mlContext = new MLContext(seed: _seed);

            // Read the Training data from a text file
            var dataFull = _mlContext.Data.LoadFromTextFile<MLBBaseballBatter>(path: _fullDataPath,
                hasHeader: true, separatorChar: ',', allowQuoting: false);

            // Retrieve Data Schema
            var dataSchema = dataFull.Schema;

            #if DEBUG
            // Debug Only: Preview the training/validation data
            var dataTrainPreview = dataFull.Preview();
            #endif

            // Cache the loaded data
            var cachedFullData = _mlContext.Data.Cache(dataFull);
            // Name of Algorithm used in training
            // Using Gam as it gives very good results and trains fast
            var _algorithmName = "Gam";
            var _jobRunId = Guid.NewGuid();

            var featureSetups = new List<FeatureSetup>();
            var dateTime = DateTime.Now.ToUniversalTime();

            for (int i = 0; i != numberOfModelsToBuildForEeachIteration; i++)
            {
                var gamAlgorithmParameters = new GamAlgorithmParameters
                {
                    // GAM Parameters
                    NumberOfIterations = new Random(_seed + i).Next(5000, 40000),
                    LearningRate = (double)new Random(_seed + i).Next(5, 30) / 10000, // 0.0005 - 0.0030
                    MaximumBinCountPerFeature = new Random(_seed + i).Next(150, 1000)
                };

                // Add a Baseline training run
                //featureSetups.Add(new FeatureSetup
                //{
                //    Name = "Baseline",
                //    FeatureColumns = featureColumns,
                //    ColumnNameRemoved = "Baseline",
                //    GamAlgorithmParameters = gamAlgorithmParameters
                //});

                // Add Feature Columns with each column removed
                //foreach (var feature in featureColumns)
                //{
                //    var featuresEdited = featureColumns.Where(a => (a != feature)).Select(b => b).ToArray();
                //    featureSetups.Add(new FeatureSetup
                //    {
                //        Name = $"Removed: {feature}",
                //        ColumnNameRemoved = feature,
                //        FeatureColumns = featuresEdited,
                //        GamAlgorithmParameters = gamAlgorithmParameters
                //    });
                //}

                var featuresEditedV1 = featureColumns.Where(a => (a != "Doubles" && a != "RBI")).Select(b => b).ToArray();
                featureSetups.Add(new FeatureSetup
                {
                    Name = $"Removed: Doubles, RBI",
                    ColumnNameRemoved = "Doubles, RBI",
                    FeatureColumns = featuresEditedV1,
                    GamAlgorithmParameters = gamAlgorithmParameters
                });

                var featuresEditedV2 = featureColumns.Where(a => (a != "Doubles" && a != "BattingAverage")).Select(b => b).ToArray();
                featureSetups.Add(new FeatureSetup
                {
                    Name = $"Removed: Doubles, BattingAverage",
                    ColumnNameRemoved = "Doubles, BattingAverage",
                    FeatureColumns = featuresEditedV2,
                    GamAlgorithmParameters = gamAlgorithmParameters
                });
            }

            // Read the file and check if the header row exists
            if (!File.Exists(_modelPerformanceMetricsFile))
            {
                File.WriteAllText(_modelPerformanceMetricsFile, string.Empty, Encoding.UTF8);
            }

            var fileText = File.ReadAllText(_modelPerformanceMetricsFile);
            if (string.IsNullOrWhiteSpace(fileText))
            {
                var metricsHeaderRow = 
                    $@"JobID,DateTime,LabelColumn,ModelTrainingTime,Description,ColumnRemoved,Algorithm,Seed," +
                    $@"Iterations,MaxBinNumber,LearningRate,GeometricMean,MCCMetric,F1ScoreMetric,AucPrecisionRecallMetric," +
                    $@"PositivePrecisionMetric,PositiveRecallMetric,NegativePrecisionMetric,NegativeRecallMetric";

                using (StreamWriter file = File.AppendText(_modelPerformanceMetricsFile))
                {
                    file.WriteLine(metricsHeaderRow);
                }
            }


            var stepNumber = 1;
            foreach (var labelColumn in labelColumns)
            {
                foreach (var featureSetup in featureSetups)
                {
                    // Build simple data pipeline
                    var learningPipelineGeneralizedAdditiveModelsOnHallOfFameBallot =
                        Utilities.GetBaseLinePipeline(_mlContext, featureSetup.FeatureColumns).Append(
                        _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: labelColumn,
                            learningRate: featureSetup.GamAlgorithmParameters.LearningRate,
                            numberOfIterations: featureSetup.GamAlgorithmParameters.NumberOfIterations,
                            maximumBinCountPerFeature: featureSetup.GamAlgorithmParameters.MaximumBinCountPerFeature
                            )
                        );
                    // Fit (Build a Machine Learning Model)
                    var stopWatch = new Stopwatch();
                    stopWatch.Start();

                    var _numberOfFolds = 5;
                    var crossValidatedModels = _mlContext.BinaryClassification.CrossValidate(cachedFullData, learningPipelineGeneralizedAdditiveModelsOnHallOfFameBallot,
                        numberOfFolds: _numberOfFolds, labelColumnName: labelColumn, seed: _seed);
                    Console.WriteLine($"Finished #{stepNumber++}: {featureSetup.Name}");
                    stopWatch.Stop();
                    //stepNumber++;
                    var secondsElapsed = Math.Round(stopWatch.Elapsed.TotalSeconds, 2);

                    var mccMetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.ConfusionMatrix.MatthewsCorrelationCoefficient()).Sum() / (_numberOfFolds), 4);
                    var geometricMeanMetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.ConfusionMatrix.GeometricMean()).Sum() / (_numberOfFolds), 4);
                    var f1MetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.F1Score).Sum() / (_numberOfFolds), 4);
                    var aucPRMetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.AreaUnderPrecisionRecallCurve).Sum() / (_numberOfFolds), 4);
                    var positivePrecisionMetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.PositivePrecision).Sum() / (_numberOfFolds), 4);
                    var positiveRecallMetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.PositiveRecall).Sum() / (_numberOfFolds), 4);
                    var negativePrecisionMetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.NegativePrecision).Sum() / (_numberOfFolds), 4);
                    var negativeRecallMetricsAvg = Math.Round(
                        crossValidatedModels.Select(fold => fold.Metrics.NegativeRecall).Sum() / (_numberOfFolds), 4);
                    var metricsRow = $@"{_jobRunId},{dateTime},{labelColumn},{secondsElapsed},{featureSetup.Name},{featureSetup.ColumnNameRemoved},"+
                        $@"{_algorithmName},{_seed},{featureSetup.GamAlgorithmParameters.NumberOfIterations},{featureSetup.GamAlgorithmParameters.MaximumBinCountPerFeature},"+
                        $@"{featureSetup.GamAlgorithmParameters.LearningRate},{mccMetricsAvg},{geometricMeanMetricsAvg},{f1MetricsAvg},{aucPRMetricsAvg},{positivePrecisionMetricsAvg},"+
                        $@"{positiveRecallMetricsAvg},{negativePrecisionMetricsAvg},{negativeRecallMetricsAvg}";

                    _modelPerformanceMetrics.Add(
                        new ModelPerformanceMetrics
                        {
                            FeatureStepName = featureSetup.Name,
                            LabelColumn = labelColumn,
                            MCCScore = mccMetricsAvg,
                            GeometricMean = geometricMeanMetricsAvg,
                            F1Score = f1MetricsAvg,
                            AreaUnderPrecisionRecallCurve = aucPRMetricsAvg,
                            PositivePrecision = positivePrecisionMetricsAvg,
                            PositiveRecall = positiveRecallMetricsAvg,
                            NegativePrecision = negativePrecisionMetricsAvg,
                            NegativeRecall = negativeRecallMetricsAvg
                        }
                    );

                    Console.WriteLine("Average Fold Crossvalidation Performance Metrics: " + labelColumn + " | " + featureSetup.Name);
                    Console.WriteLine("********************************");
                    Console.WriteLine("MCC Score:                " + mccMetricsAvg);
                    Console.WriteLine("Geometric Mean:           " + geometricMeanMetricsAvg);
                    Console.WriteLine("F1 Score:                 " + f1MetricsAvg);
                    Console.WriteLine("AUC - Prec/Recall Score:  " + aucPRMetricsAvg);
                    Console.WriteLine("Precision:                " + positivePrecisionMetricsAvg);
                    Console.WriteLine("Recall:                   " + positiveRecallMetricsAvg);
                    Console.WriteLine("Negative Precision:       " + negativePrecisionMetricsAvg);
                    Console.WriteLine("Negative Recall:          " + negativeRecallMetricsAvg);
                    Console.WriteLine("********************************");
                    Console.WriteLine("Model Build Time: " + Math.Round(stopWatch.Elapsed.TotalSeconds, 2) + "sec");
                    Console.WriteLine();

                    using (StreamWriter file = File.AppendText(_modelPerformanceMetricsFile))
                    {
                        file.WriteLine(metricsRow);
                    }
                }
            }

            Console.WriteLine("Job Finished");
        }
    }
}
