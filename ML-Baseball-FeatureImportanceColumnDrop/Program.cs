using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;

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
            Console.WriteLine("Starting Baseball Predictions - Training Model Job");
            Console.WriteLine("Using ML.NET - Version 1.4");
            Console.WriteLine();
            Console.ResetColor();
            Console.WriteLine("This job will build a series of models that will predict both:");
            Console.WriteLine("1) Whether a baseball batter would make it on the HOF Ballot (OnHallOfFameBallot)");
            Console.WriteLine("2) Whether a baseball batter would be inducted to the HOF (InductedToHallOfFame).");
            Console.WriteLine("Based on an MLB batter's summarized career batting statistics.\n");
            Console.WriteLine("Note: The goal is to build a 'good enough' set of models & showcase the ML.NET framework.");
            Console.WriteLine("Note: For better models advanced historical scaling and features should be performed.");
            Console.WriteLine();

            // Set the seed explicitly for reproducability (models will be built with consistent results)
            _mlContext = new MLContext(seed: _seed);

            // Read the training/validation data from a text file
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

            // Name of Algorithm
            var _algorithmName = "Gam";

            var featureSetups = new List<FeatureSetup>();
            featureSetups.Add(new FeatureSetup { Name = "Baseline", FeatureColumns = featureColumns });
            // Add Feature Columns with each column removed
            foreach (var feature in featureColumns)
            {
                var featuresEdited = featureColumns.Where(a => (a != feature)).Select(b => b).ToArray();
                featureSetups.Add(new FeatureSetup { Name = $"Removed: {feature}", FeatureColumns = featuresEdited });
            }

            // GAM Parameters
            var _numberOfIterations = new Random(_seed).Next(9500, 9500);
            var _learningRate = (double)new Random(_seed).Next(20, 20) / 10000;
            var _maximumBinCountPerFeature = new Random(_seed).Next(300, 300);

            foreach (var labelColumn in labelColumns)
            {
                foreach (var featureSetup in featureSetups)
                {
                    // Build simple data pipeline
                    var learningPipelineGeneralizedAdditiveModelsOnHallOfFameBallot =
                        Utilities.GetBaseLinePipeline(_mlContext, featureSetup.FeatureColumns).Append(
                        _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: labelColumn,
                            learningRate: _learningRate,
                            numberOfIterations: _numberOfIterations,
                            maximumBinCountPerFeature: _maximumBinCountPerFeature
                            )
                        );
                    // Fit (Build a Machine Learning Model)
                    var stopWatch = new Stopwatch();
                    stopWatch.Start();

                    var _numberOfFolds = 5;
                    var crossValidatedModels = _mlContext.BinaryClassification.CrossValidate(cachedFullData, learningPipelineGeneralizedAdditiveModelsOnHallOfFameBallot,
                        numberOfFolds: _numberOfFolds, labelColumnName: labelColumn, seed: _seed);
                    Console.WriteLine($"Finished: {featureSetup.Name}");
                    stopWatch.Stop();

                    var f1MetricsAvg = crossValidatedModels.Select(fold => fold.Metrics.F1Score).Sum() / (_numberOfFolds);
                    var aucPRMetricsAvg = crossValidatedModels.Select(fold => fold.Metrics.AreaUnderPrecisionRecallCurve).Sum() / (_numberOfFolds);
                    var positivePrecisionMetricsAvg = crossValidatedModels.Select(fold => fold.Metrics.PositivePrecision).Sum() / (_numberOfFolds);
                    var positiveRecallMetricsAvg = crossValidatedModels.Select(fold => fold.Metrics.PositiveRecall).Sum() / (_numberOfFolds);
                    var metricsRow = $@"{featureSetup.Name},{_algorithmName},{_seed},{_numberOfIterations},{_maximumBinCountPerFeature},{_learningRate},{f1MetricsAvg},{aucPRMetricsAvg},{positivePrecisionMetricsAvg},{positiveRecallMetricsAvg}";

                    _modelPerformanceMetrics.Add(
                        new ModelPerformanceMetrics
                        {
                            FeatureStepName = featureSetup.Name,
                            LabelColumn = labelColumn,
                            F1Score = f1MetricsAvg,
                            AreaUnderPrecisionRecallCurve = aucPRMetricsAvg,
                            PositivePrecision = positivePrecisionMetricsAvg,
                            PositiveRecall = positiveRecallMetricsAvg
                        }
                    );

                    Console.WriteLine("Crossvalidation Performance Metrics for " + labelColumn + " | " + featureSetup.Name);
                    Console.WriteLine("**************************");
                    Console.WriteLine("F1 Score:                 " + f1MetricsAvg);
                    Console.WriteLine("AUC - Prec/Recall Score:  " + aucPRMetricsAvg);
                    Console.WriteLine("Precision:                " + positivePrecisionMetricsAvg);
                    Console.WriteLine("Recall:                   " + positiveRecallMetricsAvg);
                    Console.WriteLine("**************************");
                    Console.WriteLine("Model Build Time: " + stopWatch.Elapsed.TotalSeconds);

                    using (System.IO.StreamWriter file = File.AppendText(_modelPerformanceMetricsFile))
                    {
                        file.WriteLine(metricsRow);
                    }
                }
            }

            Console.WriteLine("Job Finished");
        }
    }
}
