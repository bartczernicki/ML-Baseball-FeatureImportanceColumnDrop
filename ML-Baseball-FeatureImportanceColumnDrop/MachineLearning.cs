using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using System.Transactions;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML_Baseball_FeatureImportanceColumnDrop
{
    public static class MachineLearningExtensions
    {
        public static double MatthewsCorrelationCoefficient(this ConfusionMatrix confusionMatrix)
        {
            var truePositives = confusionMatrix.GetCountForClassPair(0, 0);
            var trueNegatives = confusionMatrix.GetCountForClassPair(1, 1);
            var falseNegatives = confusionMatrix.GetCountForClassPair(1, 0);
            var falsePositives = confusionMatrix.GetCountForClassPair(0, 1);

            var mccNumerator = truePositives * trueNegatives - falsePositives * falseNegatives;
            var mccDenominator = Math.Sqrt(
                1.0 * (truePositives + falsePositives) * (truePositives + falseNegatives) * (trueNegatives + falsePositives) * (trueNegatives + falseNegatives)
                                 );
            var mcc = mccNumerator / mccDenominator;

            return mcc;
        }
    }
}
