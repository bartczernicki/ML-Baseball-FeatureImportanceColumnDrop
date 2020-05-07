using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Baseball_FeatureImportanceColumnDrop
{
    public class ModelPerformanceMetrics
    {
        public string LabelColumn;
        public string FeatureStepName;
        public double MCCScore;
        public double F1Score;
        public double AreaUnderPrecisionRecallCurve;
        public double PositivePrecision;
        public double PositiveRecall;
        public double NegativePrecision;
        public double NegativeRecall;
    }
}
