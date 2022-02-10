using Microsoft.ML.Data;

namespace ML.Models
{
    /// <summary>
    /// Contains output prediction of the SrCnnEntireAnomalyDetector for each time input data point
    /// </summary>
    public class SrCnnTrainerOutput
    {
        [VectorType]
        public double[] Prediction { get; set; }
    }
}
