using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using System;
using System.Collections.Generic;

namespace ML.Models
{
    /// <summary>
    /// Class used for anomaly detection based on ML.NET SrCnnEntireAnomalyDetector
    /// </summary>
    public class SrCnnTrainer : ISrCnnTrainer
    {
        private MLContext _mlContext;
        private string _outputColumnName = nameof(SrCnnTrainerOutput.Prediction);
        private string _inputColumnName = nameof(SrCnnTrainerInput.Value);


        public SrCnnTrainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// <summary>
        /// Loads input data and runs with it SrCnnEntireAnomalyDetector
        /// </summary>
        /// <param name="input">Time series data</param>
        /// <param name="options">Options to configure the detector</param>
        /// <returns></returns>
        public virtual IEnumerable<SrCnnTrainerOutput> Run(SrCnnTrainerInputCollection input, SrCnnEntireAnomalyDetectorOptions options)
        {
            CheckPreCondition(input, options);

            var dataView = _mlContext.Data.LoadFromEnumerable(input.Values);
            var outputDataView = _mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, _outputColumnName, _inputColumnName, options);

            return _mlContext.Data.CreateEnumerable<SrCnnTrainerOutput>(outputDataView, reuseRowObject: false);
        }


        /// <summary>
        /// Checks preconditions for running SrCnnEntireAnomalyDetector, 
        /// any violation will result in SrCnnEntireAnomalyDetector raising an exception
        /// </summary>
        /// <param name="input"></param>
        /// <param name="options"></param>
        private void CheckPreCondition(SrCnnTrainerInputCollection input, SrCnnEntireAnomalyDetectorOptions options)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            if (options == null)
                throw new ArgumentNullException(nameof(options));

            if (options.Threshold < 0 ||
                options.Threshold > 1)
                throw new ArgumentException(nameof(options.Threshold));

            if (options.Sensitivity < 0 || 
                options.Sensitivity > 100)
                throw new ArgumentException(nameof(options.Sensitivity));

        }
    }
}
