using Microsoft.ML.TimeSeries;
using ML.Models;
using System.Collections.Generic;
using System.Linq;

namespace ML.Services
{
    /// <summary>
    /// Class that manages anomaly detection based on 
    /// Super-resolution using convolutional neural network (SRCNN) algorithm
    /// </summary>
    public class SrCnnService : ISrCnnService
    {
        private readonly ISrCnnTrainer _srCnntrainer;

        public SrCnnService(ISrCnnTrainer srCnntrainer)
        {
            _srCnntrainer = srCnntrainer;
        }

        /// <summary>
        /// Creates SrCnnTrainer and runs prediction
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public IEnumerable<SrCnnOutput> Predict(SrCnnInput input)
        {
            var inputModel = new SrCnnTrainerInputCollection()
            {
                Values = input.TrainingData.Select(x => new SrCnnTrainerInput { Value = x }).ToList()
            };

            var options = new SrCnnEntireAnomalyDetectorOptions
            {
                Threshold = input.Threshold,
                BatchSize = input.BatchSize,
                Sensitivity = input.Sensitivity,
                Period = input.Period,
            };

            IEnumerable<SrCnnOutput> predictions = _srCnntrainer
                                                        .Run(inputModel, options)
                                                        .Select(x => new SrCnnOutput
                                                        {
                                                            IsAnomaly = x.Prediction[0],
                                                            RawScore = x.Prediction[1],
                                                            Mag = x.Prediction[2],
                                                        })
                                                        .ToList();
            return predictions;
        }
    }
}
