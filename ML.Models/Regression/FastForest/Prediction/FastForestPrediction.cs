using Microsoft.ML;


namespace ML.Models
{
    /// <summary>
    /// Class used for calculating prediction of unknown data based known input data, and trained model
    /// Class uses FastForestRegressionTrainer
    /// </summary>
    public class FastForestPrediction : IFastForestPrediction
    {
        private MLContext _mlContext;

        public FastForestPrediction(MLContext mlContext)
        {
             _mlContext = mlContext;
        }

        /// <summary>
        /// Creates prediction engine from provided model, 
        /// and then runs it with input data to obtain score for the searched parameter
        /// </summary>
        /// <param name="mlModel">Model used for creating a prediction engine</param>
        /// <param name="input">Input data that the prediction is based o</param>
        /// <returns></returns>
        public FastForestPredictionOutput Run(ITransformer mlModel, FastForestPredictionInput input)
        {
            var predEngine = _mlContext.Model.CreatePredictionEngine<FastForestPredictionInput, FastForestPredictionOutput>(mlModel);

            FastForestPredictionOutput result = predEngine.Predict(input);

            return result;
        }

    }
}
