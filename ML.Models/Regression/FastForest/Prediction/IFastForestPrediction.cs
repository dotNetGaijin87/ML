using Microsoft.ML;

namespace ML.Models
{
    /// <summary>
    /// Interface for <class cref="FastForestPrediction"/> 
    /// </summary>
    public interface IFastForestPrediction
    {
        FastForestPredictionOutput Run(ITransformer mlModel, FastForestPredictionInput input);
    }
}