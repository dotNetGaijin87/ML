using Microsoft.ML.Data;
using ML.Models;
using ML.Models.Regression.FastForest;
using System.Collections.Generic;

namespace ML.Services
{
    /// <summary>
    /// Interface for <class cref="FastForestService"/> 
    /// </summary>
    public interface IFastForestService
    {
        List<RegressionMetrics> CreateModelAndValidate(FastForestModelBuilderSettings settings);
        bool Delete(string modelName);
        FastForestPredictionOutput Run(string modelName, FastForestPredictionInput input);
    }
}