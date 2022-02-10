using Microsoft.ML;
using Microsoft.ML.Data;
using ML.Helpers;
using ML.Models.Regression.FastForest;
using System.Collections.Generic;

namespace ML.Models
{
    /// <summary>
    /// Interface for <class cref="FastForestModelBuilder"/> 
    /// </summary>
    public interface IFastForestModelBuilder : IModelType
    {
        IReadOnlyList<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> Run(IDataView trainingDataView, FastForestModelBuilderSettings settings);
    }
}