using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using ML.Helpers;
using ML.Models.Regression.FastForest;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.TrainCatalogBase;

namespace ML.Models
{
    /// <summary>
    /// Class used for building models based on <class cref="FastForestRegressionTrainer"/> 
    /// </summary>
    public class FastForestModelBuilder : IFastForestModelBuilder
    {
        private readonly MLContext _mlContext;
        private readonly IMLDataLoader _mLDataLoader;
        private readonly string FEATURES = "Features";
        public string ModelTypeName { get; } = "FastForest";


        public FastForestModelBuilder(MLContext mlContext, IMLDataLoader mLDataLoader)
        {
            _mlContext = mlContext;
            _mLDataLoader = mLDataLoader;
        }

        /// <summary>
        /// Builds a model based on FastForestRegressionTrainer, 
        /// saves it and runs validation to obtain model's performance metrics.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="settings"></param>
        /// <returns></returns>
        public IReadOnlyList<CrossValidationResult<RegressionMetrics>> Run(IDataView data, FastForestModelBuilderSettings settings)
        {
            var transformPipeline = _mlContext.Transforms.Concatenate(FEATURES, settings.FeatureColumnNames);

            var trainer = _mlContext.Regression.Trainers
                                        .FastForest(
                                            numberOfLeaves: settings.LeavesCount,
                                            minimumExampleCountPerLeaf: settings.MinimumExampleCountPerLeaf,
                                            numberOfTrees: settings.TreesCount,
                                            labelColumnName: settings.LabelColumnName,
                                            featureColumnName: FEATURES);


            var trainingPipeline = _mlContext.Transforms
                    .Concatenate(FEATURES, settings.FeatureColumnNames)
                    .Append(trainer);


            ITransformer mlModel;
            if (settings.HasFeatureContributionMetrics)
            {
                mlModel = CreateModelWithFeatureContributionMetrics(data, transformPipeline, trainer);
            }
            else
            {
                mlModel = trainingPipeline.Fit(data);
            }


            _mLDataLoader.SaveModel(mlModel, data.Schema, ModelTypeName, settings.ModelName);


            return _mlContext.Regression
                    .CrossValidate(
                        data,
                        trainingPipeline,
                        numberOfFolds: settings.CrossValidationFoldsCount,
                        labelColumnName: settings.LabelColumnName);
        }

        private ITransformer CreateModelWithFeatureContributionMetrics(IDataView data, ColumnConcatenatingEstimator transformPipeline, FastForestRegressionTrainer trainer)
        {
            ITransformer mlModel;
            var transformer = transformPipeline.Fit(data);
            var transformedData = transformer.Transform(data);
            var linearModel = trainer.Fit(transformedData);
            var linearFeatureContributionCalculator = _mlContext.Transforms
                                        .CalculateFeatureContribution(linearModel, normalize: true)
                                        .Fit(transformedData);
            mlModel = transformer.Append(linearModel)
                                 .Append(linearFeatureContributionCalculator);
            return mlModel;
        }
    }
}
