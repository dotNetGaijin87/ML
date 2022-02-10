using Microsoft.ML;
using Microsoft.ML.Data;
using ML.Common;
using ML.Helpers;
using ML.Models;
using ML.Models.Regression.FastForest;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.Services
{
    /// <summary>
    /// Class that manages data analysis based on FastForest regression alogrithm. 
    /// </summary>
    public class FastForestService : IFastForestService
    {
        private readonly IFastForestModelBuilder _ffModelBuilder;
        private readonly IFastForestPrediction _fastForestPrediction;
        private readonly IMLDataLoader _mLDataLoader;
        private readonly IDataPathRegister _dataPathRegister;

        public FastForestService(
                IFastForestModelBuilder ffmodelBuilder, 
                IFastForestPrediction fastForestPrediction,
                IMLDataLoader mLDataLoader,
                IDataPathRegister dataPathRegister)
        {
            _ffModelBuilder = ffmodelBuilder;
            _fastForestPrediction = fastForestPrediction;
            _mLDataLoader = mLDataLoader;
            _dataPathRegister = dataPathRegister;
        }

        /// <summary>
        /// Creates and validates Fast Forest model
        /// </summary>
        /// <param name="settings">Training parameters for the model/param>
        /// <returns>Regression metrics of the created model</returns>
        public List<RegressionMetrics> CreateModelAndValidate(FastForestModelBuilderSettings settings)
        {
            IDataView trainingDataView = _mLDataLoader.LoadDataFromFile<FastForestPredictionInput>(_ffModelBuilder.ModelTypeName, settings.TrainigDataName);


            var validationResults = _ffModelBuilder.Run(trainingDataView, settings);
            return validationResults.Select(x => x.Metrics).ToList();
        }

        /// <summary>
        /// Deletes Fast Forest model
        /// </summary>
        /// <param name="modelName"></param>
        /// <returns></returns>
        public bool Delete(string modelName)
        {
            var pathToTrainingFile = _dataPathRegister.GetModelFilePath(_ffModelBuilder.ModelTypeName, modelName);
            if (!File.Exists(pathToTrainingFile))
            {
                return false;
            }

            File.Delete(pathToTrainingFile);
            return true;
        }

        /// <summary>
        /// Runs Fast Forest prediction engine 
        /// </summary>
        /// <param name="modelName"></param>
        /// <param name="input"></param>
        /// <returns>prediction's engine score and performance metrics</returns>
        public FastForestPredictionOutput Run(string modelName, FastForestPredictionInput input)
        {
            ITransformer model = _mLDataLoader.LoadModel(_ffModelBuilder.ModelTypeName, modelName);

            var output = _fastForestPrediction.Run(model, input);

            return output;
        }
    }
}
