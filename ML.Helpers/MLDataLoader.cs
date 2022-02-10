using Microsoft.ML;
using ML.Common;
using ML.Helpers.Exceptions;
using System;
using System.Collections.Generic;
using System.IO;

namespace ML.Helpers
{
    /// <summary>
    /// Class used for loading and saving training data files and AI models
    /// </summary>
    public class MLDataLoader : IMLDataLoader
    {
        private readonly MLContext _mlContext;
        private readonly IDataPathRegister _dataPathRegister;


        public MLDataLoader(MLContext mlContext, IDataPathRegister dataPathRegister)
        {
            _mlContext = mlContext;
            _dataPathRegister = dataPathRegister;
        }

        /// <summary>
        /// Saves given model
        /// </summary>
        /// <param name="mlModel">Model to be saved</param>
        /// <param name="modelInputSchema">ML.NET MLContext object</param>
        /// <param name="modelTypeName">Name of the type of the algorithm that is used in the model</param>
        /// <param name="modelName">Name of the trained model</param>
        /// <returns></returns>
        public bool SaveModel(ITransformer mlModel, DataViewSchema modelInputSchema, string modelTypeName, string modelName)
        {
            string path = _dataPathRegister.GetModelFilePath(modelTypeName, modelName);
            try
            {
                _mlContext.Model.Save(mlModel, modelInputSchema, path);
                return true;
            }
            catch (Exception)
            {
                throw new SavingModelFileException();
            }
        }

        /// <summary>
        /// Loads a model based on provided name and type
        /// </summary>
        /// <param name="modelTypeName">Name of the type of the algorithm that is used in the model</param>
        /// <param name="modelName">Name of the trained model</param>
        /// <returns>ITransformer object that is used for creating ML.NET prediction engine</returns>
        public ITransformer LoadModel(string modelTypeName, string modelName)
        {
            string path = _dataPathRegister.GetModelFilePath(modelTypeName, modelName);
            try
            {
                return _mlContext.Model.Load(path, out var modelInputSchema);
            }
            catch (Exception)
            {
                if (!File.Exists(path))
                {
                    throw new ModelFileNotExistException();
                }
                throw new ModelFileLoadingException();
            }
        }


        public IDataView LoadIDataView<T>(IEnumerable<T> data) where T : class
        {
            return _mlContext.Data.LoadFromEnumerable<T>(data);
        }

        /// <summary>
        /// Loads training data from a file, given its type and name
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="modelTypeName"></param>
        /// <param name="fileName"></param>
        /// <returns>IDataView object that can be fed directly intto ML.NET pipline</returns>
        public IDataView LoadDataFromFile<T>(string modelTypeName, string fileName) where T : class
        {
            string path = _dataPathRegister.GetTrainingDataFilePath(modelTypeName, fileName);

            try
            {
                return _mlContext.Data.LoadFromTextFile<T>(path: path, hasHeader: true, separatorChar: ',');
            }
            catch (Exception)
            {
                throw new TrainingDataFileNotExistException();
            }
        }

    }
}
