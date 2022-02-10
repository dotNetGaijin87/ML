using System.IO;

namespace ML.Common
{
    /// <summary>
    /// Class containig file paths for training data and model files.
    /// Each model and training data files are divided based on algorithms the are used for.
    /// ex. Models/FastFores/model1.zip, 
    /// ex. TrainingData/SrCnn/sampleData.csv
    /// </summary>
    public class DataPathRegister : IDataPathRegister
    {
        private readonly string TrainingData;
        private readonly string Models;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainingData">Path for storing training data</param>
        /// <param name="models">Path for storing model files</param>
        public DataPathRegister(string trainingData, string models)
        {
            TrainingData = trainingData;
            Models = models;
        }

        public string GetModelFilePath(string modelTypeName, string modelName)
        {
            return Path.Combine(Models, modelTypeName, $"{modelName}.zip");
        }

        public string GetTrainingDataFilePath(string modelTypeName, string fileName)
        {
            return Path.Combine(TrainingData, modelTypeName, fileName);
        }
    }
}
