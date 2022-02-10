using Microsoft.AspNetCore.Http;
using ML.Common;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace ML.Services
{
    /// <summary>
    /// Class that manages training files for ai algorithms
    /// </summary>
    public class TrainingDataService : ITrainingDataService
    {
        private readonly IDataPathRegister _dataPathRegister;

        public TrainingDataService(IDataPathRegister dataPathRegister)
        {
            _dataPathRegister = dataPathRegister;
        }

        /// <summary>
        /// Creates training file for specified ai model type.
        /// </summary>
        /// <param name="Files"></param>
        /// <param name="targetModelTypeName"></param>
        /// <returns></returns>
        public async Task<bool> Create(IEnumerable<IFormFile> Files, string targetModelTypeName)
        {
            var preconditionCheck = FilesAlreadyExist(Files, targetModelTypeName);
            if (!preconditionCheck.IsValid)
                return false;


            foreach (IFormFile file in Files)
            {
                var filePath = _dataPathRegister.GetTrainingDataFilePath(targetModelTypeName, file.FileName);
                if (file.Length > 0)
                {
                    using (Stream fileStream = new FileStream(filePath, FileMode.Create))
                    {
                        await file.CopyToAsync(fileStream);
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Deletes a training file
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="targetModelTypeName">AI model type name</param>
        /// <returns></returns>
        public bool Delete(string fileName, string targetModelTypeName)
        {
            var filePath = _dataPathRegister.GetTrainingDataFilePath(targetModelTypeName, fileName);
            if (!File.Exists(filePath))
            {
                return false;
            }

            File.Delete(filePath);
            return true;
        }

        /// <summary>
        /// Checks if any of the files already exist. If that is the case returns and
        /// eventually no file is created.
        /// </summary>
        /// <param name="files"></param>
        /// <param name="targetModelTypeName"></param>
        /// <returns></returns>
        private (bool IsValid, string FileName) FilesAlreadyExist(IEnumerable<IFormFile> files, string targetModelTypeName)
        {

            foreach (IFormFile file in files)
            {
                var filePath = _dataPathRegister.GetTrainingDataFilePath(targetModelTypeName, file.FileName);
                if (File.Exists(filePath))
                {
                    return (false, file.FileName);
                }
            }

            return (true, "");
        }
    }
}
