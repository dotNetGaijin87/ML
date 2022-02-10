using Microsoft.AspNetCore.Http;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ML.Services
{
    /// <summary>
    /// Interface for <class cref="TrainingDataService"/> 
    /// </summary>
    public interface ITrainingDataService
    {
        Task<bool> Create(IEnumerable<IFormFile> Files, string targetModelTypeName);
        bool Delete(string fileName, string targetModelTypeName);
    }
}