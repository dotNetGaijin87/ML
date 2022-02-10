using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using ML.Controllers.TrainingData;
using ML.Services;
using System;
using System.Collections.Generic;
using System.Net;
using System.Threading.Tasks;

namespace ML.Controllers
{
    /// <summary>
    /// This controller manages endpoints for storing and deleting data files used for training AI algorithms.
    /// </summary>
    [ApiController]
    //[Route("TrainingData")]
    [Route("[controller]")]
    public class TrainingDataController : ControllerBase
    {
        readonly ITrainingDataService _trainingDataService;

        public TrainingDataController(ITrainingDataService trainingDataService)
        {
            _trainingDataService = trainingDataService;
        }

        /// <summary>
        /// Creates/Saves training data file 
        /// </summary>
        /// <param name="files">Files containing training data</param>
        /// <param name="modelTypeName">Type of the model that will use the provided training data</param>
        /// <returns></returns>
        [HttpPost]
        public async Task<IActionResult> Create([FromForm] FileModel model)
        {
            try
            {
                var createSuccess = await _trainingDataService.Create(model.FormFiles, model.ModelTypeName);
                if (createSuccess)
                {
                    return Ok("");
                }
                else
                {
                    return BadRequest($"One of the files already exist error");
                }
            }
            catch (Exception ex)
            {
                return StatusCode((int)HttpStatusCode.InternalServerError, $"Error while saving data: {ex.Message}");
            }
        }

        /// <summary>
        /// Deletes training data file
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="modelTypeName"></param>
        /// <returns></returns>
        [HttpDelete]
        public IActionResult Delete(string fileName, string modelTypeName)
        {
            try
            {
                var createSuccess = _trainingDataService.Delete(fileName, modelTypeName);
                if (createSuccess)
                {
                    return Ok("");
                }
                else
                {
                    return BadRequest($"File does not exist error");
                }
            }
            catch (Exception ex)
            {
                return StatusCode((int)HttpStatusCode.InternalServerError, $"Error while deleting data: {ex.Message}");
            }
        }

    }
}
