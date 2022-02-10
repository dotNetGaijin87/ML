using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using ML.Helpers.Exceptions;
using ML.Models;
using ML.Models.Regression.FastForest;
using ML.Services;
using System;
using System.Collections.Generic;
using System.Net;
using System.Text.Json;

namespace ML.Controllers
{
    /// <summary>
    /// This controller manages endpoints for data analysis based on 
    /// FastForest regression alogrithm. 
    /// </summary>
    [ApiController]
    [Route("[controller]")]
    //[Route("fast-forest")]
    public class FastForestController : ControllerBase
    {
        private IFastForestService _fastForestService;

        public FastForestController(IFastForestService fastForestService)
        {
            _fastForestService = fastForestService;
        }

        /// <summary>
        /// Creates Fast Forest model resource
        /// </summary>
        /// <param name="settings">Training parameters for the model</param>
        /// <returns>Regression metrics of the created model</returns>
        [HttpPost]
        public IActionResult Create(FastForestModelBuilderSettings settings)
        {
            try
            {
                List<RegressionMetrics> metrics = _fastForestService.CreateModelAndValidate(settings);
                return Ok(JsonSerializer.Serialize(metrics));
            }
            catch (TrainingDataFileNotExistException)
            {
                return NotFound("Training data not found");
            }
            catch (SavingModelFileException ex)
            {
                return StatusCode((int)HttpStatusCode.InternalServerError, $"Error while saving the model: {ex.Message}");
            }
            catch (Exception ex)
            {
                return StatusCode((int)HttpStatusCode.InternalServerError, $"Error while creating the model: {ex.Message}");
            }
        }

        /// <summary>
        /// Deletes model resource
        /// </summary>
        /// <param name="modelName">Name of the model to be delet</param>
        /// <returns>Text containing result of the operation</returns>
        [HttpDelete]
        public IActionResult Delete(string modelName)
        {
            bool IsDeleted = _fastForestService.Delete(modelName);

            return IsDeleted
                ? Ok("Successfully deleted")
                : BadRequest("File does not exist");
        }

        /// <summary>
        /// Custom function for running already created prediction engine.
        /// </summary>
        /// <param name="modelName">Name of the model to be run</param>
        /// <param name="input">Input data that the prediction be based on</param>
        /// <returns>Prediction score</returns>
        [HttpPost("/Regression/[controller]:Run")]
        public IActionResult Run(string modelName, FastForestPredictionInput input)
        {
            try
            {
                FastForestPredictionOutput prediction = _fastForestService.Run(modelName, input);

                if(prediction.FeatureContributions == null)
                {
                    return Ok(prediction.Score.ToString());
                }
                else
                {
                    return Ok( new[]
                            {
                                prediction.Score.ToString(),
                                JsonSerializer.Serialize(prediction.Features),
                                JsonSerializer.Serialize(prediction.FeatureContributions)
                            });
                }
            }
            catch (ModelFileLoadingException ex)
            {
                return StatusCode((int)HttpStatusCode.InternalServerError, $"Model file loading error: {ex.Message}");
            }
            catch (ModelFileNotExistException)
            {
                return NotFound("Model file does not exist");
            }
            catch (Exception ex)
            {
                return StatusCode((int)HttpStatusCode.InternalServerError, $"Error while running prediction engine: {ex.Message}");
            }
        }

    }
}

