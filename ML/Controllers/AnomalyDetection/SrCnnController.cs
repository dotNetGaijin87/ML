using Microsoft.AspNetCore.Mvc;
using ML.Models;
using ML.Services;
using System;
using System.Net;

namespace ML.Controllers
{
    /// <summary>
    /// This controller manages endpoints for anomaly detection based on 
    /// Super-resolution using convolutional neural network (SRCNN) algorithm
    /// </summary>
    [ApiController]
    [Route("AnomalyDetection")]
    public class SrCnnController : ControllerBase
    {
        private readonly ISrCnnService _srCnnService;

        public SrCnnController(ISrCnnService srCnnService)
        {
            _srCnnService = srCnnService;
        }

        /// <summary>
        /// Custom method for running anomaly detection of provided data.
        /// </summary>
        /// <param name="model">Contains training data and training parameters</param>
        /// <returns>Returns anomaly score and anomaly flag for each training data point provided</returns>
        [HttpPost("[controller]:Run")]
        public IActionResult Get(SrCnnInput model)
        {
            try
            {
                var prediction = _srCnnService.Predict(model);
                return Ok(prediction);
            }
            catch (Exception ex)
            {
                return StatusCode((int)HttpStatusCode.InternalServerError, $"Could not run prediction engine: {ex.Message}");
            }

        }
    }
}
