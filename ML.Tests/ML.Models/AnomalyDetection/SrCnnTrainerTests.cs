using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using ML.Models;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Tests.ML.Models.AnomalyDetection
{
    [TestFixture]
    public class SrCnnTrainerTests
    {
        ISrCnnTrainer _srCnnTrainer;


        [Test]
        public void Run_ValidValues_ReturnsDataSuccess()
        {
            // ARRANGE
            _srCnnTrainer = new SrCnnTrainer(new MLContext());
            var trainingData = new SrCnnTrainerInputCollection
            {
                Values = new List<SrCnnTrainerInput>() 
                { 
                    new SrCnnTrainerInput { Value = 1 } , new SrCnnTrainerInput { Value = 1 } , 
                    new SrCnnTrainerInput { Value = 2 } , new SrCnnTrainerInput { Value = 2 } , 
                    new SrCnnTrainerInput { Value = 3 } , new SrCnnTrainerInput { Value = 3 } ,
                    new SrCnnTrainerInput { Value = 4 } , new SrCnnTrainerInput { Value = 1 } ,
                    new SrCnnTrainerInput { Value = 5 } , new SrCnnTrainerInput { Value = 2 } ,
                    new SrCnnTrainerInput { Value = 6 } , new SrCnnTrainerInput { Value = 3 } ,
                }
            };
            var options = new SrCnnEntireAnomalyDetectorOptions()
            {
                BatchSize = -1,
                DeseasonalityMode = SrCnnDeseasonalityMode.Stl,
                DetectMode = SrCnnDetectMode.AnomalyOnly,
                Period = 0,
                Sensitivity = 99,
                Threshold = 0.3
            };


            // ACT
            var result = _srCnnTrainer.Run(trainingData, options);


            // ASSERT
            Assert.IsNotNull(result.First().Prediction);
        }


        [Test]
        public void Run_inputNullValue_ThrowsArgumentNullException()
        {
            // ARRANGE
            _srCnnTrainer = new SrCnnTrainer(new MLContext());


            // ACT
            // ASSERT
            Assert.That(() => _srCnnTrainer.Run(null, new SrCnnEntireAnomalyDetectorOptions()),
                Throws.Exception
                    .TypeOf<ArgumentNullException>()
                    .With.Property("ParamName")
                    .EqualTo("input"));
        }

        [Test]
        public void Run_optionsNullValue_ThrowsArgumentNullException()
        {
            // ARRANGE
            _srCnnTrainer = new SrCnnTrainer(new MLContext());


            // ACT
            // ASSERT
            Assert.That(() => _srCnnTrainer.Run(new SrCnnTrainerInputCollection(), null),
                Throws.Exception
                    .TypeOf<ArgumentNullException>()
                    .With.Property("ParamName")
                    .EqualTo("options"));
        }

        [Test]
        [TestCase(-1)]
        [TestCase(1.1)]
        public void Run_ThresholdInvalidValue_ThrowsArgumentException1(double threshold)
        {
            // ARRANGE
            _srCnnTrainer = new SrCnnTrainer(new MLContext());
            var options = new SrCnnEntireAnomalyDetectorOptions
            { 
                Threshold = threshold,    
            };


            // ACT
            // ASSERT
            Assert.That(() => _srCnnTrainer.Run(new SrCnnTrainerInputCollection(), options),
                Throws.Exception.TypeOf<ArgumentException>());
        }


        [Test]
        [TestCase(-1)]
        [TestCase(101)]
        public void Run_SensitivityInvalidValue_ThrowsArgumentException(double sensitivity)
        {
            // ARRANGE
            _srCnnTrainer = new SrCnnTrainer(new MLContext());
            var options = new SrCnnEntireAnomalyDetectorOptions
            {
                Sensitivity = sensitivity,
            };


            // ACT
            // ASSERT
            Assert.That(() => _srCnnTrainer.Run(new SrCnnTrainerInputCollection(), options),
                Throws.Exception.TypeOf<ArgumentException>());
        }

    }
}
