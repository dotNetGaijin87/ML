using Microsoft.ML.TimeSeries;
using ML.Models;
using ML.Services;
using NSubstitute;
using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;

namespace ML.Tests
{
    [TestFixture]
    public class SrCnnServiceTests
    {
        private ISrCnnService _srCnnService;
        private ISrCnnTrainer _srCnnTrainer;


        [SetUp]
        public void Setup()
        {
            _srCnnTrainer = Substitute.For<ISrCnnTrainer>();
            _srCnnService = new SrCnnService(_srCnnTrainer);
        }

        [Test]
        public void Predict_ValidTrainingData_RunCalledWithTrainingData()
        {
            // ARRANGE
            var input = new SrCnnInput() 
            {
                Threshold = 0.2,
                BatchSize = 10,
                Sensitivity = 99,
                Period = 20,
                TrainingData = new List<double>() { 1,2,3} 
            };

            // ACT
            _srCnnService.Predict(input);

            // ASSERT
            _srCnnTrainer.Received().Run(Arg.Is<SrCnnTrainerInputCollection>( p => p.Values.ToList()[1].Value == 2), Arg.Any<SrCnnEntireAnomalyDetectorOptions>());
            _srCnnTrainer.Received().Run(Arg.Any<SrCnnTrainerInputCollection>(), Arg.Is<SrCnnEntireAnomalyDetectorOptions>(p => p.Threshold == 0.2));
            _srCnnTrainer.Received().Run(Arg.Any<SrCnnTrainerInputCollection>(), Arg.Is<SrCnnEntireAnomalyDetectorOptions>(p => p.BatchSize == 10));
            _srCnnTrainer.Received().Run(Arg.Any<SrCnnTrainerInputCollection>(), Arg.Is<SrCnnEntireAnomalyDetectorOptions>(p => p.Sensitivity == 99));
            _srCnnTrainer.Received().Run(Arg.Any<SrCnnTrainerInputCollection>(), Arg.Is<SrCnnEntireAnomalyDetectorOptions>(p => p.Period == 20));

        }
    }
}