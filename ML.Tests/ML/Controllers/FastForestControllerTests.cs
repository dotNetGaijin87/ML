using Microsoft.AspNetCore.Mvc;
using ML.Controllers;
using ML.Helpers.Exceptions;
using ML.Models;
using ML.Models.Regression.FastForest;
using ML.Services;
using NSubstitute;
using NUnit.Framework;
using System;
using System.Linq;
using System.Net;
using System.Text.Json;

namespace ML.Tests.ML.Controllers
{
    [TestFixture]
    public class FastForestControllerTests
    {
        [Test]
        public void Create_ValidData_CallReceived()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            var input = new FastForestModelBuilderSettings 
            {  
                ModelName = "Model", 
                TreesCount = 100,
                LeavesCount = 10,
                MinimumExampleCountPerLeaf = 5,
                FeatureColumnNames = new string[] {"col1","col2" }, 
                LabelColumnName = "col10",
                TrainigDataName = "sample.csv",
                CrossValidationFoldsCount = 5,
                HasFeatureContributionMetrics = true
            };

            // ACT
            ffController.Create(input);

            // ASSERT
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.ModelName == "Model"));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.TreesCount == 100));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.LeavesCount == 10));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.MinimumExampleCountPerLeaf == 5));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.FeatureColumnNames.ToList().Last() == "col2"));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.LabelColumnName == "col10"));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.TrainigDataName == "sample.csv"));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.CrossValidationFoldsCount == 5));
            ffService.Received().CreateModelAndValidate(Arg.Is<FastForestModelBuilderSettings>(p => p.HasFeatureContributionMetrics == true));
        }

        [Test]
        public void Create_ThrowsTrainingDataFileNotExistException_ReturnsNotFound()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            var ffController = new FastForestController(ffService);
            ffService.CreateModelAndValidate(default).ReturnsForAnyArgs(x => { throw new TrainingDataFileNotExistException(); });

            // ACT
            ObjectResult response = ffController.Create(default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.NotFound, response.StatusCode);
        }

        [Test]
        public void Create_ThrowsSavingModelFileException_ReturnsInternalServerError()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            var ffController = new FastForestController(ffService);
            ffService.CreateModelAndValidate(default).ReturnsForAnyArgs(x => { throw new SavingModelFileException(); });

            // ACT
            ObjectResult response = ffController.Create(default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.InternalServerError, response.StatusCode);
            Assert.AreEqual(true, response.Value.ToString().ToLower().Contains("saving"));
        }

        [Test]
        public void Create_ThrowsException_ReturnsInternalServerError()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            var ffController = new FastForestController(ffService);
            ffService.CreateModelAndValidate(default).ReturnsForAnyArgs(x => { throw new Exception(); });

            // ACT
            ObjectResult response = ffController.Create(default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.InternalServerError, response.StatusCode);
            Assert.AreEqual(true, response.Value.ToString().ToLower().Contains("creating"));
        }


        [Test]
        public void Delete_CallReceived()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);

            // ACT
            ffController.Delete("model");

            // ASSERT
            ffService.Received().Delete("model");
        }

        [Test]
        public void Delete_AlwaysTrue_ReturnsOK()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            ffService.Delete(default).ReturnsForAnyArgs(true);

            // ACT
            ObjectResult response = ffController.Delete(default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.OK, response.StatusCode);
        }

        [Test]
        public void Delete_AlwaysFalse_ReturnsBadRequest()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            ffService.Delete(default).ReturnsForAnyArgs(false);

            // ACT
            ObjectResult response = ffController.Delete(default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.BadRequest, response.StatusCode);
        }

        
        [Test]
        public void Run_CallReceivedWithCorrectData()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            string modelName = "modelName";
            var input = new FastForestPredictionInput() { Col1 = 1, Col38 = 2 };

            // ACT
            ffController.Run(modelName, input);

            // ASSERT
            ffService.Received().Run(Arg.Is<string>( x => x == "modelName"),Arg.Any<FastForestPredictionInput>());
            ffService.Received().Run(Arg.Any<string>(),Arg.Is<FastForestPredictionInput>(x => x.Col1 == 1));
            ffService.Received().Run(Arg.Any<string>(), Arg.Is<FastForestPredictionInput>(x => x.Col38 == 2));
        }


        [Test]
        public void Run_ValidData_ReturnsOKStatus()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            ffService.Run(default, default).ReturnsForAnyArgs(new FastForestPredictionOutput 
                                                            { 
                                                                Score = 1.1f, 
                                                                Features = new float[] {2.1f, 2.2f },
                                                                FeatureContributions = new float[] {0.3f, 0.7f } 
                                                            });

            // ACT
            ObjectResult response = ffController.Run(default, default) as ObjectResult;
            var serializedResponse = JsonSerializer.Serialize(response.Value);

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.OK, response.StatusCode);
            Assert.AreEqual("[\"1.1\",\"[2.1,2.2]\",\"[0.3,0.7]\"]", serializedResponse);
        }


        [Test]
        public void Run_ThrowsModelFileLoadingException_ReturnsInternalServerError()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            ffService.Run(default, default).ReturnsForAnyArgs(x => { throw new ModelFileLoadingException(); });

            // ACT
            ObjectResult response = ffController.Run(default, default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.InternalServerError, response.StatusCode);
            Assert.AreEqual(true, response.Value.ToString().ToLower().Contains("loading"));
        }

        [Test]
        public void Run_ThrowsModelFileNotExistException_ReturnsNotFound()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            ffService.Run(default, default).ReturnsForAnyArgs(x => { throw new ModelFileNotExistException(); });

            // ACT
            ObjectResult response = ffController.Run(default, default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.NotFound, response.StatusCode);
            Assert.AreEqual(true, response.Value.ToString().ToLower().Contains("exist"));
        }

        [Test]
        public void Run_ThrowsException_ReturnsInternalServerError()
        {
            // ARRANGE
            IFastForestService ffService = Substitute.For<IFastForestService>();
            FastForestController ffController = new FastForestController(ffService);
            ffService.Run(default, default).ReturnsForAnyArgs(x => { throw new Exception(); });

            // ACT
            ObjectResult response = ffController.Run(default, default) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.InternalServerError, response.StatusCode);
            Assert.AreEqual(true, response.Value.ToString().ToLower().Contains("running"));
        }
    }
}
