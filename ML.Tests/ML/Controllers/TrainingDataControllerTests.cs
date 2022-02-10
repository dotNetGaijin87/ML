using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using ML.Controllers;
using ML.Controllers.TrainingData;
using ML.Services;
using NSubstitute;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace ML.Tests.ML.Controllers
{
    [TestFixture]
    public class TrainingDataControllerTests
    {
        [Test]
        public async Task Create_CallsService_CallReceived()
        {
            // ARRANGE
            ITrainingDataService trainingDataService = Substitute.For<ITrainingDataService>();
            TrainingDataController trainingDataController = new TrainingDataController(trainingDataService);
            IFormFile formFile = Substitute.For<IFormFile>();
            formFile.FileName.Returns("File1");
            var model = new FileModel();
            model.FormFiles = new List<IFormFile> { formFile };
            model.ModelTypeName = "data";

            // ACT
            ObjectResult response = (await trainingDataController.Create(model)) as ObjectResult;

            // ASSERT
            await trainingDataService.Received().Create(Arg.Is<List<IFormFile>>( x => x.First().FileName == "File1"), Arg.Is<string>( x => x == "data"));
        }

        [Test]
        public async Task Create_AlwaysTrue_ReturnsOK()
        {
            // ARRANGE
            ITrainingDataService trainingDataService = Substitute.For<ITrainingDataService>();
            TrainingDataController trainingDataController = new TrainingDataController(trainingDataService);
            trainingDataService.Create(default, default).ReturnsForAnyArgs(true);

            // ACT
            ObjectResult response = (await trainingDataController.Create(new FileModel())) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.OK, response.StatusCode);
        }

        [Test]
        public async Task Create_AlwaysFalse_ReturnsBadRequest()
        {
            // ARRANGE
            ITrainingDataService trainingDataService = Substitute.For<ITrainingDataService>();
            TrainingDataController trainingDataController = new TrainingDataController(trainingDataService);
            trainingDataService.Create(default, default).ReturnsForAnyArgs(false);

            // ACT
            ObjectResult response = (await trainingDataController.Create(new FileModel())) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.BadRequest, response.StatusCode);
        }

        [Test]
        public async Task Create_ThrowsException_ReturnsInternalServerError()
        {
            // ARRANGE
            ITrainingDataService trainingDataService = Substitute.For<ITrainingDataService>();
            TrainingDataController trainingDataController = new TrainingDataController(trainingDataService);
            trainingDataService.Create(default, default).Returns(Task.FromException<bool>(new Exception("")));

            // ACT
            ObjectResult response = (await trainingDataController.Create(new FileModel())) as ObjectResult;

            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.InternalServerError, response.StatusCode);
            Assert.AreEqual(true, response.Value.ToString().ToLower().Contains("saving"));
        }

    }
}
