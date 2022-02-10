using Microsoft.AspNetCore.Http;
using ML.Common;
using ML.Services;
using NSubstitute;
using NUnit.Framework;
using System.Collections.Generic;

namespace ML.Tests.ML.Services.TrainingDataInput
{
    [TestFixture]
    public class TrainingDataServiceTests
    {
        IDataPathRegister _dataPathRegister;
        IFormFile _formFile;
        ITrainingDataService _trainingDataService;

        [SetUp]
        public void Setup()
        {
            _dataPathRegister = Substitute.For<IDataPathRegister>();
            _formFile = Substitute.For<IFormFile>();
            _trainingDataService = new TrainingDataService(_dataPathRegister);
        }



        [Test]
        public void Create_ValidData_GetTrainingDataFilePathCalled()
        {
            // ARRANGE
            _formFile.FileName.Returns("File1");


            // ACT
            _trainingDataService.Create(new List<IFormFile> { _formFile }, "ModelType1");


            // ASSERT
            _dataPathRegister.Received().GetTrainingDataFilePath(
                                            Arg.Is<string>(x => x == "ModelType1"),
                                            Arg.Is<string>(x => x == "File1"));
        }


        [Test]
        public void Delete_ValidData_GetTrainingDataFilePathCalled()
        {
            // ARRANGE


            // ACT
            _trainingDataService.Delete("File1", "ModelType1");


            // ASSERT
            _dataPathRegister.Received().GetTrainingDataFilePath(
                                            Arg.Is<string>(x => x == "ModelType1"),
                                            Arg.Is<string>(x => x == "File1"));

        }
    }
}
