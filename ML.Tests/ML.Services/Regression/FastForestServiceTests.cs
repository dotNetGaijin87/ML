using Microsoft.ML;
using Microsoft.ML.Data;
using ML.Common;
using ML.Helpers;
using ML.Models;
using ML.Models.Regression.FastForest;
using ML.Services;
using NSubstitute;
using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.TrainCatalogBase;

namespace ML.Tests.ML.Services.Regression
{
    [TestFixture]
    public class FastForestServiceTests
    {
        IFastForestModelBuilder _fastForestModelBuilder;
        IFastForestPrediction _fastForestPrediction;
        IMLDataLoader _mLDataLoader;
        IDataPathRegister _dataPathRegister;

        FastForestService _fastForestService;

        [SetUp]
        public void Setup()
        {
            _fastForestModelBuilder = Substitute.For<IFastForestModelBuilder>();
            _fastForestPrediction = Substitute.For<IFastForestPrediction>();
            _mLDataLoader = Substitute.For<IMLDataLoader>();
            _dataPathRegister = Substitute.For<IDataPathRegister>();
        }

        [Test]
        public void CreateModelAndValidate_LoadDataFromFileCalled()
        {
            // ARRANGE
            _fastForestModelBuilder.ModelTypeName.Returns("TempModelTypeName");
            _fastForestService = new FastForestService(_fastForestModelBuilder, _fastForestPrediction, _mLDataLoader, _dataPathRegister);


            // ACT
            _fastForestService.CreateModelAndValidate(new FastForestModelBuilderSettings() { TrainigDataName = "DummyFileName" });


            // ASSERT
            _mLDataLoader.Received().LoadDataFromFile<FastForestPredictionInput>("TempModelTypeName", "DummyFileName");
        }

        [Test]
        public void CreateModelAndValidate_ValidModelName_RunCalledWithProperData()
        {
            // ARRANGE
            var settings = new FastForestModelBuilderSettings() 
            { 
                ModelName = "DummyModel",
                TreesCount = 500,
                LeavesCount = 10,
                MinimumExampleCountPerLeaf = 5,
                LabelColumnName = "col1",
                FeatureColumnNames = new string[] { "col1", "col2" },
                TrainigDataName = "DummyTrainingDataName",
                CrossValidationFoldsCount = 3,
                HasFeatureContributionMetrics = true

            };
            _fastForestService = new FastForestService(_fastForestModelBuilder, _fastForestPrediction, _mLDataLoader, _dataPathRegister);


            // ACT
            _fastForestService.CreateModelAndValidate(settings);


            // ASSERT
            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>( x => x.ModelName == "DummyModel"));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.LeavesCount == 10));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.MinimumExampleCountPerLeaf == 5));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.TreesCount == 500));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.LabelColumnName == "col1"));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.FeatureColumnNames[1] == "col2"));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.CrossValidationFoldsCount == 3));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.TrainigDataName == "DummyTrainingDataName"));

            _fastForestModelBuilder.Received().Run(
                    Arg.Any<IDataView>(), Arg.Is<FastForestModelBuilderSettings>(x => x.HasFeatureContributionMetrics == true));

        }

        [Test]
        public void CreateModelAndValidate_ValidData_ReturnsOne()
        {
            // ARRANGE
            var settings = new FastForestModelBuilderSettings() { TrainigDataName = "DummyTrainingDataName" };
            var resultMock = Substitute.For<IReadOnlyList<CrossValidationResult<RegressionMetrics>>>();
            _fastForestService = new FastForestService(_fastForestModelBuilder, _fastForestPrediction, _mLDataLoader, _dataPathRegister);
            _fastForestModelBuilder.Run(default, default).ReturnsForAnyArgs(resultMock);


            // ACT
            List<RegressionMetrics> output = _fastForestService.CreateModelAndValidate(settings);


            // ASSERT
            resultMock.Received().ToList();
        }



        [Test]
        public void Delete_ValiData_LoadModelCalled()
        {
            // ARRANGE
            _fastForestModelBuilder.ModelTypeName.Returns("ModelType");
            _fastForestService = new FastForestService(_fastForestModelBuilder, _fastForestPrediction, _mLDataLoader, _dataPathRegister);


            // ACT
            _fastForestService.Delete("Model");


            // ASSERT
            _dataPathRegister.Received().GetModelFilePath(
                                                Arg.Is<string>( x => x == "ModelType"),
                                                Arg.Is<string>(x => x == "Model"));
        }



        [Test]
        public void Run_ValidDAta_LoadModelCalled()
        {
            // ARRANGE
            _fastForestModelBuilder.ModelTypeName.Returns("ModelType");
            _fastForestService = new FastForestService(_fastForestModelBuilder, _fastForestPrediction, _mLDataLoader, _dataPathRegister);


            // ACT
            _fastForestService.Run("Model", new FastForestPredictionInput());


            // ASSERT
            _mLDataLoader.Received().LoadModel("ModelType", "Model");
        }

        [Test]
        public void Run_ValidData_RunCalled()
        {
            // ARRANGE
            _fastForestService = new FastForestService(_fastForestModelBuilder, _fastForestPrediction, _mLDataLoader, _dataPathRegister);


            // ACT
            _fastForestService.Run("Model", new FastForestPredictionInput() { Col1 = 1 });


            // ASSERT
            _fastForestPrediction.Received().Run(
                                                Arg.Any<ITransformer>(), 
                                                Arg.Is<FastForestPredictionInput>( x => x.Col1 == 1));
        }

        [Test]
        public void Run_ValidData_ReturnsOne()
        {
            // ARRANGE
            _fastForestService = new FastForestService(_fastForestModelBuilder, _fastForestPrediction, _mLDataLoader, _dataPathRegister);
            _fastForestPrediction.Run(default, default).ReturnsForAnyArgs(new FastForestPredictionOutput() { Score = 1 });
  

            // ACT
            var output = _fastForestService.Run("Model", new FastForestPredictionInput() { Col1 = 1 });


            // ASSERT
            Assert.AreEqual(1.0, output.Score);
        }

    }
}
