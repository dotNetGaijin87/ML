using Microsoft.AspNetCore.Mvc;
using ML.Controllers;
using ML.Models;
using ML.Services;
using NSubstitute;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Net;

namespace ML.Tests.ML.Controllers
{
    [TestFixture]
    public class SrCnnControllerTests
    {
        [Test]
        public void Get_ValidData_ReturnsOk()
        {
            // ARRANGE
            ISrCnnService srCnnService = Substitute.For<ISrCnnService>();
            SrCnnController srCnnController = new SrCnnController(srCnnService);
            var returnData = new List<SrCnnOutput>()
            {
                new SrCnnOutput { RawScore = 0, IsAnomaly = 0, Mag = 2 },
                new SrCnnOutput { RawScore = 1, IsAnomaly = 1, Mag = 3 }
            };
            srCnnService.Predict(default).ReturnsForAnyArgs(returnData);


            // ACT
            ObjectResult response = srCnnController.Get(new SrCnnInput()) as ObjectResult; 


            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.OK, response.StatusCode);
            Assert.AreEqual(response.Value, returnData);
        }

        [Test]
        public void Get_ThrowsException_ReturnsInternalServerError()
        {
            // ARRANGE
            ISrCnnService srCnnService = Substitute.For<ISrCnnService>();
            SrCnnController srCnnController = new SrCnnController(srCnnService);
            srCnnService.Predict(default).ReturnsForAnyArgs(x => { throw new Exception(); });


            // ACT
            ObjectResult response = srCnnController.Get(new SrCnnInput()) as ObjectResult;


            // ASSERT
            Assert.IsNotNull(response);
            Assert.AreEqual((int)HttpStatusCode.InternalServerError, response.StatusCode);
 
        }
    }
}
