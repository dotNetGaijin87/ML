using ML.Models;
using System.Collections.Generic;

namespace ML.Services
{
    /// <summary>
    /// Interface for <class cref="SrCnnService"/> 
    /// </summary>
    public interface ISrCnnService
    {
        IEnumerable<SrCnnOutput> Predict(SrCnnInput input);
    }
}