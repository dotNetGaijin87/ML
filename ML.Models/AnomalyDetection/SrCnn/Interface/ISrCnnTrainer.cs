using Microsoft.ML.TimeSeries;
using System.Collections.Generic;

namespace ML.Models
{
    /// <summary>
    /// Interface of SrCnnTrainer
    /// </summary>
    public interface ISrCnnTrainer
    {
        IEnumerable<SrCnnTrainerOutput> Run(SrCnnTrainerInputCollection input, SrCnnEntireAnomalyDetectorOptions options);
    }
}