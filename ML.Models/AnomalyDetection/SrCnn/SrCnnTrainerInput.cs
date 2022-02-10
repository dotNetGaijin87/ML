using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace ML.Models
{
    /// <summary>
    /// Time series data to be fed into SrCnnEntireAnomalyDetector.
    /// Minimum data count is 12.
    /// </summary>
    public class SrCnnTrainerInputCollection
    {
        [Required]
        [Range(12, int.MaxValue)]
        public IEnumerable<SrCnnTrainerInput> Values { get; set; }
    }

    /// <summary>
    /// Helper class for wrapping the data
    /// </summary>
    public class SrCnnTrainerInput
    {
        public double Value { get; set; }
    }
}
