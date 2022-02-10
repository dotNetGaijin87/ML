namespace ML.Services
{
    /// <summary>
    /// Contains prediction of SrCnn algorithm
    /// </summary>
    public class SrCnnOutput
    {
        public double IsAnomaly { get; set; }
        public double RawScore { get; set; }
        public double Mag { get; set; }
    }
}
