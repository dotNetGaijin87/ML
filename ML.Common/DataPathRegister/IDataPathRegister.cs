namespace ML.Common
{
    /// <summary>
    /// Interface for <class cref="DataPathRegister"/> 
    /// </summary>
    public interface IDataPathRegister
    {
        string GetModelFilePath(string modelTypeName, string modelName);
        string GetTrainingDataFilePath(string modelTypeName, string fileName);
    }
}