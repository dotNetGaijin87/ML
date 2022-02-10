using Microsoft.ML;
using System.Collections.Generic;

namespace ML.Helpers
{
    public interface IMLDataLoader
    {
        IDataView LoadDataFromFile<T>(string modelTypeName,string FileName) where T : class;
        IDataView LoadIDataView<T>(IEnumerable<T> data) where T : class;
        ITransformer LoadModel(string modelTypeName, string modelName);
        bool SaveModel(ITransformer mlModel, DataViewSchema modelInputSchema,string modelTypeName, string modelName);
    }
}