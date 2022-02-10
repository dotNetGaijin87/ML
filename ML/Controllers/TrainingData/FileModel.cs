using Microsoft.AspNetCore.Http;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace ML.Controllers.TrainingData
{
    public class FileModel
    {
        [Required]
        public string ModelTypeName { get; set; }
        [Required]
        public IEnumerable<IFormFile> FormFiles { get; set; }
    }
}
