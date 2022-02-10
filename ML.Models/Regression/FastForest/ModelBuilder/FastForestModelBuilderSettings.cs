using System.ComponentModel.DataAnnotations;

namespace ML.Models.Regression.FastForest
{
    /// <summary>
    /// Settings used for building models with <class cref="FastForestModelBuilder"/> 
    /// </summary>
    public class FastForestModelBuilderSettings
    {
        [Required]
        [StringLength(50, MinimumLength = 3)]
        public string ModelName { get; set; }

        [Required]
        [Range(2, 10000)]
        public int LeavesCount { get; set; }

        [Required]
        [Range(2, 10000)]
        public int MinimumExampleCountPerLeaf { get; set; }

        [Required]
        [Range(1, 100000)]
        public int TreesCount { get; set; }

        [Required]
        public string LabelColumnName { get; set; }

        [Required]
        public string[] FeatureColumnNames { get; set; }

        [Required]
        [Range(1, 20)]
        public int CrossValidationFoldsCount { get; set; }

        [Required]
        [StringLength(50, MinimumLength = 3)]
        public string TrainigDataName { get; set; }


        public bool HasFeatureContributionMetrics { get; set; } = false;

    }
}
