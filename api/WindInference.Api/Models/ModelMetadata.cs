using System.Text.Json.Serialization;

namespace WindInference.Api.Models;

/// <summary>
/// Metadata stored next to an exported ONNX model bundle.
/// </summary>
/// <param name="RunId">Training or export run id.</param>
/// <param name="ModelName">Logical model name.</param>
/// <param name="Task">Model task, such as binary_classification or multiclass_classification.</param>
/// <param name="Target">Target column or label predicted by the model.</param>
/// <param name="FeatureNames">Expected feature order for ordered tensor input.</param>
/// <param name="InputName">ONNX input tensor name.</param>
/// <param name="LabelOutput">ONNX label output tensor name.</param>
/// <param name="ProbabilityOutput">ONNX probability output tensor name.</param>
/// <param name="PositiveClass">Positive class index for binary models.</param>
/// <param name="ClassNames">Human-readable class names for multiclass models.</param>
/// <param name="ClassToId">Mapping from class name to model class id.</param>
/// <param name="Metrics">Training or validation metrics logged with the artifact.</param>
public sealed record ModelMetadata(
    [property: JsonPropertyName("run_id")] string RunId,
    [property: JsonPropertyName("model_name")] string ModelName,
    [property: JsonPropertyName("task")] string Task,
    [property: JsonPropertyName("target")] string Target,
    [property: JsonPropertyName("feature_names")] string[] FeatureNames,
    [property: JsonPropertyName("input_name")] string InputName,
    [property: JsonPropertyName("label_output")] string LabelOutput,
    [property: JsonPropertyName("probability_output")] string ProbabilityOutput,
    [property: JsonPropertyName("positive_class")] int PositiveClass,
    [property: JsonPropertyName("class_names")] string[]? ClassNames,
    [property: JsonPropertyName("class_to_id")] Dictionary<string, int>? ClassToId,
    [property: JsonPropertyName("metrics")] Dictionary<string, double>? Metrics
);
