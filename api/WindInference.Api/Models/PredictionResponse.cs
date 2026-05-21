namespace WindInference.Api.Models;

/// <summary>
/// Structured model prediction response.
/// </summary>
/// <param name="TurbineId">Optional turbine identifier supplied by the caller.</param>
/// <param name="PredictedClass">Numeric class id emitted by the ONNX model.</param>
/// <param name="PredictedClassName">Human-readable class name resolved from model metadata.</param>
/// <param name="RiskProbability">Binary positive-class probability when serving a binary model; null for multiclass models.</param>
/// <param name="ClassProbabilities">Per-class probability distribution for the loaded model.</param>
/// <param name="Features">Feature values used in the request, with non-finite values represented as null for JSON compatibility.</param>
/// <param name="ModelName">Name of the loaded model.</param>
/// <param name="RunId">Training or export run id associated with the model bundle.</param>
public sealed record PredictionResponse(
    string? TurbineId,
    int PredictedClass,
    string PredictedClassName,
    float? RiskProbability,
    IReadOnlyDictionary<string, float?> ClassProbabilities,
    IReadOnlyDictionary<string, float?> Features,
    string ModelName,
    string RunId
);
