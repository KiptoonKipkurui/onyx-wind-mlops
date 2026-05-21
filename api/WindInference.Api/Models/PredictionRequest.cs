namespace WindInference.Api.Models;

/// <summary>
/// Request payload for model inference.
/// </summary>
/// <param name="Features">Ordered feature vector matching the model metadata feature order.</param>
/// <param name="NamedFeatures">Feature values keyed by exact feature name.</param>
/// <param name="TurbineId">Optional turbine identifier for tracing the prediction.</param>
public sealed record PredictionRequest(
    float[]? Features,
    Dictionary<string, float>? NamedFeatures,
    string? TurbineId = null
);
