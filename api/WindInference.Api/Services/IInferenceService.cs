using WindInference.Api.Models;

namespace WindInference.Api.Services;

public interface IInferenceService
{
    ModelMetadata Metadata { get; }

    PredictionResponse Predict(PredictionRequest request);
}
