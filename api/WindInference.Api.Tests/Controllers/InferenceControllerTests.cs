using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using WindInference.Api.Controllers;
using WindInference.Api.Models;
using WindInference.Api.Services;
using Xunit;

namespace WindInference.Api.Tests.Controllers;

public sealed class InferenceControllerTests
{
    [Fact]
    public void GetMetadata_ReturnsLoadedModelMetadata()
    {
        var metadata = TestMetadata();
        var controller = new InferenceController(
            new FakeInferenceService(metadata, TestPrediction()),
            NullLogger<InferenceController>.Instance);

        var result = Assert.IsType<OkObjectResult>(controller.GetMetadata());

        Assert.Same(metadata, result.Value);
    }

    [Fact]
    public void Predict_ReturnsPrediction()
    {
        var prediction = TestPrediction();
        var controller = new InferenceController(
            new FakeInferenceService(TestMetadata(), prediction),
            NullLogger<InferenceController>.Instance);

        var result = Assert.IsType<OkObjectResult>(controller.Predict(new PredictionRequest(
            Features: [1.0f, 2.0f],
            NamedFeatures: null,
            TurbineId: "T06")));

        Assert.Same(prediction, result.Value);
    }

    [Fact]
    public void Predict_WhenRequestIsInvalid_ReturnsBadRequest()
    {
        var controller = new InferenceController(
            new FakeInferenceService(TestMetadata(), TestPrediction(), new ArgumentException("bad features")),
            NullLogger<InferenceController>.Instance);

        var result = Assert.IsType<BadRequestObjectResult>(controller.Predict(new PredictionRequest(
            Features: [1.0f],
            NamedFeatures: null,
            TurbineId: "T06")));

        Assert.Equal(400, result.StatusCode);
    }

    [Fact]
    public void Predict_WhenInferenceFails_ReturnsInternalServerError()
    {
        var controller = new InferenceController(
            new FakeInferenceService(TestMetadata(), TestPrediction(), new InvalidOperationException("boom")),
            NullLogger<InferenceController>.Instance);

        var result = Assert.IsType<ObjectResult>(controller.Predict(new PredictionRequest(
            Features: [1.0f],
            NamedFeatures: null,
            TurbineId: "T06")));

        Assert.Equal(500, result.StatusCode);
    }

    private static ModelMetadata TestMetadata()
    {
        return new ModelMetadata(
            RunId: "run-1",
            ModelName: "test-model",
            Task: "multiclass_classification",
            Target: "next_event_type_60m",
            FeatureNames: ["Wind speed (m/s)", "Power (kW)"],
            InputName: "float_input",
            LabelOutput: "label",
            ProbabilityOutput: "probabilities",
            PositiveClass: 1,
            ClassNames: ["no_event", "forced_outage"],
            ClassToId: new Dictionary<string, int> { ["no_event"] = 0, ["forced_outage"] = 1 },
            Metrics: new Dictionary<string, double> { ["balanced_accuracy"] = 0.75 });
    }

    private static PredictionResponse TestPrediction()
    {
        return new PredictionResponse(
            TurbineId: "T06",
            PredictedClass: 1,
            PredictedClassName: "forced_outage",
            RiskProbability: null,
            ClassProbabilities: new Dictionary<string, float?> { ["no_event"] = 0.2f, ["forced_outage"] = 0.8f },
            Features: new Dictionary<string, float?> { ["Wind speed (m/s)"] = 7.5f, ["Power (kW)"] = 530f },
            ModelName: "test-model",
            RunId: "run-1");
    }

    private sealed class FakeInferenceService : IInferenceService
    {
        private readonly PredictionResponse _prediction;
        private readonly Exception? _exception;

        public FakeInferenceService(ModelMetadata metadata, PredictionResponse prediction, Exception? exception = null)
        {
            Metadata = metadata;
            _prediction = prediction;
            _exception = exception;
        }

        public ModelMetadata Metadata { get; }

        public PredictionResponse Predict(PredictionRequest request)
        {
            if (_exception is not null)
            {
                throw _exception;
            }

            return _prediction;
        }
    }
}
