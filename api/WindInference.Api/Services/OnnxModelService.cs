using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using WindInference.Api.Models;

namespace WindInference.Api.Services;

public sealed class OnnxModelService : IInferenceService, IDisposable
{
    private readonly ILogger<OnnxModelService> _logger;
    private readonly InferenceSession _session;

    public OnnxModelService(IModelProvider modelProvider, ILogger<OnnxModelService> logger)
    {
        _logger = logger;
        var bundle = modelProvider.GetModelBundle();

        _logger.LogInformation("Loading ONNX model bundle from {BundlePath}", bundle.BundlePath);
        Metadata = JsonSerializer.Deserialize<Models.ModelMetadata>(File.ReadAllText(bundle.MetadataPath))
            ?? throw new InvalidOperationException("Could not parse model metadata.");
        _session = new InferenceSession(bundle.ModelPath);
        _logger.LogInformation(
            "Loaded model {ModelName} run {RunId} task {Task} with {FeatureCount} features and {ClassCount} classes",
            Metadata.ModelName,
            Metadata.RunId,
            Metadata.Task,
            Metadata.FeatureNames.Length,
            Metadata.ClassNames?.Length ?? 0);
    }

    public Models.ModelMetadata Metadata { get; }

    public PredictionResponse Predict(PredictionRequest request)
    {
        var featureVector = BuildFeatureVector(request);
        var suppliedFeatureCount = CountSuppliedFeatures(request);
        var featureCoverage = Metadata.FeatureNames.Length == 0
            ? 0
            : (double)suppliedFeatureCount / Metadata.FeatureNames.Length;
        _logger.LogInformation(
            "Running prediction for turbine {TurbineId}. SuppliedFeatures={SuppliedFeatureCount}/{FeatureCount}, Coverage={FeatureCoverage:P1}",
            request.TurbineId,
            suppliedFeatureCount,
            Metadata.FeatureNames.Length,
            featureCoverage);
        var tensor = new DenseTensor<float>(featureVector, new[] { 1, Metadata.FeatureNames.Length });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(Metadata.InputName, tensor)
        };
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
        try
        {
            results = _session.Run(inputs);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ONNX inference failed for turbine {TurbineId}", request.TurbineId);
            throw;
        }

        using (results)
        {
            var predictedClass = ReadPredictedClass(results);
            var probabilities = ReadProbabilities(results);
            var classNames = ResolveClassNames(probabilities.Length);
            var riskProbability = ResolveRiskProbability(probabilities);
            var predictedClassName = predictedClass >= 0 && predictedClass < classNames.Length
                ? classNames[predictedClass]
                : predictedClass.ToString();
            var probabilityMap = classNames
                .Select((name, index) => new KeyValuePair<string, float?>(
                    name,
                    index < probabilities.Length ? ToJsonNumber(probabilities[index]) : null))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
            var featureMap = Metadata.FeatureNames
                .Select((name, index) => new KeyValuePair<string, float?>(name, ToJsonNumber(featureVector[index])))
                .ToDictionary(pair => pair.Key, pair => pair.Value);

            _logger.LogInformation(
                "Prediction completed for turbine {TurbineId}. PredictedClass={PredictedClass}, PredictedClassName={PredictedClassName}",
                request.TurbineId,
                predictedClass,
                predictedClassName);

            return new PredictionResponse(
                request.TurbineId,
                predictedClass,
                predictedClassName,
                ToJsonNumber(riskProbability),
                probabilityMap,
                featureMap,
                Metadata.ModelName,
                Metadata.RunId);
        }
    }

    public void Dispose() => _session.Dispose();

    private float[] BuildFeatureVector(PredictionRequest request)
    {
        if (request.Features is { Length: > 0 })
        {
            if (request.Features.Length != Metadata.FeatureNames.Length)
            {
                _logger.LogWarning(
                    "Ordered feature vector length mismatch. Expected={ExpectedFeatureCount}, Actual={ActualFeatureCount}",
                    Metadata.FeatureNames.Length,
                    request.Features.Length);
                throw new ArgumentException(
                    $"Expected {Metadata.FeatureNames.Length} ordered features, got {request.Features.Length}.");
            }

            return request.Features;
        }

        if (request.NamedFeatures is { Count: > 0 })
        {
            return Metadata.FeatureNames
                .Select(name => request.NamedFeatures.TryGetValue(name, out var value) ? value : float.NaN)
                .ToArray();
        }

        throw new ArgumentException("Provide either 'features' or 'namedFeatures'.");
    }

    private int CountSuppliedFeatures(PredictionRequest request)
    {
        if (request.Features is { Length: > 0 })
        {
            return request.Features.Count(float.IsFinite);
        }

        if (request.NamedFeatures is { Count: > 0 })
        {
            return Metadata.FeatureNames.Count(name =>
                request.NamedFeatures.TryGetValue(name, out var value) && float.IsFinite(value));
        }

        return 0;
    }

    private int ReadPredictedClass(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        var label = results.FirstOrDefault(result => result.Name == Metadata.LabelOutput) ?? results.First();
        if (label.Value is Tensor<long> longTensor)
        {
            return Convert.ToInt32(longTensor.First());
        }

        if (label.Value is Tensor<int> intTensor)
        {
            return intTensor.First();
        }

        throw new InvalidOperationException($"Unsupported label output type '{label.Value.GetType().Name}'.");
    }

    private float[] ReadProbabilities(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        var probability = results.FirstOrDefault(result => result.Name == Metadata.ProbabilityOutput)
            ?? results.FirstOrDefault(result => result.Value is Tensor<float>);

        if (probability?.Value is not Tensor<float> tensor)
        {
            return Array.Empty<float>();
        }

        return tensor.ToArray();
    }

    private float ResolveRiskProbability(float[] probabilities)
    {
        if (!Metadata.Task.Equals("binary_classification", StringComparison.OrdinalIgnoreCase))
        {
            return float.NaN;
        }

        return probabilities.Length switch
        {
            >= 2 when Metadata.PositiveClass < probabilities.Length => probabilities[Metadata.PositiveClass],
            1 => probabilities[0],
            _ => float.NaN
        };
    }

    private string[] ResolveClassNames(int probabilityCount)
    {
        if (Metadata.ClassNames is { Length: > 0 })
        {
            return Metadata.ClassNames;
        }

        var count = Math.Max(probabilityCount, 1);
        return Enumerable.Range(0, count).Select(index => $"class_{index}").ToArray();
    }

    private static float? ToJsonNumber(float value)
    {
        return float.IsFinite(value) ? value : null;
    }

}
