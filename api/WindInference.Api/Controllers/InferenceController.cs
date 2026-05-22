using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using WindInference.Api.Models;
using WindInference.Api.Services;

namespace WindInference.Api.Controllers;

[ApiController]
[Authorize]
[Route("api/inference")]
public sealed class InferenceController : ControllerBase
{
    private readonly ILogger<InferenceController> _logger;
    private readonly IInferenceService _model;

    public InferenceController(IInferenceService model, ILogger<InferenceController> logger)
    {
        _model = model;
        _logger = logger;
    }

    /// <summary>
    /// Gets metadata for the currently loaded ONNX model.
    /// </summary>
    /// <remarks>
    /// The metadata includes the expected feature order, input/output tensor names, model task,
    /// target name, and class labels for multiclass models.
    /// </remarks>
    /// <returns>The model metadata stored next to the ONNX artifact.</returns>
    [HttpGet("metadata")]
    [ProducesResponseType(typeof(ModelMetadata), StatusCodes.Status200OK)]
    public IActionResult GetMetadata()
    {
        _logger.LogInformation("Model metadata requested for model {ModelName} run {RunId}", _model.Metadata.ModelName, _model.Metadata.RunId);
        return Ok(_model.Metadata);
    }

    /// <summary>
    /// Runs inference against the loaded ONNX model.
    /// </summary>
    /// <remarks>
    /// Send either an ordered <c>features</c> array matching <c>metadata.featureNames</c>,
    /// or a <c>namedFeatures</c> dictionary keyed by exact feature name. Missing named features
    /// are passed as NaN and handled by the model pipeline imputer.
    /// </remarks>
    /// <param name="request">The turbine feature payload.</param>
    /// <returns>A prediction with class probabilities and echoed feature values.</returns>
    [HttpPost("predict")]
    [ProducesResponseType(typeof(PredictionResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public IActionResult Predict([FromBody] PredictionRequest request)
    {
        try
        {
            _logger.LogInformation(
                "Prediction request received. TurbineId={TurbineId}, HasOrderedFeatures={HasOrderedFeatures}, NamedFeatureCount={NamedFeatureCount}",
                request.TurbineId,
                request.Features is { Length: > 0 },
                request.NamedFeatures?.Count ?? 0);
            return Ok(_model.Predict(request));
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "Prediction request rejected for turbine {TurbineId}", request.TurbineId);
            return BadRequest(new { error = ex.Message });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Prediction request failed for turbine {TurbineId}", request.TurbineId);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "Prediction failed." });
        }
    }
}
