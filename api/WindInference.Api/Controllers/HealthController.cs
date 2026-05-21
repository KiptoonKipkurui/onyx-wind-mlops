using Microsoft.AspNetCore.Mvc;

namespace WindInference.Api.Controllers;

[ApiController]
public sealed class HealthController : ControllerBase
{
    /// <summary>
    /// Checks whether the inference API is running.
    /// </summary>
    /// <returns>A simple health status payload.</returns>
    [HttpGet("/health")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public IActionResult Get()
    {
        return Ok(new { status = "ok" });
    }
}
