using Microsoft.AspNetCore.Mvc;
using WindInference.Api.Controllers;
using Xunit;

namespace WindInference.Api.Tests.Controllers;

public sealed class HealthControllerTests
{
    [Fact]
    public void Get_ReturnsOk()
    {
        var controller = new HealthController();

        var result = controller.Get();

        Assert.IsType<OkObjectResult>(result);
    }
}
