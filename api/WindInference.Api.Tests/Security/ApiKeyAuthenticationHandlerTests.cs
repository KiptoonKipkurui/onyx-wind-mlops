using System.Security.Claims;
using System.Text.Encodings.Web;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using WindInference.Api.Security;
using Xunit;

namespace WindInference.Api.Tests.Security;

public sealed class ApiKeyAuthenticationHandlerTests
{
    [Fact]
    public async Task Authenticate_WhenApiKeyIsMissing_Fails()
    {
        var result = await AuthenticateAsync(
            new Dictionary<string, string?> { ["Authentication:ApiKey"] = "secret" },
            providedApiKey: null);

        Assert.False(result.Succeeded);
    }

    [Fact]
    public async Task Authenticate_WhenApiKeyIsInvalid_Fails()
    {
        var result = await AuthenticateAsync(
            new Dictionary<string, string?> { ["Authentication:ApiKey"] = "secret" },
            providedApiKey: "wrong");

        Assert.False(result.Succeeded);
    }

    [Fact]
    public async Task Authenticate_WhenApiKeyMatches_Succeeds()
    {
        var result = await AuthenticateAsync(
            new Dictionary<string, string?> { ["Authentication:ApiKey"] = "secret" },
            providedApiKey: "secret");

        Assert.True(result.Succeeded);
        Assert.Equal("api-key-client", result.Principal?.FindFirstValue(ClaimTypes.NameIdentifier));
    }

    private static async Task<AuthenticateResult> AuthenticateAsync(
        Dictionary<string, string?> configurationValues,
        string? providedApiKey)
    {
        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(configurationValues)
            .Build();
        var handler = new ApiKeyAuthenticationHandler(
            new TestOptionsMonitor(),
            NullLoggerFactory.Instance,
            UrlEncoder.Default,
            configuration);
        var context = new DefaultHttpContext();
        if (providedApiKey is not null)
        {
            context.Request.Headers[ApiKeyAuthenticationDefaults.HeaderName] = providedApiKey;
        }

        var scheme = new AuthenticationScheme(
            ApiKeyAuthenticationDefaults.SchemeName,
            ApiKeyAuthenticationDefaults.SchemeName,
            typeof(ApiKeyAuthenticationHandler));
        await handler.InitializeAsync(scheme, context);
        return await handler.AuthenticateAsync();
    }

    private sealed class TestOptionsMonitor : IOptionsMonitor<AuthenticationSchemeOptions>
    {
        public AuthenticationSchemeOptions CurrentValue { get; } = new();

        public AuthenticationSchemeOptions Get(string? name)
        {
            return CurrentValue;
        }

        public IDisposable? OnChange(Action<AuthenticationSchemeOptions, string?> listener)
        {
            return null;
        }
    }
}
