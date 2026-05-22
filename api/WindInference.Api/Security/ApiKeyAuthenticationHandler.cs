using System.Security.Claims;
using System.Text.Encodings.Web;
using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Options;

namespace WindInference.Api.Security;

public sealed class ApiKeyAuthenticationHandler : AuthenticationHandler<AuthenticationSchemeOptions>
{
    private readonly IConfiguration _configuration;

    public ApiKeyAuthenticationHandler(
        IOptionsMonitor<AuthenticationSchemeOptions> options,
        ILoggerFactory logger,
        UrlEncoder encoder,
        IConfiguration configuration)
        : base(options, logger, encoder)
    {
        _configuration = configuration;
    }

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        var configuredApiKey = Environment.GetEnvironmentVariable("API_KEY")
            ?? _configuration["Authentication:ApiKey"];

        if (string.IsNullOrWhiteSpace(configuredApiKey))
        {
            Logger.LogError("API key authentication is enabled, but no API key is configured.");
            return Task.FromResult(AuthenticateResult.Fail("API key is not configured."));
        }

        if (!Request.Headers.TryGetValue(ApiKeyAuthenticationDefaults.HeaderName, out var providedApiKey))
        {
            return Task.FromResult(AuthenticateResult.Fail("Missing API key."));
        }

        if (!string.Equals(providedApiKey.ToString(), configuredApiKey, StringComparison.Ordinal))
        {
            return Task.FromResult(AuthenticateResult.Fail("Invalid API key."));
        }

        var claims = new[]
        {
            new Claim(ClaimTypes.NameIdentifier, "api-key-client"),
            new Claim(ClaimTypes.Name, "API Key Client")
        };
        var identity = new ClaimsIdentity(claims, ApiKeyAuthenticationDefaults.SchemeName);
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, ApiKeyAuthenticationDefaults.SchemeName);
        return Task.FromResult(AuthenticateResult.Success(ticket));
    }
}
