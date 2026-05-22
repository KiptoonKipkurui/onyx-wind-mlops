using System.Reflection;
using Microsoft.AspNetCore.Authentication;
using Microsoft.OpenApi.Models;
using WindInference.Api.Security;
using WindInference.Api.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

builder.Services.AddControllers();
builder.Services
    .AddAuthentication(ApiKeyAuthenticationDefaults.SchemeName)
    .AddScheme<AuthenticationSchemeOptions, ApiKeyAuthenticationHandler>(
        ApiKeyAuthenticationDefaults.SchemeName,
        options => { });
builder.Services.AddAuthorization();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc(
        "v1",
        new OpenApiInfo
        {
            Title = "Wind Inference API",
            Version = "v1",
            Description = "ONNX Runtime API for multiclass wind turbine event inference from SCADA features."
        });
    options.AddSecurityDefinition(
        ApiKeyAuthenticationDefaults.SchemeName,
        new OpenApiSecurityScheme
        {
            Description = $"API key required in the {ApiKeyAuthenticationDefaults.HeaderName} header.",
            In = ParameterLocation.Header,
            Name = ApiKeyAuthenticationDefaults.HeaderName,
            Type = SecuritySchemeType.ApiKey,
            Scheme = ApiKeyAuthenticationDefaults.SchemeName
        });
    options.AddSecurityRequirement(
        new OpenApiSecurityRequirement
        {
            [
                new OpenApiSecurityScheme
                {
                    Reference = new OpenApiReference
                    {
                        Type = ReferenceType.SecurityScheme,
                        Id = ApiKeyAuthenticationDefaults.SchemeName
                    }
                }
            ] = Array.Empty<string>()
        });

    var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
    var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);
    if (File.Exists(xmlPath))
    {
        options.IncludeXmlComments(xmlPath);
    }
});
builder.Services.AddSingleton<IModelProvider, FileSystemModelProvider>();
builder.Services.AddSingleton<IInferenceService, OnnxModelService>();

var app = builder.Build();
app.Logger.LogInformation("Wind Inference API starting in {EnvironmentName}", app.Environment.EnvironmentName);

app.UseSwagger(options =>
{
    options.RouteTemplate = "openapi/{documentName}.json";
});
app.UseReDoc(options =>
{
    options.RoutePrefix = "docs";
    options.SpecUrl = "/openapi/v1.json";
    options.DocumentTitle = "Wind Inference API Docs";
});

app.UseAuthentication();
app.UseAuthorization();
app.MapControllers();

app.Run();
