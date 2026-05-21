using System.Reflection;
using Microsoft.OpenApi.Models;
using WindInference.Api.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

builder.Services.AddControllers();
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

app.MapControllers();

app.Run();
