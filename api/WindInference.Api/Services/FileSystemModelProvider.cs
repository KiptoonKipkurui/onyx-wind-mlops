namespace WindInference.Api.Services;

public sealed class FileSystemModelProvider : IModelProvider
{
    private readonly string _bundlePath;
    private readonly ILogger<FileSystemModelProvider> _logger;

    public FileSystemModelProvider(IConfiguration configuration, ILogger<FileSystemModelProvider> logger)
    {
        _logger = logger;
        _bundlePath = Environment.GetEnvironmentVariable("MODEL_BUNDLE_PATH")
            ?? configuration["ModelBundlePath"]
            ?? Path.GetFullPath("../../model_repository/penmanshiel-event-type-onnx/latest");
    }

    public ModelBundle GetModelBundle()
    {
        var resolvedBundle = ResolveBundlePath(_bundlePath);
        var modelPath = Path.Combine(resolvedBundle, "model.onnx");
        var metadataPath = Path.Combine(resolvedBundle, "metadata.json");

        _logger.LogInformation("Resolved filesystem model bundle to {BundlePath}", resolvedBundle);
        if (!File.Exists(modelPath) || !File.Exists(metadataPath))
        {
            _logger.LogError(
                "Model bundle is missing required files. ModelPath={ModelPath}, MetadataPath={MetadataPath}",
                modelPath,
                metadataPath);
            throw new FileNotFoundException(
                $"Expected model.onnx and metadata.json in model bundle '{resolvedBundle}'.");
        }

        return new ModelBundle(resolvedBundle, modelPath, metadataPath);
    }

    private static string ResolveBundlePath(string bundlePath)
    {
        if (!bundlePath.EndsWith("latest", StringComparison.OrdinalIgnoreCase))
        {
            return Path.GetFullPath(bundlePath);
        }

        var parent = Path.GetDirectoryName(Path.GetFullPath(bundlePath))
            ?? throw new DirectoryNotFoundException(bundlePath);
        var latest = Directory.GetDirectories(parent)
            .Where(path => !path.EndsWith("latest", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(path => path)
            .FirstOrDefault();

        return latest ?? throw new DirectoryNotFoundException($"No model versions found under {parent}.");
    }
}
