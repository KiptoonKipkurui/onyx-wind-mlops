using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using WindInference.Api.Services;
using Xunit;

namespace WindInference.Api.Tests.Services;

public sealed class FileSystemModelProviderTests : IDisposable
{
    private readonly string? _originalModelBundlePath;
    private readonly string _tempRoot;

    public FileSystemModelProviderTests()
    {
        _originalModelBundlePath = Environment.GetEnvironmentVariable("MODEL_BUNDLE_PATH");
        Environment.SetEnvironmentVariable("MODEL_BUNDLE_PATH", null);
        _tempRoot = Path.Combine(Path.GetTempPath(), $"wind-api-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempRoot);
    }

    [Fact]
    public void GetModelBundle_WithExplicitBundlePath_ReturnsModelAndMetadataPaths()
    {
        var bundlePath = CreateBundle("v1");
        var provider = CreateProvider(bundlePath);

        var bundle = provider.GetModelBundle();

        Assert.Equal(Path.GetFullPath(bundlePath), bundle.BundlePath);
        Assert.Equal(Path.Combine(bundlePath, "model.onnx"), bundle.ModelPath);
        Assert.Equal(Path.Combine(bundlePath, "metadata.json"), bundle.MetadataPath);
    }

    [Fact]
    public void GetModelBundle_WithLatestPath_ReturnsNewestVersionDirectoryByName()
    {
        var modelRoot = Path.Combine(_tempRoot, "penmanshiel-event-type-onnx");
        CreateBundle(Path.Combine(modelRoot, "run-20260101T000000Z"));
        var latest = CreateBundle(Path.Combine(modelRoot, "run-20260201T000000Z"));
        var provider = CreateProvider(Path.Combine(modelRoot, "latest"));

        var bundle = provider.GetModelBundle();

        Assert.Equal(Path.GetFullPath(latest), bundle.BundlePath);
    }

    [Fact]
    public void GetModelBundle_WhenFilesAreMissing_ThrowsFileNotFoundException()
    {
        var bundlePath = Path.Combine(_tempRoot, "missing-files");
        Directory.CreateDirectory(bundlePath);
        var provider = CreateProvider(bundlePath);

        Assert.Throws<FileNotFoundException>(() => provider.GetModelBundle());
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("MODEL_BUNDLE_PATH", _originalModelBundlePath);
        if (Directory.Exists(_tempRoot))
        {
            Directory.Delete(_tempRoot, recursive: true);
        }
    }

    private FileSystemModelProvider CreateProvider(string bundlePath)
    {
        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?> { ["ModelBundlePath"] = bundlePath })
            .Build();

        return new FileSystemModelProvider(configuration, NullLogger<FileSystemModelProvider>.Instance);
    }

    private string CreateBundle(string nameOrPath)
    {
        var bundlePath = Path.IsPathRooted(nameOrPath)
            ? nameOrPath
            : Path.Combine(_tempRoot, nameOrPath);
        Directory.CreateDirectory(bundlePath);
        File.WriteAllText(Path.Combine(bundlePath, "model.onnx"), "fake model");
        File.WriteAllText(Path.Combine(bundlePath, "metadata.json"), "{}");
        return bundlePath;
    }
}
