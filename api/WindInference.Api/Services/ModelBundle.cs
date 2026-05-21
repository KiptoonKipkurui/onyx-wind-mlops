namespace WindInference.Api.Services;

public sealed record ModelBundle(
    string BundlePath,
    string ModelPath,
    string MetadataPath
);
