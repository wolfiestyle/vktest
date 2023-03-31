use std::path::PathBuf;

pub type ImportResult<T> = Result<T, ImportError>;

#[derive(Debug)]
pub enum ImportError {
    Gltf(gltf::Error),
    Utf8(std::string::FromUtf8Error),
    Base64(base64::DecodeError),
    Io(std::io::Error, std::path::PathBuf),
    Image(image::ImageError),
    DataParsingFailed,
    ExternalFile(PathBuf),
    UnsupportedUri(String),
    MissingBlob,
    BufferSize { id: usize, actual: usize, expected: usize },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gltf(err) => write!(f, "Failed to parse glTF: {err}"),
            Self::Utf8(err) => write!(f, "{err}"),
            Self::Base64(err) => write!(f, "{err}"),
            Self::Io(err, path) => write!(f, "{err}: {path:?}"),
            Self::Image(err) => write!(f, "{err}"),
            Self::ExternalFile(path) => writeln!(f, "External file unavailable: {path:?}"),
            Self::DataParsingFailed => write!(f, "Failed to parse data Uri"),
            Self::UnsupportedUri(uri) => write!(f, "Unsupported Uri: {uri}"),
            Self::MissingBlob => write!(f, "Missing data blob"),
            Self::BufferSize { id, actual, expected } => write!(f, "Incorrect buffer {id} size (actual: {actual}, expected: {expected})"),
        }
    }
}

impl std::error::Error for ImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gltf(err) => Some(err),
            Self::Utf8(err) => Some(err),
            Self::Base64(err) => Some(err),
            Self::Io(err, _) => Some(err),
            Self::Image(err) => Some(err),
            _ => None,
        }
    }
}

impl From<gltf::Error> for ImportError {
    fn from(err: gltf::Error) -> Self {
        Self::Gltf(err)
    }
}
