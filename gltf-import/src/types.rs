use std::path::PathBuf;
use thiserror::Error;

pub type ImportResult<T> = Result<T, ImportError>;

#[derive(Debug, Error)]
pub enum ImportError {
    #[error("Failed to parse glTF: {0}")]
    Gltf(#[from] gltf::Error),

    #[error("{0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("{0}")]
    Base64(#[from] base64::DecodeError),

    #[error("{0}: {1}")]
    Io(#[source] std::io::Error, std::path::PathBuf),

    #[error("{0}")]
    Image(#[from] image::ImageError),

    #[error("Failed to parse data Uri")]
    DataParsingFailed,

    #[error("External file unavailable: {0}")]
    ExternalFile(PathBuf),

    #[error("Unsupported Uri: {0}")]
    UnsupportedUri(String),

    #[error("Missing data blob")]
    MissingBlob,

    #[error("Incorrect buffer {id} size (actual: {actual}, expected: {expected})")]
    BufferSize { id: usize, actual: usize, expected: usize },
}
