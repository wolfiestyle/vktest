use crate::types::*;
use base64::engine::{Engine, general_purpose::STANDARD as BASE64};
use std::borrow::Cow;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Uri<'a> {
    Data(&'a str, Encoding, &'a str),
    Relative(Cow<'a, str>),
    Other(&'a str),
}

impl<'a> Uri<'a> {
    pub fn parse(uri: &'a str) -> ImportResult<Self> {
        if let Some((scheme, rest)) = uri.split_once(':') {
            match scheme.to_ascii_lowercase().as_str() {
                "data" => rest.split_once(',').ok_or(ImportError::DataParsingFailed).map(|(meta, data)| {
                    let (mtype, enc, data) = if let Some(mtype) = meta.strip_suffix(";base64") {
                        (mtype, Encoding::Base64, data)
                    } else {
                        (meta, Encoding::UrlEnc, data)
                    };
                    Uri::Data(mtype, enc, data)
                }),
                _ => Ok(Uri::Other(uri)),
            }
        } else {
            urlencoding::decode(uri).map(Uri::Relative).map_err(ImportError::Utf8)
        }
    }

    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::Data(mtype, _, _) if !mtype.is_empty() => Some(mtype),
            _ => None,
        }
    }

    pub fn read_contents(&self, base_path: Option<&Path>) -> ImportResult<Vec<u8>> {
        match self {
            Self::Data(_, Encoding::Base64, data) => BASE64.decode(data).map_err(ImportError::Base64),
            Self::Data(_, Encoding::UrlEnc, data) => Ok(urlencoding::decode_binary(data.as_bytes()).into_owned()),
            Self::Relative(path) if base_path.is_some() => {
                let path = base_path.unwrap().join(&**path);
                std::fs::read(&path).map_err(|err| ImportError::Io(err, path))
            }
            Self::Relative(path) => Err(ImportError::ExternalFile(path.to_string().into())),
            Self::Other(uri) => Err(ImportError::UnsupportedUri(uri.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Encoding {
    Base64,
    UrlEnc,
}
