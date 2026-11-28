use base64::{DecodeError, Engine as _, engine::general_purpose::STANDARD};
use gltf_import::{GltfData, ImageData, ImportError};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use std::{io::Cursor, path::Path};

#[test]
fn decodes_embedded_buffer_data_uris() {
    let model = br#"{
        "asset": {"version": "2.0"},
        "buffers": [
            {"byteLength": 4, "uri": "data:application/octet-stream;base64,AAEs/w=="},
            {"byteLength": 4, "uri": "data:application/octet-stream,%00%01%2C%FF"}
        ]
    }"#;
    let data = GltfData::from_memory(model).unwrap();

    assert_eq!(data.buffers.len(), 2);
    for buffer in data.buffers {
        assert_eq!(buffer.data, [0, 1, b',', 255]);
    }
}

#[test]
fn reports_decode_errors_for_malformed_base64_data_uris() {
    for (encoded, expected) in [
        ("AAEs/w", DecodeError::InvalidPadding),
        ("AAEs!w==", DecodeError::InvalidByte(4, b'!')),
        (
            "AAEs/x==",
            DecodeError::InvalidLastSymbol {
                offset: 5,
                symbol: b'x',
                symbol_value: 49,
            },
        ),
    ] {
        let model = format!(
            r#"{{
                "asset": {{"version": "2.0"}},
                "buffers": [{{"byteLength": 4, "uri": "data:application/octet-stream;base64,{encoded}"}}]
            }}"#
        );
        let error = GltfData::from_memory(model.as_bytes()).unwrap_err();

        let ImportError::Base64(actual) = error else {
            panic!("expected a base64 decode error for {encoded}, got {error:?}");
        };
        assert_eq!(actual, expected, "{encoded}");
    }
}

#[test]
fn resolves_relative_buffer_uri_against_the_model_directory() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/external_buffer/model.gltf");
    let data = GltfData::from_file(path).unwrap();
    assert_eq!(data.buffers[0].data, include_bytes!("fixtures/external_buffer/payload bytes.txt"));

    let model = include_bytes!("fixtures/external_buffer/model.gltf");
    let error = GltfData::from_memory(model).unwrap_err();
    assert!(matches!(error, ImportError::ExternalFile(path) if path == Path::new("payload bytes.txt")));
}

#[test]
fn reports_a_buffer_shorter_than_its_declared_length() {
    let model = br#"{
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 4, "uri": "data:application/octet-stream;base64,AAEs"}]
    }"#;

    assert!(matches!(
        GltfData::from_memory(model),
        Err(ImportError::BufferSize {
            id: 0,
            actual: 3,
            expected: 4
        })
    ));
}

#[test]
fn decodes_embedded_images_and_marks_color_textures_as_srgb() {
    let pixel = Rgba([32, 64, 96, 128]);
    let mut png = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, pixel))
        .write_to(&mut png, ImageFormat::Png)
        .unwrap();
    let png = png.into_inner();
    let png_len = png.len();
    let png_uri = STANDARD.encode(&png);

    let mut buffer = vec![0; 4];
    buffer.extend_from_slice(&png);
    let buffer_len = buffer.len();
    let buffer_uri = STANDARD.encode(&buffer);
    let model = format!(
        r#"{{
            "asset": {{"version": "2.0"}},
            "buffers": [{{"byteLength": {buffer_len}, "uri": "data:application/octet-stream;base64,{buffer_uri}"}}],
            "bufferViews": [{{"buffer": 0, "byteOffset": 4, "byteLength": {png_len}}}],
            "images": [
                {{"uri": "data:image/png;base64,{png_uri}"}},
                {{"bufferView": 0, "mimeType": "image/png"}},
                {{"uri": "data:image/png;base64,{png_uri}"}}
            ],
            "textures": [{{"source": 2}}, {{"source": 0}}, {{"source": 1}}],
            "materials": [{{
                "pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0}}, "metallicRoughnessTexture": {{"index": 1}}}},
                "emissiveTexture": {{"index": 2}},
                "normalTexture": {{"index": 1}},
                "occlusionTexture": {{"index": 1}}
            }}]
        }}"#
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();

    assert_eq!(data.images.iter().map(|image| image.srgb).collect::<Vec<_>>(), [false, true, true]);
    for image in &data.images {
        assert_eq!(image.mime_type.as_deref(), Some("image/png"));
        let ImageData::Decoded(decoded) = &image.data else {
            panic!("expected a decoded PNG, got {:?}", image.data);
        };
        let decoded = decoded.to_rgba8();
        assert_eq!(decoded.dimensions(), (1, 1));
        assert_eq!(*decoded.get_pixel(0, 0), pixel);
    }
}

#[test]
fn preserves_a_decode_error_on_the_failed_image() {
    let model = br#"{
        "asset": {"version": "2.0"},
        "images": [{"uri": "data:image/png;base64,AAEs/w=="}]
    }"#;
    let data = GltfData::from_memory(model).unwrap();

    let ImageData::Missing(error) = &data.images[0].data else {
        panic!("expected an image decode error, got {:?}", data.images[0].data);
    };
    assert!(matches!(error.as_ref(), ImportError::Image(_)));
}
