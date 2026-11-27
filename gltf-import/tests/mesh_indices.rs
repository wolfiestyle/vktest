use base64::{Engine as _, engine::general_purpose::STANDARD};
use gltf_import::{GltfData, ImportError};

fn triangle(indices: &[u8], count: usize, attributes: &str) -> String {
    let positions = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let normals = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let uvs = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
    let bytes: Vec<_> = positions
        .into_iter()
        .chain(normals)
        .chain(uvs)
        .flat_map(f32::to_le_bytes)
        .chain(indices.iter().copied())
        .collect();
    let encoded = STANDARD.encode(&bytes);
    let length = bytes.len();
    let index_bytes = indices.len();
    format!(
        r#"{{
            "asset": {{"version": "2.0"}},
            "buffers": [{{"uri": "data:application/octet-stream;base64,{encoded}", "byteLength": {length}}}],
            "bufferViews": [
                {{"buffer": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 72, "byteLength": 24}},
                {{"buffer": 0, "byteOffset": 96, "byteLength": {index_bytes}}}
            ],
            "accessors": [
                {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}},
                {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}},
                {{"bufferView": 3, "componentType": 5121, "count": {count}, "type": "SCALAR"}}
            ],
            "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0{attributes}}}, "indices": 3}}]}}]
        }}"#
    )
}

#[test]
fn reports_malformed_triangle_indices() {
    for (indices, count, attributes, expected) in [
        (&[0, 1, 2, 0][..], 4, "", "positive multiple of 3"),
        (&[0, 1, 99][..], 3, "", "index 99 at element 2"),
        (&[0, 1, 99][..], 3, r#", "NORMAL": 1, "TEXCOORD_0": 2"#, "index 99 at element 2"),
        (&[0, 1][..], 3, "", "unreadable index accessor"),
    ] {
        let model = triangle(indices, count, attributes);
        let error = GltfData::from_memory(model.as_bytes()).unwrap_err();
        assert!(
            matches!(&error, ImportError::InvalidPrimitive { mesh: 0, primitive: 0, reason } if reason.contains(expected)),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn imports_valid_indexed_triangle() {
    let indices = [0, 2, 1];
    let model = triangle(&indices, indices.len(), r#", "NORMAL": 1, "TEXCOORD_0": 2"#);
    let data = GltfData::from_memory(model.as_bytes()).unwrap();
    assert_eq!(data.meshes[0].indices, indices.map(u32::from));
    assert_eq!(data.meshes[0].vertices.len(), 3);
}
