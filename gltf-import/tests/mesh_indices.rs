use base64::{Engine as _, engine::general_purpose::STANDARD};
use gltf_import::{GltfData, ImportError};

fn triangle_model(indices: &[u8], count: usize, primitives: &str) -> String {
    triangle_model_with_uvs(indices, count, primitives, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
}

fn triangle_model_with_uvs(indices: &[u8], count: usize, primitives: &str, uvs: [[f32; 2]; 3]) -> String {
    let positions = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let normals = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let bytes: Vec<_> = positions
        .into_iter()
        .chain(normals)
        .chain(uvs.into_iter().flatten())
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
            "meshes": [{{"primitives": [{primitives}]}}]
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
        let primitive = format!(r#"{{"attributes": {{"POSITION": 0{attributes}}}, "indices": 3}}"#);
        let model = triangle_model(indices, count, &primitive);
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
    let model = triangle_model(
        &indices,
        indices.len(),
        r#"{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3}"#,
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();
    assert_eq!(data.meshes[0].indices, indices.map(u32::from));
    assert_eq!(data.meshes[0].vertices.len(), 3);
}

#[test]
fn generates_indices_normals_and_tangents_for_an_unindexed_triangle() {
    // The indexed primitive in the same fixture exercises normal generation with reversed winding.
    let model = triangle_model(
        &[0, 2, 1],
        3,
        r#"{"attributes": {"POSITION": 0, "TEXCOORD_0": 2}},
           {"attributes": {"POSITION": 0}, "indices": 3}"#,
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();
    let mesh = &data.meshes[0];
    let unindexed = &mesh.primitives[0];
    assert_eq!(mesh.indices[unindexed.index_range()], [0, 1, 2]);
    for vertex in &mesh.vertices[unindexed.vertex_range()] {
        assert_eq!(vertex.normal, [0, 0, i16::MAX, 0]);
        assert_eq!(vertex.tangent, [i16::MAX, 0, 0, -i16::MAX]);
    }

    let reversed = &mesh.primitives[1];
    for vertex in &mesh.vertices[reversed.vertex_range()] {
        assert_eq!(vertex.normal, [0, 0, -i16::MAX, 0]);
    }
}

#[test]
fn generates_the_opposite_tangent_sign_for_mirrored_uvs() {
    let model = triangle_model_with_uvs(
        &[0, 1, 2],
        3,
        r#"{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3}"#,
        [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();

    for vertex in &data.meshes[0].vertices {
        assert_eq!(vertex.tangent, [i16::MAX, 0, 0, i16::MAX]);
    }
}

#[test]
fn supplies_fallback_tangents_only_for_referenced_vertices_with_normals() {
    let model = triangle_model(
        &[0, 0, 1],
        3,
        r#"{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3},
           {"attributes": {"POSITION": 0, "TEXCOORD_0": 2}, "indices": 3}"#,
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();
    let tangents: Vec<_> = data.meshes[0].vertices.iter().map(|vertex| vertex.tangent).collect();
    let fallback = [0, -i16::MAX, 0, i16::MAX];
    assert_eq!(tangents, [fallback, fallback, [0; 4], [0; 4], [0; 4], [0; 4]]);
}

#[test]
fn uses_a_tangent_orthogonal_to_the_normal_for_collapsed_uvs() {
    let model = triangle_model_with_uvs(
        &[0, 1, 2],
        3,
        r#"{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3}"#,
        [[0.0, 0.0]; 3],
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();

    for vertex in &data.meshes[0].vertices {
        assert_eq!(vertex.tangent, [0, -i16::MAX, 0, i16::MAX]);
    }
}

#[test]
fn keeps_generated_tangents_on_vertices_shared_with_a_degenerate_triangle() {
    let model = triangle_model(
        &[0, 1, 2, 0, 0, 1],
        6,
        r#"{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3}"#,
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();

    for vertex in &data.meshes[0].vertices {
        assert_eq!(vertex.tangent, [i16::MAX, 0, 0, -i16::MAX]);
    }
}

#[test]
fn keeps_indices_local_to_each_primitive_in_a_shared_mesh() {
    let indices = [2, 0, 1, 1, 2, 0];
    let model = triangle_model(
        &indices,
        indices.len(),
        r#"{"attributes": {"POSITION": 0}},
           {"attributes": {"POSITION": 0}, "indices": 3}"#,
    );
    let data = GltfData::from_memory(model.as_bytes()).unwrap();
    let mesh = &data.meshes[0];
    let [first, second] = mesh.primitives.as_slice() else {
        panic!("expected two primitives, got {}", mesh.primitives.len());
    };

    assert_eq!(first.vertex_range(), 0..3);
    assert_eq!(second.vertex_range(), 3..6);
    assert_eq!(first.index_range(), 0..3);
    assert_eq!(second.index_range(), 3..9);
    assert_eq!(mesh.vertices.len(), 6);
    assert_eq!(mesh.indices.len(), 9);
    assert_eq!(mesh.indices[first.index_range()], [0, 1, 2]);
    assert_eq!(mesh.indices[second.index_range()], indices.map(u32::from));
}
