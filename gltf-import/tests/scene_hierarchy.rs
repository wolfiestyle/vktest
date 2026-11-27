use glam::Vec3;
use gltf_import::{GltfData, NodeId};

const MODEL: &[u8] = br#"{
    "asset": {"version": "2.0"},
    "nodes": [
        {"translation": [1, 2, 0]},
        {"translation": [4, 0, 0], "children": [0]},
        {"translation": [10, 0, 0], "scale": [2, 3, 1], "children": [1]},
        {"translation": [-5, 0, 0]},
        {"translation": [0, 0, 7]}
    ],
    "scenes": [{"nodes": [2, 3]}, {"nodes": [4]}]
}"#;

#[test]
fn resolves_world_transforms_when_children_precede_parents() {
    let data = GltfData::from_memory(MODEL).unwrap();
    let grandchild = &data[NodeId(0)];
    let parent = &data[NodeId(1)];
    let root = &data[NodeId(2)];

    assert_eq!(grandchild.parent, Some(parent.id));
    assert_eq!(parent.parent, Some(root.id));
    assert_eq!(root.parent, None);

    assert_eq!(grandchild.local_transform.transform_point3(Vec3::ZERO), Vec3::new(1.0, 2.0, 0.0));
    assert_eq!(root.transform.transform_point3(Vec3::ZERO), Vec3::new(10.0, 0.0, 0.0));
    assert_eq!(parent.transform.transform_point3(Vec3::ZERO), Vec3::new(18.0, 0.0, 0.0));
    assert_eq!(grandchild.transform.transform_point3(Vec3::ZERO), Vec3::new(20.0, 6.0, 0.0));
    assert_eq!(grandchild.transform.transform_point3(Vec3::X), Vec3::new(22.0, 6.0, 0.0));
}

#[test]
fn visits_all_nodes_in_the_selected_scene() {
    let data = GltfData::from_memory(MODEL).unwrap();
    let mut first_scene_nodes: Vec<_> = data.scenes[0].nodes(&data).map(|node| node.id.0).collect();
    first_scene_nodes.sort_unstable();

    assert_eq!(first_scene_nodes, [0, 1, 2, 3]);
    assert_eq!(data.scenes[1].nodes(&data).map(|node| node.id).collect::<Vec<_>>(), [NodeId(4)]);
}
