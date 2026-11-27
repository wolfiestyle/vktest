use std::{path::Path, process::Command};

#[test]
fn exits_with_a_model_import_diagnostic_for_an_out_of_range_index() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/invalid-index.gltf");
    let output = Command::new(env!("CARGO_BIN_EXE_vktest"))
        .arg("--model")
        .arg(&model)
        // Keep the subprocess headless even if a regression lets the model load.
        .arg("--skybox")
        .arg(&model)
        .output()
        .expect("failed to launch vktest");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("index 3 at element 2 is outside the 3 vertices"), "{stderr}");
}
