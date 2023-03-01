use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

pub type NodeId = usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RexFile {
    name: OsString,
}

impl RexFile {
    pub fn new(name: OsString) -> Self {
        Self { name }
    }

    pub fn name(&self) -> &OsString {
        &self.name
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Node {
    pub path: PathBuf,
    pub parent: Option<usize>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<usize>,
    pub rooms: Vec<usize>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<RexFile>,
}

/// Returns a node and its children from a directory path.
pub fn generate_nodes(path: &Path) -> crate::Result<Vec<Node>> {
    let mut nodes = Vec::<Node>::with_capacity(100);

    let mut to_process = vec![Node {
        path: path.to_owned(),
        parent: None,
        children: vec![],
        rooms: vec![],
        files: vec![],
    }];

    while let Some(mut node_being_processed) = to_process.pop() {
        let node_being_processed_idx = nodes.len();

        if let Some(parent) = node_being_processed.parent {
            nodes[parent].children.push(node_being_processed_idx);
        }

        let dir_entries = fs::read_dir(&node_being_processed.path);
        match dir_entries {
            Ok(dir_entries) => {
                for dir_entry in dir_entries {
                    let Ok(dir_entry) = dir_entry else { continue; };

                    let file_type = dir_entry.file_type()?;
                    if file_type.is_dir() {
                        to_process.push(Node {
                            path: dir_entry.path(),
                            parent: Some(node_being_processed_idx),
                            ..Default::default()
                        });
                    }
                    if file_type.is_file() {
                        node_being_processed.files.push(RexFile {
                            name: dir_entry.file_name(),
                        });
                    }
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => (),
            Err(err) => return Err(err.into()),
        }

        nodes.push(node_being_processed);
    }

    Ok(nodes)
}
