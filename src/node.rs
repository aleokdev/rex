use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

pub type NodeId = usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub path: PathBuf,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub rooms: Vec<usize>,
}

/// Returns a node and its children from a directory path.
pub fn generate_nodes(path: &Path) -> crate::Result<Vec<Node>> {
    let mut nodes = Vec::<Node>::with_capacity(100);

    let mut to_process = vec![Node {
        path: path.to_owned(),
        parent: None,
        children: vec![],
        rooms: vec![],
    }];

    while let Some(node_being_processed) = to_process.pop() {
        let node_being_processed_idx = nodes.len();

        if let Some(parent) = node_being_processed.parent {
            nodes[parent].children.push(node_being_processed_idx);
        }

        let dir_entries = fs::read_dir(&node_being_processed.path);
        match dir_entries {
            Ok(dir_entries) => {
                for dir_entry in dir_entries {
                    let Ok(dir_entry) = dir_entry else { continue; };

                    if dir_entry.file_type()?.is_dir() {
                        to_process.push(Node {
                            path: dir_entry.path(),
                            parent: Some(node_being_processed_idx),
                            children: vec![],
                            rooms: vec![],
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
