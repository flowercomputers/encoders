//! Locating model files on disk, downloading them from the hub on a miss.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::{Error, Source};

pub(crate) struct Files {
    root: PathBuf,
    hub: Option<(String, String)>,
}

impl Files {
    pub fn resolve(
        source: &Source,
        cache_dir: Option<&Path>,
        out_dir: &Path,
    ) -> Result<Self, Error> {
        match source {
            Source::Local(dir) => {
                if !dir.is_dir() {
                    return Err(Error::MissingFile(format!(
                        "{} is not a directory",
                        dir.display()
                    )));
                }
                Ok(Files {
                    root: dir.clone(),
                    hub: None,
                })
            }
            Source::Hub { repo, revision } => {
                let base = cache_dir
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| default_cache_dir(out_dir));
                let root = base
                    .join(repo.replace('/', "--"))
                    .join(revision.replace('/', "--"));
                fs::create_dir_all(&root)?;
                Ok(Files {
                    root,
                    hub: Some((repo.clone(), revision.clone())),
                })
            }
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to `rel` if it exists locally or could be downloaded; `None` if
    /// the hub says it does not exist.
    pub fn get(&self, rel: &str) -> Result<Option<PathBuf>, Error> {
        let path = self.root.join(rel);
        if path.is_file() {
            rerun_if_changed(&path);
            return Ok(Some(path));
        }
        let Some((repo, revision)) = &self.hub else {
            return Ok(None);
        };
        let marker = self
            .root
            .join(format!("{}.missing", rel.replace('/', "--")));
        if marker.exists() {
            return Ok(None);
        }
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{rel}");
        if download(&url, &path)? {
            rerun_if_changed(&path);
            Ok(Some(path))
        } else {
            fs::write(&marker, b"")?;
            Ok(None)
        }
    }

    pub fn require(&self, rel: &str) -> Result<PathBuf, Error> {
        self.get(rel)?
            .ok_or_else(|| Error::MissingFile(format!("{rel} (looked in {})", self.root.display())))
    }
}

fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}

/// `target/ese-cache` when `OUT_DIR` lives under a `target` directory, so all
/// consumers in a workspace share one download; otherwise inside `OUT_DIR`.
fn default_cache_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .ancestors()
        .find(|p| p.file_name().is_some_and(|f| f == "target"))
        .map(|t| t.join("ese-cache"))
        .unwrap_or_else(|| out_dir.join("ese-cache"))
}

/// Streams `url` to `path`. Returns `false` on HTTP 404.
fn download(url: &str, path: &Path) -> Result<bool, Error> {
    eprintln!("ese-build: downloading {url}");
    let mut resp = minreq::get(url)
        .with_timeout(600)
        .send_lazy()
        .map_err(|source| Error::Network {
            url: url.to_string(),
            source,
        })?;
    match resp.status_code {
        200 => {}
        404 => return Ok(false),
        status => {
            return Err(Error::Http {
                url: url.to_string(),
                status,
            });
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("part");
    {
        let mut file = io::BufWriter::new(fs::File::create(&tmp)?);
        io::copy(&mut resp, &mut file)?;
    }
    fs::rename(&tmp, path)?;
    Ok(true)
}
