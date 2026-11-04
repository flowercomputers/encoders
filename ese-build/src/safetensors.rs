//! Minimal safetensors reader: header parsing plus f32 decoding of the dtypes
//! static embedding models are shipped in.

use std::{fs, path::Path};

use crate::Error;

pub(crate) struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub(crate) struct SafeTensors {
    data: Vec<u8>,
    base: usize,
    tensors: Vec<Tensor>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let data = fs::read(path)?;
        if data.len() < 8 {
            return Err(Error::UnsupportedTensor("file too short".into()));
        }
        let meta_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&data[8..8 + meta_len]).map_err(|source| Error::Json {
                file: path.to_path_buf(),
                source,
            })?;
        let obj = header
            .as_object()
            .ok_or_else(|| Error::UnsupportedTensor("header is not an object".into()))?;
        let mut tensors = Vec::new();
        for (name, t) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = t["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<usize> = t["shape"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_default();
            let offs = t["data_offsets"].as_array();
            let (start, end) = match offs.map(|o| (o.first(), o.get(1))) {
                Some((Some(s), Some(e))) => (
                    s.as_u64().unwrap_or(0) as usize,
                    e.as_u64().unwrap_or(0) as usize,
                ),
                _ => {
                    return Err(Error::UnsupportedTensor(format!(
                        "tensor {name} has no data_offsets"
                    )));
                }
            };
            tensors.push(Tensor {
                name: name.clone(),
                dtype,
                shape,
                start,
                end,
            });
        }
        Ok(SafeTensors {
            data,
            base: 8 + meta_len,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The token-embedding matrix: a well-known name first, else the only 2-D
    /// tensor that is not one of model2vec's auxiliary tensors.
    pub fn find_embedding(&self) -> Result<&Tensor, Error> {
        for name in ["embedding.weight", "embeddings", "0"] {
            if let Some(t) = self.get(name) {
                if t.shape.len() != 2 {
                    return Err(Error::UnsupportedTensor(format!(
                        "{name} has shape {:?}, expected 2-D",
                        t.shape
                    )));
                }
                return Ok(t);
            }
        }
        let candidates: Vec<&Tensor> = self
            .tensors
            .iter()
            .filter(|t| t.shape.len() == 2 && !matches!(t.name.as_str(), "weights" | "mapping"))
            .collect();
        match candidates.as_slice() {
            [t] => Ok(t),
            [] => Err(Error::UnsupportedTensor(format!(
                "no 2-D embedding tensor; found {:?}",
                self.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
            ))),
            many => Err(Error::UnsupportedTensor(format!(
                "several 2-D tensors, cannot pick the embedding: {:?}",
                many.iter().map(|t| &t.name).collect::<Vec<_>>()
            ))),
        }
    }

    fn bytes(&self, t: &Tensor) -> Result<&[u8], Error> {
        self.data
            .get(self.base + t.start..self.base + t.end)
            .ok_or_else(|| Error::UnsupportedTensor(format!("{} points outside the file", t.name)))
    }

    /// Rows of a 2-D tensor, each truncated to `dims` columns.
    pub fn read_rows(&self, t: &Tensor, dims: usize) -> Result<Vec<Vec<f32>>, Error> {
        let [rows, cols] = t.shape[..] else {
            return Err(Error::UnsupportedTensor(format!("{} is not 2-D", t.name)));
        };
        let dims = dims.min(cols);
        let esz = elem_size(&t.dtype)?;
        let bytes = self.bytes(t)?;
        if bytes.len() != rows * cols * esz {
            return Err(Error::UnsupportedTensor(format!(
                "{}: {} bytes for shape {:?} of {}",
                t.name,
                bytes.len(),
                t.shape,
                t.dtype
            )));
        }
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &bytes[r * cols * esz..][..dims * esz];
            out.push(
                row.chunks_exact(esz)
                    .map(|b| read_elem(&t.dtype, b))
                    .collect(),
            );
        }
        Ok(out)
    }

    /// All elements of a tensor as f32, flattened.
    pub fn read_f32_1d(&self, t: &Tensor) -> Result<Vec<f32>, Error> {
        let esz = elem_size(&t.dtype)?;
        Ok(self
            .bytes(t)?
            .chunks_exact(esz)
            .map(|b| read_elem(&t.dtype, b))
            .collect())
    }

    /// All elements of an integer tensor as indices, flattened.
    pub fn read_indices(&self, t: &Tensor) -> Result<Vec<usize>, Error> {
        let bytes = self.bytes(t)?;
        let read = |esz: usize, f: fn(&[u8]) -> i64| -> Vec<i64> {
            bytes.chunks_exact(esz).map(f).collect()
        };
        let raw = match t.dtype.as_str() {
            "I32" => read(4, |b| i32::from_le_bytes(b.try_into().unwrap()) as i64),
            "U32" => read(4, |b| u32::from_le_bytes(b.try_into().unwrap()) as i64),
            "I64" => read(8, |b| i64::from_le_bytes(b.try_into().unwrap())),
            "U64" => read(8, |b| u64::from_le_bytes(b.try_into().unwrap()) as i64),
            "I16" => read(2, |b| i16::from_le_bytes(b.try_into().unwrap()) as i64),
            "U16" => read(2, |b| u16::from_le_bytes(b.try_into().unwrap()) as i64),
            other => {
                return Err(Error::UnsupportedTensor(format!(
                    "{} has non-integer dtype {other}",
                    t.name
                )));
            }
        };
        raw.into_iter()
            .map(|v| {
                usize::try_from(v).map_err(|_| {
                    Error::UnsupportedTensor(format!("{} contains negative index {v}", t.name))
                })
            })
            .collect()
    }
}

fn elem_size(dtype: &str) -> Result<usize, Error> {
    Ok(match dtype {
        "I8" | "U8" => 1,
        "F16" | "BF16" => 2,
        "F32" => 4,
        "F64" => 8,
        other => return Err(Error::UnsupportedTensor(format!("dtype {other}"))),
    })
}

fn read_elem(dtype: &str, b: &[u8]) -> f32 {
    match dtype {
        "I8" => b[0] as i8 as f32,
        "U8" => b[0] as f32,
        "F16" => f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())),
        "BF16" => f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16),
        "F32" => f32::from_le_bytes(b.try_into().unwrap()),
        "F64" => f64::from_le_bytes(b.try_into().unwrap()) as f32,
        _ => unreachable!("dtype validated by elem_size"),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let (f_exp, f_mant) = match exp {
        0 => {
            if mant == 0 {
                (0u32, 0u32)
            } else {
                let mut m = mant;
                let mut e = 1u32;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e += 1;
                }
                (127 - 15 + 1 - e, (m & 0x3FF) << 13)
            }
        }
        31 => (255u32, mant << 13),
        _ => (exp + 127 - 15, mant << 13),
    };
    f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_samples() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333252).abs() < 1e-5);
    }

    #[test]
    fn parses_header_and_rows() {
        let header = r#"{"__metadata__":{"x":"y"},"embeddings":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"mapping":{"dtype":"I32","shape":[2],"data_offsets":[24,32]}}"#;
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for v in [1i32, 0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let dir = std::env::temp_dir().join(format!("ese-build-st-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        fs::write(&path, &data).unwrap();
        let st = SafeTensors::open(&path).unwrap();
        let emb = st.find_embedding().unwrap();
        assert_eq!(
            st.read_rows(emb, 2).unwrap(),
            vec![vec![1.0, 2.0], vec![4.0, 5.0]]
        );
        assert_eq!(
            st.read_indices(st.get("mapping").unwrap()).unwrap(),
            vec![1, 0]
        );
        fs::remove_dir_all(&dir).ok();
    }
}
