use coreml_native::{compile_model, ComputeUnits, Model};
use std::path::PathBuf;

fn main() {
    let path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("OPENSTETHO_MODEL").map(PathBuf::from))
        .or_else(|| std::env::var_os("EKO_MODEL").map(PathBuf::from))
        .expect("pass model path as arg or OPENSTETHO_MODEL env");
    println!("source: {}", path.display());
    let compiled = match compile_model(&path) {
        Ok(p) => {
            println!("compiled → {}", p.display());
            p
        }
        Err(e) => {
            eprintln!("compile_model FAILED: {e}");
            std::process::exit(1);
        }
    };
    match Model::load(&compiled, ComputeUnits::All) {
        Ok(m) => {
            println!("OK");
            for f in m.inputs() {
                println!(
                    "  input  {} shape={:?} dtype={:?}",
                    f.name(),
                    f.shape(),
                    f.data_type()
                );
            }
            for f in m.outputs() {
                println!(
                    "  output {} shape={:?} dtype={:?}",
                    f.name(),
                    f.shape(),
                    f.data_type()
                );
            }
        }
        Err(e) => {
            eprintln!("Model::load FAILED: {e}");
            std::process::exit(1);
        }
    }
}
