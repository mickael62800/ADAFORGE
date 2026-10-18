use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match adaforge_cli::run(
        &args,
        &adaforge_cuda_backend::CudaDiagnostics,
        &adaforge_model_formats::GgufInspector::default(),
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("AdaForge : {error}");
            ExitCode::FAILURE
        }
    }
}
