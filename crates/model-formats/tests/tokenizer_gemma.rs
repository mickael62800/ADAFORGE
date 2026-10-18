use adaforge_application::ModelInspector;
use adaforge_domain::TokenizeOptions;
use adaforge_model_formats::GgufInspector;

#[test]
#[ignore = "Nécessite le GGUF Gemma 4 local ; aucune exécution GPU"]
fn tokenizer_matches_reference_fixtures() {
    let model = std::env::var_os("ADAFORGE_TEST_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf")
        });
    let tokenizer = GgufInspector::default().read_tokenizer(&model).unwrap();
    let expected = tokenizer
        .encode(
            "<|turn>user\nBonjour<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
            TokenizeOptions {
                add_bos: true,
                parse_special: true,
            },
        )
        .unwrap();
    assert_eq!(tokenizer.encode_user_turn("  Bonjour  ").unwrap(), expected);
    let literal = tokenizer.encode_user_turn("<turn|>").unwrap();
    let decoded = tokenizer.decode(&literal, true).unwrap();
    assert!(decoded.contains("user\n<turn|><turn|>"));
    let mut count = 0;
    let fixtures = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gemma4_tokenizer.tsv"),
    )
    .unwrap();
    for line in fixtures
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let columns: Vec<_> = line.split('\t').collect();
        assert_eq!(columns.len(), 4);
        let bytes = (0..columns[2].len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&columns[2][i..i + 2], 16).unwrap())
            .collect();
        let text = String::from_utf8(bytes).unwrap();
        let expected: Vec<u32> = if columns[3].is_empty() {
            vec![]
        } else {
            columns[3].split(',').map(|s| s.parse().unwrap()).collect()
        };
        let options = TokenizeOptions {
            add_bos: columns[0] == "1",
            parse_special: columns[1] == "1",
        };
        let actual = tokenizer.encode(&text, options).unwrap();
        assert_eq!(
            actual, expected,
            "input {text:?}, BOS {}, parse {}",
            options.add_bos, options.parse_special
        );
        count += 1;
    }
    assert!(count >= 87, "La comparaison doit couvrir au moins 87 cas");
}
