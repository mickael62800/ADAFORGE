//! Adaptateur CLI, indépendant de CUDA.
use adaforge_application::{GpuDiagnostics, check_gpu};

pub fn run(
    args: &[String],
    backend: &impl GpuDiagnostics,
    inspector: &impl adaforge_application::ModelInspector,
) -> Result<(), String> {
    match args {
        [] => {
            print_help();
            Ok(())
        }
        [command] if command == "--help" || command == "-h" => {
            print_help();
            Ok(())
        }
        [command] if command == "gpu-check" => {
            let report = check_gpu(backend).map_err(|e| e.to_string())?;
            let gib = 1024.0_f64.powi(3);
            println!("GPU : {}", report.device_name);
            println!(
                "Capacité CUDA : {}.{} (sm_{}{})",
                report.compute_capability.0,
                report.compute_capability.1,
                report.compute_capability.0,
                report.compute_capability.1
            );
            println!(
                "VRAM visible : {:.2} GiB ; libre : {:.2} GiB (avant buffers de test)",
                report.total_memory_bytes as f64 / gib,
                report.free_memory_bytes as f64 / gib
            );
            println!(
                "Addition FP32 : {} éléments vérifiés",
                report.checked_elements
            );
            println!("Erreur absolue maximale : {}", report.max_absolute_error);
            println!(
                "Kernel seul, après échauffement : {:.4} ms",
                report.kernel_ms
            );
            println!("Diagnostic réussi. Cette mesure unique n’est pas un benchmark d’inférence.");
            Ok(())
        }
        [command, path] if command == "model-info" => {
            let info = adaforge_application::inspect_model(inspector, std::path::Path::new(path)).map_err(|e| e.to_string())?;
            println!("GGUF v{} — {:.3} Gio", info.format_version, info.file_bytes as f64 / 1073741824.0);
            for (key,value) in &info.metadata {
                if key.starts_with("general.") || key.starts_with("gemma4.") {
                    println!("{key} : {value:?}");
                }
            }
            let mut types = std::collections::BTreeMap::new();
            for tensor in &info.tensors { *types.entry(format!("{:?}", tensor.weight_type)).or_insert(0_u64) += 1; }
            println!("{} tenseurs ; formats : {types:?}", info.tensors.len());
            println!("Début des poids : {} octets ; dimensions et plages vérifiées.", info.data_offset);
            println!("Métadonnées et inventaire uniquement. Aucune inférence ni allocation GPU.");
            Ok(())
        }
        [command, path] | [command, path, _] if command == "matvec-check" => {
            let tensor=args.get(2).map(String::as_str).unwrap_or("blk.0.attn_q.weight");
            let report=adaforge_application::check_model_matvec(inspector,backend,std::path::Path::new(path),tensor).map_err(|e| e.to_string())?;
            println!("Tenseur : {tensor} (Q4_0)");
            println!("Matrice : {} lignes × {} colonnes",report.rows,report.columns);
            println!("{} vecteurs vérifiés sur toutes les lignes, référence CPU FP64",report.checked_vectors);
            println!("Erreur absolue maximale : {:.8}",report.max_absolute_error);
            println!("Ratio de tolérance maximal : {:.5} (limite 1 ; atol=rtol=1e-4)",report.max_tolerance_ratio);
            println!("Kernel seul : médiane {:.4} ms sur {} mesures après échauffement",report.median_kernel_ms,report.repetitions);
            println!("Calcul validé. Kernel initial, aucune comparaison de performance avec un moteur existant.");
            Ok(())
        }
        [command, path] | [command,path,_] if command=="rmsnorm-check" => {
            let tensor=args.get(2).map(String::as_str).unwrap_or("blk.0.attn_norm.weight");
            let report=adaforge_application::check_model_rms_norm(inspector,backend,std::path::Path::new(path),tensor).map_err(|e| e.to_string())?;
            println!("RMSNorm : {tensor} (échelle F32 directe, sans ajout de 1)");
            println!("Largeur : {} ; epsilon : {}",report.width,report.epsilon);
            println!("{} vecteurs vérifiés, référence CPU FP64",report.vectors);
            println!("Erreur absolue maximale : {:.8}",report.max_absolute_error);
            println!("Ratio de tolérance maximal : {:.5} (limite 1 ; atol=rtol=2e-5)",report.max_tolerance_ratio);
            println!("Kernel seul, lot complet : médiane {:.4} ms sur 10 mesures après échauffement",report.median_kernel_ms);
            println!("RMSNorm validée. Pas encore de couche complète ni de benchmark d’inférence.");
            Ok(())
        }
        [command,path] if command=="projection-check" => {
            let report=adaforge_application::check_model_projection(inspector,backend,std::path::Path::new(path)).map_err(|e| e.to_string())?;
            println!("Chaîne : blk.0.attn_norm.weight → blk.0.attn_q.weight");
            println!("RMSNorm {} → projection Q4_0 {} × {}",report.columns,report.rows,report.columns);
            println!("{} vecteurs vérifiés sur toutes les sorties ; référence CPU FP64, intermédiaire arrondi FP32",report.checked_vectors);
            println!("Erreur absolue maximale : {:.8}",report.max_absolute_error);
            println!("Ratio de tolérance maximal : {:.5} (limite 1 ; atol=rtol=1e-4)",report.max_tolerance_ratio);
            println!("Chaîne GPU seule : médiane {:.4} ms sur {} mesures après échauffement",report.median_kernel_ms,report.repetitions);
            println!("Intermédiaire conservé sur GPU ; poids et buffers réutilisés. Deux kernels, pas de fusion.");
            println!("Validation d’un début de couche uniquement, pas encore d’inférence complète.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="qkv-check" => {
            let layer=args.get(2).map(|v| v.parse::<u32>().map_err(|_| "Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_qkv(inspector,backend,std::path::Path::new(path),layer).map_err(|e| e.to_string())?;
            println!("Couche {layer} : RMSNorm {} → projections Q/K/V Q4_0",report.columns);
            for (name,error) in ["Q","K","V"].iter().zip(&report.projections) {
                println!("{name} : {} × {} ; erreur max {:.8} ; ratio tolérance {:.5}",error.rows,report.columns,error.max_absolute_error,error.max_tolerance_ratio);
            }
            println!("{} vecteurs vérifiés, référence CPU FP64 avec frontière FP32 ; atol=rtol=1e-4",report.checked_vectors);
            println!("Chaîne GPU seule : médiane {:.4} ms sur {} mesures après échauffement",report.median_kernel_ms,report.repetitions);
            if report.value_uses_key_weights { println!("V utilise les poids K (attention alternative Gemma 4, matrice V absente)."); }
            println!("Une normalisation partagée, trois projections ; intermédiaire GPU et buffers réutilisés.");
            println!("Q/K/V bruts validés ; normalisation des têtes, RoPE et attention restent à ajouter.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="qknorm-check" => {
            let layer=args.get(2).map(|v| v.parse::<u32>().map_err(|_| "Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_qk_norm(inspector,backend,std::path::Path::new(path),layer).map_err(|e| e.to_string())?;
            println!("Couche {layer} : RMSNorm → Q/K/V → normalisation par tête Q/K");
            println!("Têtes Q : {} ; têtes K/V : {} ; largeur : {}",report.q_heads,report.kv_heads,report.head_width);
            for (name,error) in ["Q normalisé","K normalisé","V brut"].iter().zip(&report.qkv.projections) {
                println!("{name} : erreur max {:.8} ; ratio tolérance {:.5}",error.max_absolute_error,error.max_tolerance_ratio);
            }
            println!("{} vecteurs vérifiés ; référence CPU FP64 avec frontières FP32 ; atol=rtol=1e-4",report.qkv.checked_vectors);
            println!("Chaîne GPU seule : médiane {:.4} ms sur {} mesures",report.qkv.median_kernel_ms,report.qkv.repetitions);
            if report.qkv.value_uses_key_weights {println!("Poids K/V partagés ; sorties séparées, V brut préservé.");}
            println!("Aucun transfert CPU intermédiaire. RoPE, normalisation V et attention restent à ajouter.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="rope-check" => {
            let layer=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_rope(inspector,backend,std::path::Path::new(path),layer).map_err(|e|e.to_string())?;
            println!("Couche {layer} : RMSNorm → Q/K/V → normes Q/K → RoPE NeoX");
            println!("Attention {} ; largeur {} ; base {} ; {} paires à fréquence atténuée",if report.local_attention {"locale"} else {"globale"},report.heads.head_width,report.base,report.attenuated_pairs);
            println!("Positions testées : {:?}",report.positions);
            for (name,error) in ["Q après RoPE","K après RoPE","V brut préservé"].iter().zip(&report.heads.qkv.projections) {
                println!("{name} : erreur max {:.8} ; ratio tolérance {:.5}",error.max_absolute_error,error.max_tolerance_ratio);
            }
            println!("{} vecteurs vérifiés ; référence CPU FP64 avec frontières FP32 ; atol=rtol=1e-4",report.heads.qkv.checked_vectors);
            println!("Chaîne GPU seule : médiane {:.4} ms sur {} mesures",report.heads.qkv.median_kernel_ms,report.heads.qkv.repetitions);
            println!("Intermédiaires sur GPU. Normalisation V, cache KV et attention restent à ajouter.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="vnorm-check" => {
            let layer=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_vnorm(inspector,backend,std::path::Path::new(path),layer).map_err(|e|e.to_string())?;
            println!("Couche {layer} : RMSNorm → Q/K/V → normes Q/K → RoPE Q/K + norme V sans échelle");
            println!("Attention {} ; largeur {} ; base {} ; {} paires à fréquence atténuée",if report.local_attention {"locale"} else {"globale"},report.heads.head_width,report.base,report.attenuated_pairs);
            println!("Positions testées : {:?}",report.positions);
            for (name,error) in ["Q après RoPE","K après RoPE","V normalisé sans échelle"].iter().zip(&report.heads.qkv.projections) {
                println!("{name} : erreur max {:.8} ; ratio tolérance {:.5}",error.max_absolute_error,error.max_tolerance_ratio);
            }
            println!("{} vecteurs vérifiés ; référence CPU FP64 avec frontières FP32 ; atol=rtol=1e-4",report.heads.qkv.checked_vectors);
            println!("Chaîne GPU seule : médiane {:.4} ms sur {} mesures",report.heads.qkv.median_kernel_ms,report.heads.qkv.repetitions);
            println!("Intermédiaires sur GPU. Q/K tournés, V normalisé sans rotation. Cache KV et attention restent à ajouter.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="kv-cache-check" => {
            let layer=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_kv_cache(inspector,backend,std::path::Path::new(path),layer).map_err(|e|e.to_string())?;
            println!("Cache KV FP32 couche {layer} : {} tokens × {} têtes × {} canaux",report.capacity,report.heads,report.width);
            println!("Allocation K+V : {} octets ; ordre [token][tête][canal]",report.allocated_bytes);
            println!("{} valeurs vérifiées exactement après ajouts 3+1+4 tokens, dépassement et réutilisation",report.verified_values);
            println!("Capacité dépassée refusée : {} ; reset et nouveau préfixe vérifiés : {}",report.overflow_rejected,report.reset_verified);
            println!("Écritures GPU → GPU. Diagnostic avec données synthétiques ; connexion à la chaîne Gemma et attention à venir.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="kv-chain-check" => {
            let layer=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_cached_kv(inspector,backend,std::path::Path::new(path),layer).map_err(|e|e.to_string())?;
            let stats=report.heads.qkv.cache.as_ref().ok_or("Rapport cache absent")?;
            println!("Couche {layer} : projections Gemma → normes Q/K/V → RoPE Q/K → cache KV GPU");
            println!("{} tokens successifs ; {} têtes KV × {} canaux ; positions {:?}",stats.tokens,report.heads.kv_heads,report.heads.head_width,report.positions);
            println!("{} valeurs de préfixes vérifiées face au CPU",stats.verified_values);
            println!("Erreur absolue maximale cache : {:.8} ; ratio tolérance {:.5} (limite 1, atol=rtol=1e-4)",stats.max_absolute_error,stats.max_tolerance_ratio);
            println!("Tokens précédents inchangés exactement : {} ; dépassement refusé : {}",stats.previous_tokens_unchanged,stats.overflow_rejected);
            println!("K tourné et V normalisé copiés directement GPU → cache GPU, sans retour CPU intermédiaire.");
            println!("Poids réels, activations de test synthétiques. Attention et génération de texte restent à ajouter.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="attention-check" => {
            let layer=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_attention(inspector,backend,std::path::Path::new(path),layer).map_err(|e|e.to_string())?;
            let stats=report.heads.qkv.attention.as_ref().ok_or("Rapport attention absent")?;
            println!("Couche {layer} : préparation Q/K/V → cache KV → attention causale GPU");
            println!("{} têtes Q / {} têtes KV ; largeur {} ; échelle {} ; fenêtre {:?}",report.heads.q_heads,report.heads.kv_heads,report.heads.head_width,stats.scale,stats.window);
            println!("Positions {:?} ; {} valeurs de sortie vérifiées face au CPU",report.positions,stats.checked_values);
            println!("Erreur absolue maximale attention : {:.8} ; ratio tolérance {:.5} (limite 1, atol=rtol=1e-4)",stats.max_absolute_error,stats.max_tolerance_ratio);
            println!("Masque causal, regroupement des têtes et softmax stable. Intermédiaires et cache sur GPU.");
            println!("Poids Gemma réels, activations synthétiques. Projection de sortie et FFN restent à ajouter.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="attention-branch-check" => {
            let layer=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_attention_branch(inspector,backend,std::path::Path::new(path),layer).map_err(|e|e.to_string())?;
            let stats=report.heads.qkv.output_branch.as_ref().ok_or("Rapport branche absent")?;
            println!("Couche {layer} : Q/K/V → cache → attention → projection sortie → RMSNorm → résiduel");
            println!("Positions {:?} ; {} valeurs de sortie vérifiées ; largeur finale {}",report.positions,stats.checked_values,stats.width);
            println!("Erreur absolue maximale branche : {:.8} ; ratio tolérance {:.5} (limite 1, atol=rtol=1e-4)",stats.max_absolute_error,stats.max_tolerance_ratio);
            println!("Résiduel = entrée brute de la branche ; tous les calculs restent sur GPU.");
            println!("Poids Gemma réels, activations synthétiques. FFN et couche complète restent à ajouter.");
            Ok(())
        }
        [command,path] | [command,path,_] if command=="layer-check" => {
            let layer=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Indice de couche invalide".to_string())).transpose()?.unwrap_or(0);
            let report=adaforge_application::check_model_layer(inspector,backend,std::path::Path::new(path),layer).map_err(|e|e.to_string())?;
            let stats=report.heads.qkv.ffn.as_ref().ok_or("Rapport couche absent")?;
            println!("Couche dense Gemma {layer} : attention + résiduel → RMSNorm → gate/up GELU(tanh) → down → RMSNorm + résiduel + échelle de couche");
            println!("Positions {:?} ; {} sorties vérifiées ; largeur finale {}",report.positions,stats.checked_values,stats.width);
            println!("Erreur absolue maximale couche : {:.8} ; ratio tolérance {:.5} (limite 1, atol=rtol=1e-4)",stats.max_absolute_error,stats.max_tolerance_ratio);
            println!("Tous les calculs et le cache restent sur GPU. Poids réels, activations synthétiques.");
            println!("Une couche dense testée, pas encore les 48 couches ni la génération de texte.");
            Ok(())
        }
        [command,path,ids @ ..] if command=="embedding-check" => {
            let ids=ids.iter().map(|s|s.parse::<u32>().map_err(|_|format!("ID token invalide : {s}"))).collect::<Result<Vec<_>,_>>()?;
            let report=adaforge_application::check_model_embeddings(inspector,backend,std::path::Path::new(path),&ids).map_err(|e|e.to_string())?;
            println!("Embeddings Gemma Q6_K : vocabulaire {} ; largeur {}",report.vocabulary,report.width);
            println!("IDs vérifiés (ordre et répétitions conservés) : {:?}",report.token_ids);
            println!("Déquantification brute et multiplication par sqrt(largeur) = {}",report.scale);
            println!("{} valeurs vérifiées ; erreur max {:.8} ; ratio tolérance {:.5} (atol=rtol=1e-6)",report.verified_values,report.max_absolute_error,report.max_tolerance_ratio);
            println!("Seules les lignes demandées sont lues et transférées, pas la table complète.");
            println!("IDs numériques uniquement ; tokenisation et connexion aux couches à venir.");
            Ok(())
        }
        [command,path] if command=="embedding-layer-check" => {
            let report=adaforge_application::check_model_embedded_layer(inspector,backend,std::path::Path::new(path)).map_err(|e|e.to_string())?;
            let stats=report.heads.qkv.ffn.as_ref().ok_or("Rapport couche absent")?;
            let ids=report.heads.qkv.embedding_ids.as_ref().ok_or("IDs embeddings absents")?;
            println!("IDs réels → embeddings Q6_K sur GPU → première couche dense Gemma complète");
            println!("IDs : {ids:?} ; positions {:?}",report.positions);
            println!("{} sorties vérifiées face au CPU ; largeur {}",stats.checked_values,stats.width);
            println!("Erreur absolue maximale : {:.8} ; ratio tolérance {:.5} (limite 1, atol=rtol=1e-4)",stats.max_absolute_error,stats.max_tolerance_ratio);
            println!("Embeddings GPU transmis à la couche par copie GPU → GPU ; aucune activation CPU utilisée comme entrée GPU.");
            println!("Un seul bloc vérifié ; tokenisation du texte, 47 blocs suivants et génération à venir.");
            Ok(())
        }
        [command,path,text] | [command,path,text,_] | [command,path,text,_,_] if command=="bench" => {
            let count=args.get(3).map(|v|v.parse::<u32>().map_err(|_|"Nombre de tokens invalide".to_string())).transpose()?.unwrap_or(24);
            let runs=args.get(4).map(|v|v.parse::<u32>().map_err(|_|"Nombre de mesures invalide".to_string())).transpose()?.unwrap_or(5);
            if !(1..=20).contains(&runs){return Err("Choisir de 1 à 20 mesures".into());}
            let loading=std::time::Instant::now();
            let mut session=adaforge_application::open_chat(inspector,backend,std::path::Path::new(path),count).map_err(|e|e.to_string())?;
            println!("Chargement : {:.3} s ; un échauffement, puis {runs} mesures ; préfixe KV remis à zéro entre requêtes.",loading.elapsed().as_secs_f64());
            session.send(text).map_err(|e|e.to_string())?;
            for run in 0..runs{
                session.reset().map_err(|e|e.to_string())?;
                let report=session.send(text).map_err(|e|e.to_string())?;
                let stats=report.stats.as_ref().ok_or("Mesures backend absentes")?;
                print_stats(Some(stats),&mut std::io::stdout().lock())?;
                println!("BENCH_SAMPLE {{\"run\":{run},\"prompt_ids\":{:?},\"generated_ids\":{:?},\"prefill_ms\":{},\"ttft_ms\":{},\"decode_tokens\":{},\"decode_ms\":{},\"total_ms\":{}}}",report.prompt_ids,report.generated_ids,stats.prefill_ms,stats.first_token_ms.map_or("null".into(),|v|v.to_string()),stats.decode_tokens,stats.decode_ms,stats.total_ms);
            }
            Ok(())
        }
        [command,path] | [command,path,_] if command=="chat" => {
            let max_new=args.get(2).map(|v|v.parse::<u32>().map_err(|_|"Limite de tokens invalide".to_string())).transpose()?.unwrap_or(128);
            println!("Chargement d’AdaForge…");
            let mut session=adaforge_application::open_chat(inspector,backend,std::path::Path::new(path),max_new).map_err(|e|e.to_string())?;
            chat_loop(&mut session,&mut std::io::stdin().lock(),&mut std::io::stdout().lock())
        }
        [command,path,text] | [command,path,text,_] if command=="generate" || command=="ask" => {
            let max_new=args.get(3).map(|v|v.parse::<u32>().map_err(|_|"Limite de tokens invalide".to_string())).transpose()?.unwrap_or(64);
            let report=if command=="ask" {adaforge_application::ask_text(inspector,backend,std::path::Path::new(path),text,max_new)}else{adaforge_application::generate_text(inspector,backend,std::path::Path::new(path),text,max_new)}.map_err(|e|e.to_string())?;
            println!("{} tokens de prompt ; {} tokens générés ; sélection greedy",report.prompt_ids.len(),report.generated_ids.len());
            println!("{} : {}",if command=="ask" {"Réponse"}else{"Complétion"},report.text);
            println!("Arrêt : {}",if report.stopped {"marqueur de fin du modèle"}else{"limite de tokens"});
            print_stats(report.stats.as_ref(),&mut std::io::stdout().lock())?;
            println!("{} Poids et caches réutilisés ; aucun recalcul CPU des couches.",if command=="ask" {"Un tour Gemma, sans outils ni pensée."}else{"Texte brut, sans template de conversation."});
            Ok(())
        }
        [command,path,text] if command=="logits-check" => {
            let report=adaforge_application::check_text_logits(inspector,backend,std::path::Path::new(path),text).map_err(|e|e.to_string())?;
            let h=report.stack.logits.as_ref().ok_or("Rapport logits absent")?;
            println!("Texte → 48 blocs GPU → RMSNorm finale → projection Q6_K → softcap");
            println!("IDs : {:?} ; dernière position {}",report.tokenization.ids,h.position);
            println!("48 blocs contrôlés face au CPU ; {} sorties de vocabulaire vérifiées",h.vocabulary);
            println!("Norme finale : erreur max {:.8} ; ratio {:.5}",h.norm_error,h.norm_ratio);
            println!("Logits bruts : erreur max {:.8} ; ratio {:.5}",h.raw_error,h.raw_ratio);
            println!("Logits après softcap : erreur max {:.8} ; ratio {:.5}",h.capped_error,h.capped_ratio);
            println!("Référence CPU depuis les embeddings ; atol=rtol=1e-4, limite ratio 1.");
            println!("Tête alimentée directement par la sortie GPU du bloc 47. Aucun token généré.");
            Ok(())
        }
        [command,path,text] | [command,path,text,_] if command=="text-stack-check" => {
            let depth=args.get(3).map(|v|v.parse::<u32>().map_err(|_|"Nombre de blocs invalide".to_string())).transpose()?.unwrap_or(48);
            let report=adaforge_application::check_text_stack(inspector,backend,std::path::Path::new(path),text,depth).map_err(|e|e.to_string())?;
            println!("Texte → tokens → embeddings GPU → {depth} blocs Gemma consécutifs depuis le bloc 0");
            println!("IDs : {:?} ; {} positions à partir de zéro",report.stack.token_ids,report.stack.token_ids.len());
            println!("Texte redécodé : {:?} ; aller-retour exact : {}",report.tokenization.decoded,report.tokenization.decoded==*text);
            for (index,stats) in report.stack.layers.iter().enumerate() {
                let g=&report.stack.geometry[index];
                println!("Bloc {index} : {} ; {} têtes Q / {} KV ; largeur {} ; poids K/V partagés {} ; {} paires RoPE atténuées",if g.window.is_some(){"local"}else{"global"},g.q_heads,g.kv_heads,g.head_width,g.shared_kv_weights,g.attenuated_pairs);
                println!("Bloc {index} : {} sorties vérifiées ; erreur max {:.8} ; ratio tolérance {:.5}",stats.checked_values,stats.max_absolute_error,stats.max_tolerance_ratio);
            }
            println!("Référence CPU indépendante depuis les embeddings ; atol=rtol=1e-4, limite ratio 1.");
            println!("Sorties des blocs lues directement sur GPU par les suivants ; cache KV distinct par bloc.");
            println!("{depth} blocs vérifiés ; normalisation finale, logits et génération restent à connecter.");
            Ok(())
        }
        [command,path,text] if command=="text-layer-check" => {
            let report=adaforge_application::check_text_layer(inspector,backend,std::path::Path::new(path),text).map_err(|e|e.to_string())?;
            let stats=report.layer.heads.qkv.ffn.as_ref().ok_or("Rapport couche absent")?;
            println!("Texte brut → tokenizer Gemma → embeddings Q6_K → premier bloc dense GPU");
            println!("IDs : {:?} ; positions {:?}",report.tokenization.ids,report.layer.positions);
            println!("Texte redécodé : {:?} ; aller-retour exact : {}",report.tokenization.decoded,report.tokenization.decoded==*text);
            println!("{} sorties vérifiées face au CPU ; largeur {}",stats.checked_values,stats.width);
            println!("Erreur absolue maximale : {:.8} ; ratio tolérance {:.5} (limite 1, atol=rtol=1e-4)",stats.max_absolute_error,stats.max_tolerance_ratio);
            println!("Embeddings et activations intermédiaires sur GPU. Un bloc seulement ; pas encore de génération de texte.");
            Ok(())
        }
        [command,path,text,flags @ ..] if command=="tokenize" => {
            if flags.iter().any(|f|f!="--no-bos" && f!="--parse-special") {return Err("Options : --no-bos et --parse-special uniquement".into());}
            let options=adaforge_application::TokenizeOptions {add_bos:!flags.iter().any(|f|f=="--no-bos"),parse_special:flags.iter().any(|f|f=="--parse-special")};
            let report=adaforge_application::tokenize_text(inspector,std::path::Path::new(path),text,options).map_err(|e|e.to_string())?;
            println!("Tokenizer Gemma 4 : {} tokens de vocabulaire ; BOS {} ; analyse contrôles {}",report.vocabulary,options.add_bos,options.parse_special);
            println!("IDs : {:?}",report.ids);
            println!("Texte décodé hors BOS automatique : {:?}",report.decoded);
            println!("Aller-retour exact : {}",report.decoded==*text);
            Ok(())
        }
        _ => Err("Commande inconnue. Utiliser : adaforge gpu-check, adaforge model-info <fichier.gguf> ou adaforge --help".into()),
    }
}

fn print_help() {
    println!(
        "AdaForge — moteur RTX 4090 en développement\n\nUsage : adaforge gpu-check\n        adaforge model-info <fichier.gguf>\n        adaforge matvec-check <fichier.gguf> [tenseur]\n        adaforge rmsnorm-check <fichier.gguf> [tenseur]\n        adaforge projection-check <fichier.gguf>\n        adaforge qkv-check <fichier.gguf> [couche]\n        adaforge qknorm-check <fichier.gguf> [couche]\n        adaforge rope-check <fichier.gguf> [couche]\n        adaforge vnorm-check <fichier.gguf> [couche]\n        adaforge kv-cache-check <fichier.gguf> [couche]\n        adaforge kv-chain-check <fichier.gguf> [couche]\n        adaforge attention-check <fichier.gguf> [couche]\n        adaforge attention-branch-check <fichier.gguf> [couche]\n        adaforge layer-check <fichier.gguf> [couche]\n        adaforge embedding-check <fichier.gguf> [ids...]\n        adaforge embedding-layer-check <fichier.gguf>\n        adaforge bench <fichier.gguf> <message> [tokens=24] [mesures=5]\n        adaforge chat <fichier.gguf> [tokens par réponse: 1..128, défaut 128]\n        adaforge ask <fichier.gguf> <message> [nouveaux tokens: 1..128, défaut 64]\n        adaforge generate <fichier.gguf> <texte> [nouveaux tokens: 1..128, défaut 64]\n        adaforge logits-check <fichier.gguf> <texte>\n        adaforge text-stack-check <fichier.gguf> <texte> [blocs: 1..48, défaut 48]\n        adaforge text-layer-check <fichier.gguf> <texte>\n        adaforge tokenize <fichier.gguf> <texte> [--no-bos] [--parse-special]\n        adaforge --help"
    );
}

fn print_stats(
    stats: Option<&adaforge_application::GenerationStats>,
    output: &mut impl std::io::Write,
) -> Result<(), String> {
    if let Some(s) = stats {
        let rate = if s.decode_tokens > 0 && s.decode_ms > 0.0 {
            format!("{:.2} tok/s", s.decode_tokens as f64 * 1000.0 / s.decode_ms)
        } else {
            "non mesurable (moins de deux tokens)".into()
        };
        let ttft = s
            .first_token_ms
            .map_or("aucun token".into(), |v| format!("{v:.1} ms"));
        writeln!(output,"[Mesures : premier token {ttft} ; génération {rate} ; {} nouveaux tokens de prompt, {} en cache ; hors chargement]",s.prompt_tokens,s.cached_tokens).map_err(|e|e.to_string())?;
    }
    Ok(())
}

fn chat_loop(
    session: &mut adaforge_application::ChatSession,
    input: &mut impl std::io::BufRead,
    output: &mut impl std::io::Write,
) -> Result<(), String> {
    writeln!(output,"Chat prêt. /reset : nouvel historique ; /tokens : budget ; /quit : quitter ; /help : aide.\nContexte : {} tokens, sans troncature automatique. Réponses : {} tokens maximum.",adaforge_application::CHAT_CONTEXT,session.max_new_tokens()).map_err(|e|e.to_string())?;
    loop {
        write!(output, "Vous > ").map_err(|e| e.to_string())?;
        output.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        if input.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            writeln!(output).map_err(|e| e.to_string())?;
            break;
        }
        match line.trim(){
            "/quit" | "/exit"=>break,
            "/reset"=>{session.reset().map_err(|e|e.to_string())?;writeln!(output,"Historique effacé. Modèle conservé en mémoire.").map_err(|e|e.to_string())?;},
            "/help"=>writeln!(output,"Un message par ligne. /tokens [1..128] règle la longueur ; /reset efface l’historique ; /quit quitte.").map_err(|e|e.to_string())?,
            "/tokens"=>writeln!(output,"Budget : {} tokens maximum par réponse.",session.max_new_tokens()).map_err(|e|e.to_string())?,
            command if command.starts_with("/tokens ")=>{
                let result=command[8..].trim().parse::<u32>().map_err(|_|"Budget invalide : choisir de 1 à 128 tokens".to_string()).and_then(|n|session.set_max_new_tokens(n).map_err(|e|e.to_string()));
                match result {Ok(())=>writeln!(output,"Budget : {} tokens maximum par réponse.",session.max_new_tokens()),Err(e)=>writeln!(output,"Erreur : {e}")}.map_err(|e|e.to_string())?;
            },
            ""=>{},
            message=>{
                let mut started=false;let mut io_error=None;
                let result=session.send_stream(message,&mut |piece|{
                    if io_error.is_some(){return;}
                    let mut write=||->std::io::Result<()>{
                        if !started{write!(output,"\nAssistant : ")?;started=true;}
                        write!(output,"{piece}")?;output.flush()
                    };
                    if let Err(e)=write(){io_error=Some(e.to_string());}
                });
                if let Some(error)=io_error{return Err(error);}
                match result {
                    Ok(report)=>{if !started{write!(output,"\nAssistant : ").map_err(|e|e.to_string())?;}writeln!(output).map_err(|e|e.to_string())?;print_stats(report.stats.as_ref(),output)?;if !report.stopped{writeln!(output,"[Limite de tokens atteinte{}]",if session.max_new_tokens()<128 {" ; /tokens 128 pour augmenter le budget"}else{""}).map_err(|e|e.to_string())?;}},
                    Err(e)=>{if started{writeln!(output).map_err(|e|e.to_string())?;}writeln!(output,"Erreur : {e}").map_err(|e|e.to_string())?;},
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adaforge_application::{DiagnosticError, GpuCheckReport};
    struct NoModels;
    impl adaforge_application::ModelInspector for NoModels {
        fn inspect(
            &self,
            _: &std::path::Path,
        ) -> Result<adaforge_application::ModelInfo, DiagnosticError> {
            panic!("Pas de lecture de modèle attendue")
        }
    }
    struct Unavailable;
    impl GpuDiagnostics for Unavailable {
        fn check(&self) -> Result<GpuCheckReport, DiagnosticError> {
            Err(DiagnosticError("GPU indisponible".into()))
        }
    }
    #[test]
    fn help_and_bad_arguments_do_not_access_gpu() {
        struct MustNotRun;
        impl GpuDiagnostics for MustNotRun {
            fn check(&self) -> Result<GpuCheckReport, DiagnosticError> {
                panic!("Le backend ne doit pas être appelé");
            }
        }
        assert!(run(&[], &MustNotRun, &NoModels).is_ok());
        assert!(run(&["--help".into()], &MustNotRun, &NoModels).is_ok());
        assert!(run(&["unknown".into()], &MustNotRun, &NoModels).is_err());
        assert!(
            run(
                &["gpu-check".into(), "extra".into()],
                &MustNotRun,
                &NoModels
            )
            .is_err()
        );
    }
    #[test]
    fn diagnostic_errors_reach_the_caller() {
        assert_eq!(
            run(&["gpu-check".into()], &Unavailable, &NoModels).unwrap_err(),
            "GPU indisponible"
        );
    }
}
