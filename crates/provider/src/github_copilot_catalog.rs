use std::collections::HashMap;

use crate::tura_llm::{
    CatalogModelConfig, CatalogModelDetail, ModelCatalog, ProviderCatalogConfig, RootConfig,
};

const PROVIDER_ID: &str = "github-copilot";
const SDK_BASE_URL: &str = "sdk://github-copilot";

/// Upgrade the existing `github-copilot` auth-registry entry into a callable
/// model provider backed by the official SDK.
///
/// The auth registry already maps `github-copilot` to the same runtime id. This
/// overlay supplies the SDK transport, a selectable automatic model, and an
/// empty credential sentinel so the central provider dispatcher can reach the
/// adapter when the SDK should discover a stored Copilot/gh login instead.
pub(crate) fn apply(config: &mut RootConfig) {
    config
        .provider_base_url
        .insert(PROVIDER_ID.to_string(), SDK_BASE_URL.to_string());
    ensure_tier(&mut config.model_catalog, "thinking");
    ensure_tier(&mut config.model_catalog, "fast");
    ensure_enum(&mut config.provider_enums.api_styles, "copilot_sdk");
    ensure_enum(&mut config.provider_enums.auth_methods, "local_cli_token");
    ensure_enum(&mut config.provider_enums.capabilities, "llm.streaming");

    let auto = CatalogModelConfig::Detailed(CatalogModelDetail {
        id: "auto".to_string(),
        visible: true,
        name: "Copilot automatic model".to_string(),
        family: "github-copilot".to_string(),
        reasoning: true,
        temperature: false,
        tool_call: true,
        ..CatalogModelDetail::default()
    });
    config.model_catalog.providers.insert(
        PROVIDER_ID.to_string(),
        ProviderCatalogConfig {
            display_name: "GitHub Copilot".to_string(),
            runtime_provider: PROVIDER_ID.to_string(),
            api_style: "copilot_sdk".to_string(),
            base_url: SDK_BASE_URL.to_string(),
            token_env: Some("COPILOT_GITHUB_TOKEN".to_string()),
            env: vec![
                "COPILOT_GITHUB_TOKEN".to_string(),
                "GH_TOKEN".to_string(),
                "GITHUB_TOKEN".to_string(),
                "COPILOT_CLI_PATH".to_string(),
            ],
            domains: vec!["llm".to_string()],
            capabilities: vec![
                "llm.chat".to_string(),
                "llm.tool_call".to_string(),
                "llm.streaming".to_string(),
            ],
            auth_methods: vec!["local_cli_token".to_string(), "api_key".to_string()],
            api_docs: Some("https://github.com/github/copilot-sdk".to_string()),
            status: Some("configured".to_string()),
            models: HashMap::from([
                ("thinking".to_string(), vec![auto.clone()]),
                ("fast".to_string(), vec![auto]),
            ]),
        },
    );

    if std::env::var_os("COPILOT_GITHUB_TOKEN").is_none() {
        std::env::set_var("COPILOT_GITHUB_TOKEN", "");
    }
}

fn ensure_tier(catalog: &mut ModelCatalog, tier: &str) {
    if !catalog.tiers.iter().any(|existing| existing == tier) {
        catalog.tiers.push(tier.to_string());
    }
}

fn ensure_enum(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::{apply, PROVIDER_ID, SDK_BASE_URL};
    use crate::tura_llm::RootConfig;

    #[test]
    fn overlay_makes_existing_auth_provider_callable_through_sdk_runtime() {
        let mut config: RootConfig = serde_json::from_value(serde_json::json!({
            "provider_base_url": {},
            "routes": {}
        }))
        .expect("minimal config");
        apply(&mut config);

        assert_eq!(
            crate::auth_registry::runtime_provider_id(PROVIDER_ID),
            PROVIDER_ID
        );
        assert_eq!(config.provider_base_url[PROVIDER_ID], SDK_BASE_URL);
        let provider = &config.model_catalog.providers[PROVIDER_ID];
        assert_eq!(provider.runtime_provider, PROVIDER_ID);
        assert_eq!(provider.api_style, "copilot_sdk");
        assert_eq!(provider.models["thinking"][0].id(), "auto");
    }
}
