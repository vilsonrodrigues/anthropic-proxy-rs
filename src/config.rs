use anyhow::{bail, Result};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Url,
};
use std::{env, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub base_url: String,
    pub api_key: Option<String>,
    pub upstream_headers: HeaderMap,
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub debug: bool,
    pub verbose: bool,
}

impl Config {
    fn load_dotenv(custom_path: Option<PathBuf>) -> Option<PathBuf> {
        if let Some(path) = custom_path {
            if path.exists() {
                if let Ok(_) = dotenvy::from_path(&path) {
                    return Some(path);
                }
            }
            eprintln!(
                "⚠️  WARNING: Custom config file not found: {}",
                path.display()
            );
        }

        if let Ok(path) = dotenvy::dotenv() {
            return Some(path);
        }

        if let Some(home) = env::var("HOME").ok() {
            let home_config = PathBuf::from(home).join(".anthropic-proxy.env");
            if home_config.exists() {
                if let Ok(_) = dotenvy::from_path(&home_config) {
                    return Some(home_config);
                }
            }
        }

        let etc_config = PathBuf::from("/etc/anthropic-proxy/.env");
        if etc_config.exists() {
            if let Ok(_) = dotenvy::from_path(&etc_config) {
                return Some(etc_config);
            }
        }

        None
    }

    pub fn from_env() -> Result<Self> {
        Self::from_env_with_path(None)
    }

    pub fn from_env_with_path(custom_path: Option<PathBuf>) -> Result<Self> {
        if let Some(path) = Self::load_dotenv(custom_path) {
            eprintln!("📄 Loaded config from: {}", path.display());
        } else {
            eprintln!("ℹ️  No .env file found, using environment variables only");
        }

        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        let base_url = env::var("UPSTREAM_BASE_URL")
            .or_else(|_| env::var("ANTHROPIC_PROXY_BASE_URL"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "UPSTREAM_BASE_URL is required. Set it to your OpenAI-compatible endpoint.\n\
                Examples:\n\
                  - OpenRouter: https://openrouter.ai/api\n\
                  - OpenAI: https://api.openai.com\n\
                  - Versioned gateway: https://gateway.example.com/v2\n\
                  - Local: http://localhost:11434"
                )
            })?;

        Self::validate_base_url(&base_url)?;

        let api_key = env::var("UPSTREAM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());

        let upstream_headers = env::var("ANTHROPIC_PROXY_UPSTREAM_HEADERS")
            .ok()
            .map(|value| Self::parse_upstream_headers(&value))
            .transpose()?
            .unwrap_or_default();

        let reasoning_model = env::var("REASONING_MODEL").ok();
        let completion_model = env::var("COMPLETION_MODEL").ok();

        let debug = env::var("DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let verbose = env::var("VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Ok(Config {
            port,
            base_url,
            api_key,
            upstream_headers,
            reasoning_model,
            completion_model,
            debug,
            verbose,
        })
    }

    pub fn chat_completions_url(&self) -> String {
        Self::resolve_chat_completions_url(&self.base_url)
            .expect("UPSTREAM_BASE_URL should be validated during configuration loading")
    }

    fn validate_base_url(base_url: &str) -> Result<()> {
        Self::resolve_chat_completions_url(base_url).map(|_| ())
    }

    fn resolve_chat_completions_url(base_url: &str) -> Result<String> {
        let normalized = base_url.trim();

        if normalized.is_empty() {
            bail!("UPSTREAM_BASE_URL must not be empty");
        }

        let parsed = Url::parse(normalized).map_err(|err| {
            anyhow::anyhow!("UPSTREAM_BASE_URL must be a valid http(s) URL: {}", err)
        })?;

        if !matches!(parsed.scheme(), "http" | "https") {
            bail!("UPSTREAM_BASE_URL must use http or https");
        }

        if parsed.query().is_some() || parsed.fragment().is_some() {
            bail!("UPSTREAM_BASE_URL must not include query parameters or fragments");
        }

        let path_segments: Vec<_> = parsed
            .path_segments()
            .map(|segments| segments.filter(|segment| !segment.is_empty()).collect())
            .unwrap_or_default();

        if Self::is_chat_completions_path(&path_segments) {
            return Ok(normalized.trim_end_matches('/').to_string());
        }

        let last_segment = path_segments.last().copied();
        if matches!(last_segment, Some("chat") | Some("completions")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../chat/completions endpoint"
            );
        }

        let normalized = normalized.trim_end_matches('/');
        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/chat/completions", normalized));
        }

        Ok(format!("{}/v1/chat/completions", normalized))
    }

    fn is_chat_completions_path(segments: &[&str]) -> bool {
        matches!(segments, [.., "chat", "completions"])
    }

    fn is_version_segment(segment: &str) -> bool {
        let version = segment
            .strip_prefix('v')
            .or_else(|| segment.strip_prefix('V'));

        version
            .is_some_and(|value| !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit()))
    }

    pub fn parse_upstream_headers(value: &str) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();

        for entry in value
            .split(|ch| ch == ';' || ch == '\n')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            let (name, raw_value) = entry
                .split_once('=')
                .or_else(|| entry.split_once(':'))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Invalid ANTHROPIC_PROXY_UPSTREAM_HEADERS entry '{}'. Expected Header=Value",
                        entry
                    )
                })?;

            let name = name.trim();
            let value = raw_value.trim();

            if name.is_empty() || value.is_empty() {
                bail!(
                    "Invalid ANTHROPIC_PROXY_UPSTREAM_HEADERS entry '{}'. Header name and value must be non-empty",
                    entry
                );
            }

            let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|err| {
                anyhow::anyhow!("Invalid upstream header name '{}': {}", name, err)
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|err| {
                anyhow::anyhow!(
                    "Invalid upstream header value for '{}': {}",
                    header_name.as_str(),
                    err
                )
            })?;

            headers.insert(header_name, header_value);
        }

        Ok(headers)
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use reqwest::header::AUTHORIZATION;

    #[test]
    fn base_url_without_version_defaults_to_v1_endpoint() {
        let url = Config::resolve_chat_completions_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn versioned_base_url_preserves_existing_version() {
        let url = Config::resolve_chat_completions_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn full_chat_completions_endpoint_is_used_as_is() {
        let url = Config::resolve_chat_completions_url(
            "https://gateway.example.com/v2/chat/completions/",
        )
        .unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn partial_chat_path_is_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2/chat")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn query_strings_are_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2?foo=bar")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn parse_upstream_headers_supports_semicolons_newlines_and_colons() {
        let headers = Config::parse_upstream_headers(
            "x-tenant=asimov;authorization: Bearer test\nx-trace-id=req-1",
        )
        .unwrap();

        assert_eq!(headers.get("x-tenant").unwrap(), "asimov");
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer test");
        assert_eq!(headers.get("x-trace-id").unwrap(), "req-1");
    }

    #[test]
    fn parse_upstream_headers_rejects_invalid_entries() {
        let err = Config::parse_upstream_headers("missing-separator").unwrap_err();

        assert!(err.to_string().contains("Expected Header=Value"));
    }
}
