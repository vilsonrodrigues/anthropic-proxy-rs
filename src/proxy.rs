use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use crate::transform;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, HeaderValue},
    middleware::Next,
    response::{IntoResponse, Response},
    Extension, Json,
};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::{header::AUTHORIZATION, Client, RequestBuilder};
use serde_json::json;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

const REQUEST_ID_HEADER: &str = "x-request-id";
const CORRELATION_ID_HEADER: &str = "x-correlation-id";
const TRACEPARENT_HEADER: &str = "traceparent";

#[derive(Clone, Debug)]
pub struct RequestContext {
    pub request_id: String,
    pub correlation_id: Option<String>,
    pub traceparent: Option<String>,
}

impl RequestContext {
    fn from_headers(headers: &HeaderMap) -> Self {
        let request_id =
            header_value(headers, REQUEST_ID_HEADER).unwrap_or_else(generate_request_id);

        Self {
            request_id,
            correlation_id: header_value(headers, CORRELATION_ID_HEADER),
            traceparent: header_value(headers, TRACEPARENT_HEADER),
        }
    }
}

pub async fn request_context_middleware(mut request: Request, next: Next) -> Response {
    let request_context = RequestContext::from_headers(request.headers());
    request.extensions_mut().insert(request_context.clone());

    let mut response = next.run(request).await;
    if let Ok(header_value) = HeaderValue::from_str(&request_context.request_id) {
        response
            .headers_mut()
            .insert(REQUEST_ID_HEADER, header_value);
    }

    response
}

pub async fn proxy_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    Extension(request_context): Extension<RequestContext>,
    headers: HeaderMap,
    Json(req): Json<anthropic::AnthropicRequest>,
) -> ProxyResult<Response> {
    let started_at = Instant::now();
    let is_streaming = req.stream.unwrap_or(false);
    let model_in = req.model.clone();

    tracing::info!(
        request_id = %request_context.request_id,
        route = "/v1/messages",
        model_in = %model_in,
        streaming = is_streaming,
        "Handling messages request"
    );

    if config.verbose {
        tracing::trace!(
            request_id = %request_context.request_id,
            "Incoming Anthropic request: {}",
            serde_json::to_string_pretty(&req).unwrap_or_default()
        );
    }

    let openai_req = transform::anthropic_to_openai(req, &config)?;
    let model_out = openai_req.model.clone();

    if config.verbose {
        tracing::trace!(
            request_id = %request_context.request_id,
            "Transformed OpenAI request: {}",
            serde_json::to_string_pretty(&openai_req).unwrap_or_default()
        );
    }

    let result = if is_streaming {
        handle_streaming(&request_context, &headers, config, client, openai_req).await
    } else {
        handle_non_streaming(&request_context, &headers, config, client, openai_req).await
    };

    match &result {
        Ok(response) => {
            tracing::info!(
                request_id = %request_context.request_id,
                route = "/v1/messages",
                model_in = %model_in,
                model_out = %model_out,
                streaming = is_streaming,
                status = response.status().as_u16(),
                latency_ms = started_at.elapsed().as_millis(),
                "Messages request completed"
            );
        }
        Err(err) => {
            tracing::warn!(
                request_id = %request_context.request_id,
                route = "/v1/messages",
                model_in = %model_in,
                model_out = %model_out,
                streaming = is_streaming,
                status = status_code_for_proxy_error(err).as_u16(),
                latency_ms = started_at.elapsed().as_millis(),
                error = %err,
                "Messages request failed"
            );
        }
    }

    result
}

async fn handle_non_streaming(
    request_context: &RequestContext,
    incoming_headers: &HeaderMap,
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
) -> ProxyResult<Response> {
    let url = config.chat_completions_url();
    tracing::debug!(
        request_id = %request_context.request_id,
        upstream_url = %url,
        model_out = %openai_req.model,
        "Sending non-streaming request upstream"
    );

    let req_builder = apply_upstream_headers(
        client.post(&url).json(&openai_req).timeout(Duration::from_secs(300)),
        &config,
        request_context,
        incoming_headers,
    );

    let response = req_builder.send().await.map_err(|err| {
        log_upstream_send_error(request_context, "/v1/messages", &url, Some(&openai_req.model), &err);
        ProxyError::Http(err)
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        log_upstream_error_response(request_context, "/v1/messages", &url, Some(&openai_req.model), status, &error_text);
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {}: {}",
            status, error_text
        )));
    }

    let openai_resp: openai::OpenAIResponse = response.json().await?;

    if config.verbose {
        tracing::trace!(
            request_id = %request_context.request_id,
            "Received OpenAI response: {}",
            serde_json::to_string_pretty(&openai_resp).unwrap_or_default()
        );
    }

    let anthropic_resp = transform::openai_to_anthropic(openai_resp, &openai_req.model)?;

    if config.verbose {
        tracing::trace!(
            request_id = %request_context.request_id,
            "Transformed Anthropic response: {}",
            serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default()
        );
    }

    Ok(Json(anthropic_resp).into_response())
}

async fn handle_streaming(
    request_context: &RequestContext,
    incoming_headers: &HeaderMap,
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
) -> ProxyResult<Response> {
    let url = config.chat_completions_url();
    tracing::debug!(
        request_id = %request_context.request_id,
        upstream_url = %url,
        model_out = %openai_req.model,
        "Sending streaming request upstream"
    );

    let req_builder = apply_upstream_headers(
        client.post(&url).json(&openai_req).timeout(Duration::from_secs(300)),
        &config,
        request_context,
        incoming_headers,
    );

    let response = req_builder.send().await.map_err(|err| {
        log_upstream_send_error(request_context, "/v1/messages", &url, Some(&openai_req.model), &err);
        ProxyError::Http(err)
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        log_upstream_error_response(request_context, "/v1/messages", &url, Some(&openai_req.model), status, &error_text);
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {} from {}: {}",
            status, url, error_text
        )));
    }

    let stream = response.bytes_stream();
    let sse_stream = create_sse_stream(stream, openai_req.model.clone());

    let mut headers = HeaderMap::new();
    headers.insert(
        "Content-Type",
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));

    Ok((headers, Body::from_stream(sse_stream)).into_response())
}

fn create_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    fallback_model: String,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut message_id = None;
        let mut current_model = None;
        let mut content_index = 0;
        let mut tool_call_id = None;
        let mut _tool_call_name = None;
        let mut tool_call_args = String::new();
        let mut has_sent_message_start = false;
        let mut current_block_type: Option<String> = None;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(pos) = buffer.find("\n\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        for l in line.lines() {
                            if let Some(data) = l.strip_prefix("data: ") {
                                if data.trim() == "[DONE]" {
                                    let event = json!({"type": "message_stop"});
                                    let sse_data = format!("event: message_stop\ndata: {}\n\n",
                                        serde_json::to_string(&event).unwrap_or_default());
                                    yield Ok(Bytes::from(sse_data));
                                    continue;
                                }

                                if let Ok(chunk) = serde_json::from_str::<openai::StreamChunk>(data) {
                                    if message_id.is_none() {
                                        if let Some(id) = &chunk.id {
                                            message_id = Some(id.clone());
                                        }
                                    }
                                    if current_model.is_none() {
                                        if let Some(model) = &chunk.model {
                                            current_model = Some(model.clone());
                                        }
                                    }

                                    if let Some(choice) = chunk.choices.first() {

                                        if !has_sent_message_start {
                                            let event = anthropic::StreamEvent::MessageStart {
                                                message: anthropic::MessageStartData {
                                                    id: message_id.clone().unwrap_or_else(|| "msg_proxy".to_string()),
                                                    message_type: "message".to_string(),
                                                    role: "assistant".to_string(),
                                                    model: current_model.clone().unwrap_or_else(|| fallback_model.clone()),
                                                    usage: anthropic::Usage {
                                                        input_tokens: 0,
                                                        output_tokens: 0,
                                                    },
                                                },
                                            };
                                            let sse_data = format!("event: message_start\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                            has_sent_message_start = true;
                                        }

                                        if let Some(reasoning) = &choice.delta.reasoning {
                                            if current_block_type.is_none() {
                                                let event = json!({
                                                    "type": "content_block_start",
                                                    "index": content_index,
                                                    "content_block": {
                                                        "type": "thinking",
                                                        "thinking": ""
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                                current_block_type = Some("thinking".to_string());
                                            }

                                            let event = json!({
                                                "type": "content_block_delta",
                                                "index": content_index,
                                                "delta": {
                                                    "type": "thinking_delta",
                                                    "thinking": reasoning
                                                }
                                            });
                                            let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                        }

                                        if let Some(content) = &choice.delta.content {
                                            if !content.is_empty() {
                                                if current_block_type.as_deref() != Some("text") {
                                                    if current_block_type.is_some() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": content_index
                                                        });
                                                        let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        content_index += 1;
                                                    }

                                                    // Start text block
                                                    let event = json!({
                                                        "type": "content_block_start",
                                                        "index": content_index,
                                                        "content_block": {
                                                            "type": "text",
                                                            "text": ""
                                                        }
                                                    });
                                                    let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                    current_block_type = Some("text".to_string());
                                                }

                                                // Send text delta
                                                let event = json!({
                                                    "type": "content_block_delta",
                                                    "index": content_index,
                                                    "delta": {
                                                        "type": "text_delta",
                                                        "text": content
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }
                                        }

                                        // Handle tool calls
                                        if let Some(tool_calls) = &choice.delta.tool_calls {
                                            for tool_call in tool_calls {
                                                if let Some(id) = &tool_call.id {
                                                    // Start of new tool call
                                                    if current_block_type.is_some() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": content_index
                                                        });
                                                        let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        content_index += 1;
                                                    }

                                                    tool_call_id = Some(id.clone());
                                                    tool_call_args.clear();
                                                }

                                                if let Some(function) = &tool_call.function {
                                                    if let Some(name) = &function.name {
                                                        _tool_call_name = Some(name.clone());

                                                        // Start tool_use block
                                                        let event = json!({
                                                            "type": "content_block_start",
                                                            "index": content_index,
                                                            "content_block": {
                                                                "type": "tool_use",
                                                                "id": tool_call_id.clone().unwrap_or_default(),
                                                                "name": name
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        current_block_type = Some("tool_use".to_string());
                                                    }

                                                    if let Some(args) = &function.arguments {
                                                        tool_call_args.push_str(args);

                                                        // Send input_json_delta
                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": content_index,
                                                            "delta": {
                                                                "type": "input_json_delta",
                                                                "partial_json": args
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                    }
                                                }
                                            }
                                        }

                                        // Handle finish reason
                                        if let Some(finish_reason) = &choice.finish_reason {
                                            // Close current content block
                                            if current_block_type.is_some() {
                                                let event = json!({
                                                    "type": "content_block_stop",
                                                    "index": content_index
                                                });
                                                let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }

                                            // Send message_delta with stop_reason
                                            let stop_reason = transform::map_stop_reason(Some(finish_reason));
                                            let event = json!({
                                                "type": "message_delta",
                                                "delta": {
                                                    "stop_reason": stop_reason,
                                                    "stop_sequence": serde_json::Value::Null
                                                },
                                                "usage": chunk.usage.as_ref().map(|u| json!({
                                                    "output_tokens": u.completion_tokens
                                                }))
                                            });
                                            let sse_data = format!("event: message_delta\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                        }
                                    }
                                } else {
                                    tracing::debug!("Ignoring unrecognized upstream stream chunk: {}", data);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    let error_event = json!({
                        "type": "error",
                        "error": {
                            "type": "stream_error",
                            "message": format!("Stream error: {}", e)
                        }
                    });
                    let sse_data = format!("event: error\ndata: {}\n\n",
                        serde_json::to_string(&error_event).unwrap_or_default());
                    yield Ok(Bytes::from(sse_data));
                    break;
                }
            }
        }
    }
}

fn apply_upstream_headers(
    mut req_builder: RequestBuilder,
    config: &Config,
    request_context: &RequestContext,
    _incoming_headers: &HeaderMap,
) -> RequestBuilder {
    if !config.upstream_headers.is_empty() {
        req_builder = req_builder.headers(config.upstream_headers.clone());
    }

    if !config.upstream_headers.contains_key(REQUEST_ID_HEADER) {
        req_builder = req_builder.header(REQUEST_ID_HEADER, request_context.request_id.as_str());
    }
    if let Some(correlation_id) = &request_context.correlation_id {
        if !config.upstream_headers.contains_key(CORRELATION_ID_HEADER) {
            req_builder = req_builder.header(CORRELATION_ID_HEADER, correlation_id.as_str());
        }
    }
    if let Some(traceparent) = &request_context.traceparent {
        if !config.upstream_headers.contains_key(TRACEPARENT_HEADER) {
            req_builder = req_builder.header(TRACEPARENT_HEADER, traceparent.as_str());
        }
    }

    if let Some(api_key) = &config.api_key {
        if !config.upstream_headers.contains_key(AUTHORIZATION) {
            req_builder = req_builder.header(AUTHORIZATION, format!("Bearer {}", api_key));
        }
    }

    tracing::debug!(
        request_id = %request_context.request_id,
        correlation_id = request_context.correlation_id.as_deref().unwrap_or("n/a"),
        traceparent = request_context.traceparent.as_deref().unwrap_or("n/a"),
        "Applied upstream tracing headers"
    );

    req_builder
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn generate_request_id() -> String {
    let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("req_{millis}_{counter}")
}

fn classify_reqwest_error(err: &reqwest::Error) -> &'static str {
    let message = err.to_string().to_ascii_lowercase();

    if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connect"
    } else if message.contains("certificate") || message.contains("tls") {
        "tls"
    } else if err.is_body() {
        "body"
    } else if err.is_request() {
        "request"
    } else {
        "http"
    }
}

fn log_upstream_send_error(
    request_context: &RequestContext,
    route: &str,
    upstream_url: &str,
    model_out: Option<&str>,
    err: &reqwest::Error,
) {
    tracing::error!(
        request_id = %request_context.request_id,
        route = route,
        upstream_url = upstream_url,
        model_out = model_out.unwrap_or("n/a"),
        error_type = classify_reqwest_error(err),
        error = %err,
        "Failed to reach upstream"
    );
}

fn log_upstream_error_response(
    request_context: &RequestContext,
    route: &str,
    upstream_url: &str,
    model_out: Option<&str>,
    status: reqwest::StatusCode,
    body: &str,
) {
    tracing::error!(
        request_id = %request_context.request_id,
        route = route,
        upstream_url = upstream_url,
        model_out = model_out.unwrap_or("n/a"),
        upstream_status = status.as_u16(),
        response_body = body,
        "Upstream returned an error response"
    );
}

fn status_code_for_proxy_error(err: &ProxyError) -> axum::http::StatusCode {
    match err {
        ProxyError::Transform(_) | ProxyError::Serialization(_) => {
            axum::http::StatusCode::BAD_REQUEST
        }
        ProxyError::Upstream(_) | ProxyError::Http(_) => axum::http::StatusCode::BAD_GATEWAY,
        ProxyError::Config(_) | ProxyError::Internal(_) => {
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_upstream_headers, RequestContext, CORRELATION_ID_HEADER, REQUEST_ID_HEADER,
        TRACEPARENT_HEADER,
    };
    use crate::config::Config;
    use reqwest::{header::HeaderMap, Client};

    fn test_config() -> Config {
        Config {
            port: 3000,
            base_url: "https://example.com/v1".to_string(),
            api_key: Some("sk-test".to_string()),
            upstream_headers: HeaderMap::new(),
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
        }
    }

    #[test]
    fn request_context_keeps_request_and_correlation_distinct() {
        let mut headers = HeaderMap::new();
        headers.insert(REQUEST_ID_HEADER, "req-client-123".parse().unwrap());
        headers.insert(CORRELATION_ID_HEADER, "corr-client-456".parse().unwrap());
        headers.insert(TRACEPARENT_HEADER, "00-abc-def-01".parse().unwrap());

        let context = RequestContext::from_headers(&headers);

        assert_eq!(context.request_id, "req-client-123");
        assert_eq!(context.correlation_id.as_deref(), Some("corr-client-456"));
        assert_eq!(context.traceparent.as_deref(), Some("00-abc-def-01"));
    }

    #[test]
    fn request_context_generates_request_id_when_only_correlation_is_present() {
        let mut headers = HeaderMap::new();
        headers.insert(CORRELATION_ID_HEADER, "corr-client-456".parse().unwrap());

        let context = RequestContext::from_headers(&headers);

        assert_ne!(context.request_id, "corr-client-456");
        assert!(context.request_id.starts_with("req_"));
        assert_eq!(context.correlation_id.as_deref(), Some("corr-client-456"));
    }

    #[test]
    fn apply_upstream_headers_forwards_request_and_correlation_headers() {
        let client = Client::new();
        let config = test_config();
        let request_context = RequestContext {
            request_id: "req-proxy-123".to_string(),
            correlation_id: Some("corr-client-456".to_string()),
            traceparent: Some("00-abc-def-01".to_string()),
        };

        let request = apply_upstream_headers(
            client.post("https://example.com/v1/chat/completions"),
            &config,
            &request_context,
            &HeaderMap::new(),
        )
        .build()
        .unwrap();

        assert_eq!(
            request.headers().get(REQUEST_ID_HEADER).unwrap(),
            "req-proxy-123"
        );
        assert_eq!(
            request.headers().get(CORRELATION_ID_HEADER).unwrap(),
            "corr-client-456"
        );
        assert_eq!(
            request.headers().get(TRACEPARENT_HEADER).unwrap(),
            "00-abc-def-01"
        );
        assert_eq!(
            request
                .headers()
                .get(reqwest::header::AUTHORIZATION)
                .unwrap(),
            "Bearer sk-test"
        );
    }

    #[test]
    fn apply_upstream_headers_does_not_invent_correlation_id() {
        let client = Client::new();
        let config = test_config();
        let request_context = RequestContext {
            request_id: "req-proxy-123".to_string(),
            correlation_id: None,
            traceparent: None,
        };

        let request = apply_upstream_headers(
            client.post("https://example.com/v1/chat/completions"),
            &config,
            &request_context,
            &HeaderMap::new(),
        )
        .build()
        .unwrap();

        assert_eq!(
            request.headers().get(REQUEST_ID_HEADER).unwrap(),
            "req-proxy-123"
        );
        assert!(request.headers().get(CORRELATION_ID_HEADER).is_none());
    }
}
