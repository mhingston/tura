//! GitHub Copilot provider backed by the official Copilot SDK.
//!
//! Tura remains the orchestration/runtime boundary. The SDK is started in
//! `ClientMode::Empty`, config discovery is disabled, and only Tura's canonical
//! tool declarations are exposed as custom SDK tools.

use std::ffi::OsString;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use github_copilot_sdk::handler::DenyAllHandler;
use github_copilot_sdk::tool::ToolHandler;
use github_copilot_sdk::{
    Client, ClientMode, ClientOptions, MessageOptions, SessionConfig, SystemMessageConfig, Tool,
    ToolInvocation, ToolResult, ToolSet,
};
use serde_json::{json, Value};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::tura_llm::{
    project_root, provider_latency_timeouts, CallMetrics, CallOptions, ProviderResponse,
    ProviderStreamEvent, ProviderStreamEventSink, TuraError,
};
use crate::utils::emit_command_run_stream_events_from_content;

const PROVIDER_ID: &str = "github-copilot";
const AUTO_MODELS: &[&str] = &["auto", "default", "copilot"];

#[derive(Debug, Clone)]
struct CapturedToolCall {
    id: String,
    name: String,
    arguments: Value,
}

struct CaptureToolHandler {
    name: String,
    captured: Arc<Mutex<Vec<CapturedToolCall>>>,
}

#[async_trait]
impl ToolHandler for CaptureToolHandler {
    async fn call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<ToolResult, github_copilot_sdk::Error> {
        let call = CapturedToolCall {
            id: Uuid::new_v4().simple().to_string(),
            name: self.name.clone(),
            arguments: invocation.arguments,
        };
        if let Ok(mut captured) = self.captured.lock() {
            captured.push(call);
        }
        Ok(ToolResult::Text(
            "The host application captured this tool request. Stop this turn and wait for the host to execute it."
                .to_string(),
        ))
    }
}

pub async fn call_with_stream_events(
    _base_url: &str,
    model: &str,
    access_token: &str,
    messages: &[Value],
    options: &CallOptions,
    stream_events: Option<ProviderStreamEventSink>,
) -> Result<ProviderResponse, TuraError> {
    let mut client_options = ClientOptions::default();
    client_options.mode = ClientMode::Empty;
    client_options.working_directory = project_root();
    client_options.use_logged_in_user = Some(access_token.trim().is_empty());
    if access_token.trim().is_empty() {
        // Tura uses an empty internal credential sentinel to reach this adapter
        // when the SDK should discover an existing Copilot/gh login.
        client_options
            .env_remove
            .push(OsString::from("COPILOT_GITHUB_TOKEN"));
    } else {
        client_options.github_token = Some(access_token.to_string());
    }

    let client = Client::start(client_options)
        .await
        .map_err(|error| sdk_error("start Copilot SDK client", error))?;

    let captured = Arc::new(Mutex::new(Vec::<CapturedToolCall>::new()));
    let (tools, available_tools) = sdk_tools(options.tools.as_deref(), captured.clone())?;
    let mut session_config = SessionConfig::default()
        .with_client_name("tura")
        .with_streaming(stream_events.is_some() || options.stream.unwrap_or(false))
        .with_permission_handler(Arc::new(DenyAllHandler));
    session_config.enable_config_discovery = Some(false);
    session_config.enable_host_git_operations = Some(false);
    session_config.enable_skills = Some(false);
    session_config.enable_session_store = Some(false);
    session_config.skip_embedding_retrieval = Some(true);
    session_config.system_message = Some(
        SystemMessageConfig::new()
            .with_mode("customize")
            .with_content(system_message(messages, options.tools.as_deref())),
    );
    if !AUTO_MODELS
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(model.trim()))
    {
        session_config.model = Some(model.to_string());
    }
    if let Some(reasoning_effort) = options
        .reasoning_effort
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        session_config.reasoning_effort = Some(reasoning_effort.to_string());
    }
    if !tools.is_empty() {
        session_config = session_config
            .with_tools(tools)
            .with_available_tools(available_tools);
    }

    let session = match client.create_session(session_config).await {
        Ok(session) => session,
        Err(error) => {
            let _ = client.stop().await;
            return Err(sdk_error("create Copilot SDK session", error));
        }
    };

    let stream_task = stream_events
        .clone()
        .map(|sink| spawn_stream_forwarder(session.subscribe(), sink));
    let timeout = Duration::from_millis(provider_latency_timeouts().total_timeout_ms.max(1));
    let prompt = canonical_prompt(messages)?;
    let result = session
        .send_and_wait(MessageOptions::new(prompt).with_wait_timeout(timeout))
        .await;

    if let Some(task) = stream_task {
        let _ = task.await;
    }
    let _ = session.disconnect().await;
    let _ = client.stop().await;

    let assistant_event = result.map_err(|error| sdk_error("send Copilot SDK prompt", error))?;
    let captured = captured
        .lock()
        .map(|calls| calls.clone())
        .unwrap_or_default();
    let assistant_text = assistant_event
        .as_ref()
        .and_then(|event| event.data.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let (content, finish_reason) = if captured.is_empty() {
        (Value::String(assistant_text.clone()), "stop")
    } else {
        let tool_calls = captured
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut object = serde_json::Map::new();
        if !assistant_text.trim().is_empty() {
            object.insert("text".to_string(), Value::String(assistant_text.clone()));
        }
        object.insert("tool_calls".to_string(), Value::Array(tool_calls));
        (Value::Object(object), "tool_calls")
    };

    emit_command_run_stream_events_from_content(&content, stream_events.as_ref());
    let raw = json!({
        "provider": PROVIDER_ID,
        "session_id": session.id().as_str(),
        "assistant_event": assistant_event.map(|event| event.data),
        "captured_tool_calls": captured.iter().map(|call| json!({
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        })).collect::<Vec<_>>(),
    });
    let metrics = CallMetrics {
        tool_call_count: captured.len(),
        finish_reason: Some(finish_reason.to_string()),
        ..CallMetrics::default()
    };

    Ok(ProviderResponse {
        content,
        raw,
        metrics: Some(metrics),
    })
}

fn sdk_tools(
    configured_tools: Option<&[Value]>,
    captured: Arc<Mutex<Vec<CapturedToolCall>>>,
) -> Result<(Vec<Tool>, Vec<String>), TuraError> {
    let mut tools = Vec::new();
    let mut available = ToolSet::new();
    for configured in configured_tools.unwrap_or_default() {
        let function = configured.get("function").unwrap_or(configured);
        let Some(name) = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            continue;
        };
        let description = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parameters = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
        if !parameters.is_object() {
            return Err(TuraError::Validation {
                message: format!("GitHub Copilot tool '{name}' parameters must be a JSON object"),
            });
        }
        let handler = Arc::new(CaptureToolHandler {
            name: name.to_string(),
            captured: captured.clone(),
        });
        tools.push(
            Tool::new(name)
                .with_description(description)
                .with_parameters(parameters)
                .with_skip_permission(true)
                .with_handler(handler),
        );
        available = available
            .add_custom(name)
            .map_err(|error| sdk_error("register Copilot SDK tool", error))?;
    }
    Ok((tools, available.into_vec()))
}

fn system_message(messages: &[Value], tools: Option<&[Value]>) -> String {
    let mut system_parts = messages
        .iter()
        .filter(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .filter_map(|message| message.get("content"))
        .map(value_text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>();
    system_parts.push(
        "You are the model provider inside Tura. Tura owns orchestration, permissions, filesystem access, command execution, and tool execution. Do not use ambient Copilot CLI tools, config discovery, skills, agents, MCP servers, host Git operations, or host filesystem access. Only call custom tools explicitly supplied by Tura. When you call one or more supplied tools, stop the turn after the host-capture result and do not continue with a synthesized answer."
            .to_string(),
    );
    if tools.unwrap_or_default().is_empty() {
        system_parts
            .push("No tools are available in this turn; answer with text only.".to_string());
    }
    system_parts.join("\n\n")
}

fn canonical_prompt(messages: &[Value]) -> Result<String, TuraError> {
    let non_system = messages
        .iter()
        .filter(|message| {
            !matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .collect::<Vec<_>>();
    let transcript = serde_json::to_string_pretty(&non_system)?;
    Ok(format!(
        "Continue the canonical conversation below as the assistant. Preserve the meaning of prior assistant tool calls and tool results. Do not quote or explain the JSON transcript.\n\n{transcript}"
    ))
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

fn spawn_stream_forwarder(
    mut events: github_copilot_sdk::EventSubscription,
    sink: ProviderStreamEventSink,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut started = false;
        while let Ok(event) = events.recv().await {
            match event.event_type.as_str() {
                "assistant.message_delta" => {
                    let delta = event
                        .data
                        .get("delta")
                        .or_else(|| event.data.get("content"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if delta.is_empty() {
                        continue;
                    }
                    if !started {
                        started = true;
                        sink(ProviderStreamEvent::ProviderOutputStarted);
                    }
                    sink(ProviderStreamEvent::TextDelta {
                        text: delta.to_string(),
                    });
                }
                "session.idle" | "session.error" => break,
                _ => {}
            }
        }
    })
}

fn sdk_error(context: &str, error: github_copilot_sdk::Error) -> TuraError {
    TuraError::ProviderRequest {
        provider: PROVIDER_ID.to_string(),
        message: format!("{context}: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{canonical_prompt, sdk_tools, system_message};
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    #[test]
    fn canonical_prompt_excludes_system_messages_but_preserves_tool_results() {
        let prompt = canonical_prompt(&[
            json!({"role": "system", "content": "private system"}),
            json!({"role": "user", "content": "hello"}),
            json!({"role": "tool", "tool_call_id": "call-1", "content": "done"}),
        ])
        .expect("prompt");

        assert!(!prompt.contains("private system"));
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("tool_call_id"));
    }

    #[test]
    fn system_message_keeps_tura_as_the_runtime_boundary() {
        let message = system_message(
            &[json!({"role": "system", "content": "Be precise"})],
            Some(&[json!({"type": "function", "function": {"name": "command_run"}})]),
        );

        assert!(message.contains("Be precise"));
        assert!(message.contains("Tura owns orchestration"));
        assert!(message.contains("Only call custom tools"));
    }

    #[test]
    fn sdk_tools_translate_openai_function_shape() {
        let configured = vec![json!({
            "type": "function",
            "function": {
                "name": "command_run",
                "description": "Run commands",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let (tools, available) =
            sdk_tools(Some(&configured), Arc::new(Mutex::new(Vec::new()))).expect("tools");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "command_run");
        assert_eq!(available, vec!["custom:command_run"]);
    }
}
