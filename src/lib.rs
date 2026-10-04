//! PyO3 bindings over `rust_local_ai`.
//!
//! Every I/O method has an awaitable form and a `_sync` form. Both run the same
//! Rust future on one shared Tokio runtime; the `_sync` form releases the GIL
//! while it blocks.

use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use futures_util::StreamExt;
use pyo3::{
    exceptions::{PyStopAsyncIteration, PyStopIteration, PyTypeError},
    prelude::*,
    types::{PyDict, PyString},
};
use rust_local_ai as core;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Errors

fn to_py_err(err: core::LocalAiError) -> PyErr {
    Python::attach(|py| match build_py_err(py, &err) {
        Ok(err) => err,
        Err(import_failure) => import_failure,
    })
}

fn build_py_err(py: Python<'_>, err: &core::LocalAiError) -> PyResult<PyErr> {
    use core::LocalAiError as E;
    let errors = py.import("python_local_ai._errors")?;
    let message = err.to_string();
    let kwargs = PyDict::new(py);
    let class = match err {
        E::Unavailable { reason } => {
            kwargs.set_item("reason", availability_reason(*reason))?;
            "UnavailableError"
        }
        E::SessionBusy => "SessionBusyError",
        E::SessionClosed => "SessionClosedError",
        E::ModelClosed => "ModelClosedError",
        E::UnsupportedCapability { capability } => {
            kwargs.set_item("capability", *capability)?;
            "UnsupportedCapabilityError"
        }
        E::InvalidConfig(_) => "InvalidConfigError",
        E::Backend { backend, .. } => {
            kwargs.set_item("backend", backend_kind(*backend))?;
            "BackendError"
        }
        E::Cancelled => "GenerationCancelledError",
        _ => "LocalAiError",
    };
    let instance = errors.getattr(class)?.call((message,), Some(&kwargs))?;
    Ok(PyErr::from_value(instance))
}

// ---------------------------------------------------------------------------
// Enum spellings: lower snake case strings, typed as Literal in the stubs.

fn backend_kind(kind: core::BackendKind) -> &'static str {
    use core::BackendKind as K;
    match kind {
        K::AppleFoundationModels => "apple_foundation_models",
        K::WindowsAi => "windows_ai",
        K::UbuntuInferenceSnap => "ubuntu_inference_snap",
        K::LinuxProvider => "linux_provider",
        K::Fake => "fake",
        K::Unsupported => "unsupported",
        _ => "unknown",
    }
}

fn availability_reason(reason: core::AvailabilityReason) -> &'static str {
    use core::AvailabilityReason as R;
    match reason {
        R::UnsupportedOperatingSystem => "unsupported_operating_system",
        R::UnsupportedOsVersion => "unsupported_os_version",
        R::UnsupportedHardware => "unsupported_hardware",
        R::SystemFeatureDisabled => "system_feature_disabled",
        R::ModelNotReady => "model_not_ready",
        R::ProviderNotInstalled => "provider_not_installed",
        R::BackendFailure => "backend_failure",
        _ => "unknown",
    }
}

fn parse_availability_reason(value: &str) -> PyResult<core::AvailabilityReason> {
    use core::AvailabilityReason as R;
    Ok(match value {
        "unsupported_operating_system" => R::UnsupportedOperatingSystem,
        "unsupported_os_version" => R::UnsupportedOsVersion,
        "unsupported_hardware" => R::UnsupportedHardware,
        "system_feature_disabled" => R::SystemFeatureDisabled,
        "model_not_ready" => R::ModelNotReady,
        "provider_not_installed" => R::ProviderNotInstalled,
        "backend_failure" => R::BackendFailure,
        other => {
            return Err(PyTypeError::new_err(format!(
                "unknown availability reason {other:?}"
            )))
        }
    })
}

// ---------------------------------------------------------------------------
// Value types

/// Which backend was selected.
#[pyclass(module = "python_local_ai", frozen, get_all, eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
struct BackendInfo {
    kind: &'static str,
    name: String,
    system_managed_model: bool,
}

#[pymethods]
impl BackendInfo {
    fn __repr__(&self) -> String {
        format!(
            "BackendInfo(kind={:?}, name={:?}, system_managed_model={})",
            self.kind,
            self.name,
            py_bool(self.system_managed_model)
        )
    }
}

impl From<core::BackendInfo> for BackendInfo {
    fn from(info: core::BackendInfo) -> Self {
        Self {
            kind: backend_kind(info.kind),
            name: info.name,
            system_managed_model: info.system_managed_model,
        }
    }
}

/// Whether a session can be opened right now, and why not when it cannot.
#[pyclass(module = "python_local_ai", frozen, get_all, eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
struct Availability {
    available: bool,
    reason: Option<&'static str>,
    detail: Option<String>,
}

#[pymethods]
impl Availability {
    fn __bool__(&self) -> bool {
        self.available
    }

    fn __repr__(&self) -> String {
        format!(
            "Availability(available={}, reason={}, detail={})",
            py_bool(self.available),
            py_opt(self.reason),
            py_opt(self.detail.as_deref())
        )
    }
}

impl From<core::Availability> for Availability {
    fn from(value: core::Availability) -> Self {
        Self {
            available: value.available,
            reason: value.reason.map(availability_reason),
            detail: value.detail,
        }
    }
}

/// What the active backend really supports.
#[pyclass(module = "python_local_ai", frozen, get_all, eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
struct Capabilities {
    text_generation: bool,
    streaming: bool,
    structured_output: bool,
    tool_calling: bool,
    vision: bool,
    token_counting: bool,
    cancellation: bool,
    concurrent_sessions: bool,
    system_managed_model: bool,
}

#[pymethods]
impl Capabilities {
    #[new]
    #[pyo3(signature = (*, text_generation=false, streaming=false, structured_output=false, tool_calling=false, vision=false, token_counting=false, cancellation=false, concurrent_sessions=false, system_managed_model=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        text_generation: bool,
        streaming: bool,
        structured_output: bool,
        tool_calling: bool,
        vision: bool,
        token_counting: bool,
        cancellation: bool,
        concurrent_sessions: bool,
        system_managed_model: bool,
    ) -> Self {
        Self {
            text_generation,
            streaming,
            structured_output,
            tool_calling,
            vision,
            token_counting,
            cancellation,
            concurrent_sessions,
            system_managed_model,
        }
    }

    fn __repr__(&self) -> String {
        let fields = [
            ("text_generation", self.text_generation),
            ("streaming", self.streaming),
            ("structured_output", self.structured_output),
            ("tool_calling", self.tool_calling),
            ("vision", self.vision),
            ("token_counting", self.token_counting),
            ("cancellation", self.cancellation),
            ("concurrent_sessions", self.concurrent_sessions),
            ("system_managed_model", self.system_managed_model),
        ];
        let body: Vec<String> = fields
            .iter()
            .map(|(name, value)| format!("{name}={}", py_bool(*value)))
            .collect();
        format!("Capabilities({})", body.join(", "))
    }
}

impl From<core::Capabilities> for Capabilities {
    fn from(c: core::Capabilities) -> Self {
        Self {
            text_generation: c.text_generation,
            streaming: c.streaming,
            structured_output: c.structured_output,
            tool_calling: c.tool_calling,
            vision: c.vision,
            token_counting: c.token_counting,
            cancellation: c.cancellation,
            concurrent_sessions: c.concurrent_sessions,
            system_managed_model: c.system_managed_model,
        }
    }
}

impl From<&Capabilities> for core::Capabilities {
    fn from(c: &Capabilities) -> Self {
        Self {
            text_generation: c.text_generation,
            streaming: c.streaming,
            structured_output: c.structured_output,
            tool_calling: c.tool_calling,
            vision: c.vision,
            token_counting: c.token_counting,
            cancellation: c.cancellation,
            concurrent_sessions: c.concurrent_sessions,
            system_managed_model: c.system_managed_model,
        }
    }
}

/// A completed generation.
#[pyclass(module = "python_local_ai", frozen, get_all, eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
struct AiResponse {
    text: String,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    finish_reason: Option<String>,
}

#[pymethods]
impl AiResponse {
    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!(
            "AiResponse(text={:?}, input_tokens={}, output_tokens={}, finish_reason={})",
            self.text,
            py_opt(self.input_tokens),
            py_opt(self.output_tokens),
            py_opt(self.finish_reason.as_deref())
        )
    }
}

impl From<core::AiResponse> for AiResponse {
    fn from(r: core::AiResponse) -> Self {
        Self {
            text: r.text,
            input_tokens: r.input_tokens,
            output_tokens: r.output_tokens,
            finish_reason: r.finish_reason,
        }
    }
}

fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

fn py_opt<T: std::fmt::Debug>(value: Option<T>) -> String {
    value.map_or_else(|| "None".into(), |v| format!("{v:?}"))
}

// ---------------------------------------------------------------------------
// Running futures

fn runtime() -> &'static tokio::runtime::Runtime {
    pyo3_async_runtimes::tokio::get_runtime()
}

/// Runs `fut` on the shared runtime, returning an awaitable.
fn awaitable<'py, F, T>(py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = core::Result<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, async move { fut.await.map_err(to_py_err) })
}

/// Runs `fut` on the shared runtime, blocking without holding the GIL.
fn blocking<F, T>(py: Python<'_>, fut: F) -> PyResult<T>
where
    F: Future<Output = core::Result<T>> + Send,
    T: Send,
{
    py.detach(|| runtime().block_on(fut)).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Generation options

fn generation_config(
    py: Python<'_>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    response_format: Option<&Bound<'_, PyAny>>,
) -> PyResult<core::GenerationConfig> {
    let response_format = match response_format {
        None => core::ResponseFormat::Text,
        Some(value) if value.is_none() => core::ResponseFormat::Text,
        Some(value) => {
            if let Ok(name) = value.cast::<PyString>() {
                match name.to_cow()?.as_ref() {
                    "text" => core::ResponseFormat::Text,
                    "json" => core::ResponseFormat::Json,
                    other => {
                        return Err(PyTypeError::new_err(format!(
                            "response_format must be 'text', 'json' or a JSON schema dict, not {other:?}"
                        )))
                    }
                }
            } else {
                let json: String = py
                    .import("json")?
                    .call_method1("dumps", (value,))?
                    .extract()?;
                let schema = serde_json::from_str(&json).map_err(|e| {
                    PyTypeError::new_err(format!("response_format schema is not JSON: {e}"))
                })?;
                core::ResponseFormat::JsonSchema(schema)
            }
        }
    };
    Ok(core::GenerationConfig {
        max_output_tokens,
        temperature,
        top_p,
        seed,
        response_format,
    })
}

/// A session plus a binding-level turn flag. The core only takes its
/// generation lease inside `generate`, after `add_query_chunk` has already
/// changed the conversation; reserving the turn first means a rejected
/// (busy) call never leaves its prompt behind.
#[derive(Clone)]
struct SessionHandle {
    session: core::LocalAiSession,
    turn: Arc<AtomicBool>,
}

struct TurnGuard(Arc<AtomicBool>);

impl Drop for TurnGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl SessionHandle {
    fn begin_turn(&self) -> core::Result<TurnGuard> {
        if self.session.is_closed() {
            return Err(core::LocalAiError::SessionClosed);
        }
        self.turn
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| core::LocalAiError::SessionBusy)?;
        Ok(TurnGuard(self.turn.clone()))
    }

    async fn generate(
        self,
        prompt: Option<String>,
        config: core::GenerationConfig,
    ) -> core::Result<AiResponse> {
        let _turn = self.begin_turn()?;
        if let Some(prompt) = prompt {
            self.session.add_query_chunk(&prompt).await?;
        }
        self.session.generate(config).await.map(AiResponse::from)
    }

    async fn start_stream(
        self,
        prompt: Option<String>,
        config: core::GenerationConfig,
    ) -> core::Result<(core::ResponseStream, TurnGuard)> {
        let turn = self.begin_turn()?;
        if let Some(prompt) = prompt {
            self.session.add_query_chunk(&prompt).await?;
        }
        let stream = self.session.generate_stream(config).await?;
        Ok((stream, turn))
    }
}

// ---------------------------------------------------------------------------
// Model

/// Entry point to the selected backend. Obtain one with :func:`detect`.
#[pyclass(module = "python_local_ai", frozen)]
struct LocalAiModel {
    inner: core::LocalAiModel,
}

#[pymethods]
impl LocalAiModel {
    #[getter]
    fn backend(&self) -> BackendInfo {
        self.inner.backend().into()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn availability<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let model = self.inner.clone();
        awaitable(py, async move {
            Ok(Availability::from(model.availability().await))
        })
    }

    fn availability_sync(&self, py: Python<'_>) -> PyResult<Availability> {
        let model = self.inner.clone();
        blocking(py, async move {
            Ok(Availability::from(model.availability().await))
        })
    }

    fn capabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let model = self.inner.clone();
        awaitable(py, async move {
            Ok(Capabilities::from(model.capabilities().await))
        })
    }

    fn capabilities_sync(&self, py: Python<'_>) -> PyResult<Capabilities> {
        let model = self.inner.clone();
        blocking(py, async move {
            Ok(Capabilities::from(model.capabilities().await))
        })
    }

    #[pyo3(signature = (instructions=None))]
    fn open_session<'py>(
        &self,
        py: Python<'py>,
        instructions: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.inner.clone();
        awaitable(py, async move {
            let inner = model.open_session(instructions.as_deref()).await?;
            Ok(LocalAiSession::new(inner))
        })
    }

    #[pyo3(signature = (instructions=None))]
    fn open_session_sync(
        &self,
        py: Python<'_>,
        instructions: Option<String>,
    ) -> PyResult<LocalAiSession> {
        let model = self.inner.clone();
        blocking(py, async move {
            let inner = model.open_session(instructions.as_deref()).await?;
            Ok(LocalAiSession::new(inner))
        })
    }

    /// Prevents new sessions. Existing sessions stay usable until closed.
    fn close(&self) {
        self.inner.close();
    }

    fn __repr__(&self) -> String {
        format!("LocalAiModel(backend={})", self.backend().__repr__())
    }
}

/// Selects the best OS- or distribution-managed backend in this build.
///
/// Never installs a provider, starts a service, or downloads a model.
#[pyfunction]
fn detect(py: Python<'_>) -> PyResult<LocalAiModel> {
    blocking(py, async { core::detect().await }).map(|inner| LocalAiModel { inner })
}

// ---------------------------------------------------------------------------
// Session

/// A stateful conversation. One generation runs at a time.
///
/// Use ``async with`` or ``with`` to close it automatically.
#[pyclass(module = "python_local_ai", frozen)]
struct LocalAiSession {
    inner: core::LocalAiSession,
    turn: Arc<AtomicBool>,
}

impl LocalAiSession {
    fn new(inner: core::LocalAiSession) -> Self {
        Self {
            inner,
            turn: Arc::new(AtomicBool::new(false)),
        }
    }

    fn handle(&self) -> SessionHandle {
        SessionHandle {
            session: self.inner.clone(),
            turn: self.turn.clone(),
        }
    }
}

#[pymethods]
impl LocalAiSession {
    #[getter]
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities().into()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Appends text to the pending prompt without generating.
    fn add_query_chunk<'py>(&self, py: Python<'py>, chunk: String) -> PyResult<Bound<'py, PyAny>> {
        let session = self.inner.clone();
        awaitable(py, async move { session.add_query_chunk(&chunk).await })
    }

    fn add_query_chunk_sync(&self, py: Python<'_>, chunk: String) -> PyResult<()> {
        let session = self.inner.clone();
        blocking(py, async move { session.add_query_chunk(&chunk).await })
    }

    /// Adds ``prompt`` (if given) to the pending prompt and generates a reply.
    #[pyo3(signature = (prompt=None, *, max_output_tokens=None, temperature=None, top_p=None, seed=None, response_format=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        prompt: Option<String>,
        max_output_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        seed: Option<u64>,
        response_format: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let config = generation_config(
            py,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            response_format,
        )?;
        awaitable(py, self.handle().generate(prompt, config))
    }

    #[pyo3(signature = (prompt=None, *, max_output_tokens=None, temperature=None, top_p=None, seed=None, response_format=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate_sync(
        &self,
        py: Python<'_>,
        prompt: Option<String>,
        max_output_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        seed: Option<u64>,
        response_format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<AiResponse> {
        let config = generation_config(
            py,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            response_format,
        )?;
        blocking(py, self.handle().generate(prompt, config))
    }

    /// Streams the reply as text chunks. Iterate with ``async for`` or ``for``.
    ///
    /// The generation starts on the first iteration.
    #[pyo3(signature = (prompt=None, *, max_output_tokens=None, temperature=None, top_p=None, seed=None, response_format=None))]
    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        py: Python<'_>,
        prompt: Option<String>,
        max_output_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        seed: Option<u64>,
        response_format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<ResponseStream> {
        let config = generation_config(
            py,
            max_output_tokens,
            temperature,
            top_p,
            seed,
            response_format,
        )?;
        Ok(ResponseStream {
            state: Arc::new(Mutex::new(StreamState::Pending {
                session: self.handle(),
                prompt,
                config,
            })),
        })
    }

    /// Asks the backend to stop the in-flight generation.
    fn cancel<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let session = self.inner.clone();
        awaitable(py, async move { session.cancel().await })
    }

    fn cancel_sync(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.inner.clone();
        blocking(py, async move { session.cancel().await })
    }

    fn count_tokens<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let session = self.inner.clone();
        awaitable(py, async move { session.count_tokens(&text).await })
    }

    fn count_tokens_sync(&self, py: Python<'_>, text: String) -> PyResult<u64> {
        let session = self.inner.clone();
        blocking(py, async move { session.count_tokens(&text).await })
    }

    /// Closes the session. Idempotent once it succeeds; retry if it raises.
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let session = self.inner.clone();
        awaitable(py, async move { session.close().await })
    }

    fn close_sync(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.inner.clone();
        blocking(py, async move { session.close().await })
    }

    fn __aenter__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this: Py<Self> = slf.unbind();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(this) })
    }

    #[pyo3(signature = (*_exc))]
    fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        _exc: &Bound<'py, pyo3::types::PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let session = self.inner.clone();
        awaitable(py, async move { session.close().await.map(|()| false) })
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_exc))]
    fn __exit__(&self, py: Python<'_>, _exc: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        self.close_sync(py).map(|()| false)
    }

    fn __repr__(&self) -> String {
        format!("LocalAiSession(closed={})", py_bool(self.closed()))
    }
}

// ---------------------------------------------------------------------------
// Streaming

enum StreamState {
    Pending {
        session: SessionHandle,
        prompt: Option<String>,
        config: core::GenerationConfig,
    },
    // The guard keeps the turn reserved until the stream ends or is closed.
    Running {
        stream: core::ResponseStream,
        _turn: TurnGuard,
    },
    Done,
}

/// Iterator of text chunks, usable with both ``async for`` and ``for``.
#[pyclass(module = "python_local_ai", frozen)]
struct ResponseStream {
    state: Arc<Mutex<StreamState>>,
}

/// Returns the next chunk, `None` at the end. Starts the stream on first use.
async fn next_chunk(state: Arc<Mutex<StreamState>>) -> core::Result<Option<String>> {
    let mut guard = state.lock().await;
    if let StreamState::Pending { .. } = &*guard {
        let StreamState::Pending {
            session,
            prompt,
            config,
        } = std::mem::replace(&mut *guard, StreamState::Done)
        else {
            unreachable!()
        };
        let (stream, turn) = session.start_stream(prompt, config).await?;
        *guard = StreamState::Running {
            stream,
            _turn: turn,
        };
    }
    let StreamState::Running { stream, .. } = &mut *guard else {
        return Ok(None);
    };
    match stream.next().await {
        Some(Ok(chunk)) => Ok(Some(chunk)),
        Some(Err(err)) => {
            *guard = StreamState::Done;
            Err(err)
        }
        None => {
            // Dropping the stream releases the core lease and the turn.
            *guard = StreamState::Done;
            Ok(None)
        }
    }
}

#[pymethods]
impl ResponseStream {
    fn __aiter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match next_chunk(state).await.map_err(to_py_err)? {
                Some(chunk) => Ok(chunk),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<String> {
        let state = self.state.clone();
        match blocking(py, next_chunk(state))? {
            Some(chunk) => Ok(chunk),
            None => Err(PyStopIteration::new_err(())),
        }
    }

    /// Stops consuming the stream and releases the session for the next turn.
    fn close(&self, py: Python<'_>) {
        let state = self.state.clone();
        py.detach(|| runtime().block_on(async move { *state.lock().await = StreamState::Done }));
    }
}

// ---------------------------------------------------------------------------
// Testing

/// Deterministic in-memory backend from ``rust_local_ai::testing``.
#[pyclass(module = "python_local_ai.testing", frozen)]
struct FakeBackend {
    inner: core::testing::FakeBackend,
}

#[pymethods]
impl FakeBackend {
    /// Defaults to text generation, streaming, structured output, token
    /// counting, cancellation and concurrent sessions.
    #[new]
    #[pyo3(signature = (capabilities=None))]
    fn new(capabilities: Option<PyRef<'_, Capabilities>>) -> Self {
        let inner = match capabilities {
            Some(c) => core::testing::FakeBackend::new((&*c).into()),
            None => core::testing::FakeBackend::default(),
        };
        Self { inner }
    }

    fn model(&self) -> LocalAiModel {
        LocalAiModel {
            inner: self.inner.model(),
        }
    }

    #[pyo3(signature = (available, reason=None, detail=None))]
    fn set_availability(
        &self,
        available: bool,
        reason: Option<&str>,
        detail: Option<String>,
    ) -> PyResult<()> {
        let mut value = if available {
            core::Availability::available()
        } else {
            core::Availability::unavailable(parse_availability_reason(
                reason.unwrap_or("backend_failure"),
            )?)
        };
        if let Some(detail) = detail {
            value = value.with_detail(detail);
        }
        self.inner.set_availability(value);
        Ok(())
    }

    /// Queues the next ``generate`` result. Without one, ``generate`` echoes the prompt.
    #[pyo3(signature = (text, *, input_tokens=None, output_tokens=None, finish_reason=None))]
    fn push_response(
        &self,
        text: String,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        finish_reason: Option<String>,
    ) {
        let mut response = core::AiResponse::text(text);
        response.input_tokens = input_tokens;
        response.output_tokens = output_tokens;
        response.finish_reason = finish_reason;
        self.inner.push_response(Ok(response));
    }

    /// Queues a backend failure for the next ``generate``.
    fn push_backend_error(&self, message: String) {
        self.inner.push_response(Err(core::LocalAiError::Backend {
            backend: core::BackendKind::Fake,
            message,
        }));
    }

    /// Sets the chunks the next ``stream`` yields.
    fn set_stream_chunks(&self, chunks: Vec<String>) {
        self.inner
            .set_stream_chunks(chunks.into_iter().map(Ok).collect());
    }

    fn fail_next_closes(&self, count: usize) {
        self.inner.fail_next_closes(count);
    }

    /// While blocked, ``generate`` waits; use it to test busy and cancel paths.
    fn set_generation_blocked(&self, blocked: bool) {
        self.inner.set_generation_blocked(blocked);
    }
}

// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_class::<LocalAiModel>()?;
    m.add_class::<LocalAiSession>()?;
    m.add_class::<ResponseStream>()?;
    m.add_class::<Availability>()?;
    m.add_class::<Capabilities>()?;
    m.add_class::<BackendInfo>()?;
    m.add_class::<AiResponse>()?;
    m.add_class::<FakeBackend>()?;
    m.add(
        "RUST_LOCAL_AI_REVISION",
        "68d9edc026dac2c5e645a7f22bd8b6965aec2f40",
    )?;
    Ok(())
}
