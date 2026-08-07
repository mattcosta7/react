pub mod code_frame;
pub mod js_string;

pub use js_string::JsString;

use std::sync::{Mutex, OnceLock};

use rustc_hash::FxHashSet;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An interned source filename.
///
/// `SourceLocation` is `Copy` and is threaded through every IR in the compiler,
/// so the filename cannot be an owned `String`. The set of distinct filenames a
/// process observes is bounded by the number of modules it compiles, so they
/// are interned once and referenced as `&'static str` from then on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceFilename(&'static str);

impl SourceFilename {
    pub fn new(name: &str) -> Self {
        static FILENAMES: OnceLock<Mutex<FxHashSet<&'static str>>> = OnceLock::new();
        let filenames = FILENAMES.get_or_init(|| Mutex::new(FxHashSet::default()));
        let mut filenames = filenames.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(existing) = filenames.get(name) {
            return Self(existing);
        }
        let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
        filenames.insert(leaked);
        Self(leaked)
    }

    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for SourceFilename {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl Serialize for SourceFilename {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for SourceFilename {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(Self::new(&name))
    }
}

/// Error categories matching the TS ErrorCategory enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    Hooks,
    CapitalizedCalls,
    StaticComponents,
    UseMemo,
    VoidUseMemo,
    PreserveManualMemo,
    MemoDependencies,
    IncompatibleLibrary,
    Immutability,
    Globals,
    Refs,
    EffectDependencies,
    EffectExhaustiveDependencies,
    EffectSetState,
    EffectDerivationsOfState,
    ErrorBoundaries,
    Purity,
    RenderSetState,
    Invariant,
    Todo,
    Syntax,
    UnsupportedSyntax,
    Config,
    Gating,
    Suppression,
    FBT,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Error,
    Warning,
    Hint,
    Off,
}

impl ErrorCategory {
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // These map to "Compilation Skipped" (Warning severity)
            ErrorCategory::EffectDependencies
            | ErrorCategory::IncompatibleLibrary
            | ErrorCategory::PreserveManualMemo
            | ErrorCategory::UnsupportedSyntax => ErrorSeverity::Warning,

            // Todo is Hint
            ErrorCategory::Todo => ErrorSeverity::Hint,

            // Invariant and all others are Error severity
            _ => ErrorSeverity::Error,
        }
    }

    /// The severity to use in logged output, matching the TS compiler's
    /// `getRuleForCategory()`. This may differ from the internal `severity()`
    /// used for panicThreshold logic. In particular, `PreserveManualMemo` is
    /// `Warning` internally (so it doesn't trigger panicThreshold throws) but
    /// `Error` in logged output (matching TS behavior).
    pub fn logged_severity(&self) -> ErrorSeverity {
        match self {
            ErrorCategory::PreserveManualMemo => ErrorSeverity::Error,
            _ => self.severity(),
        }
    }
}

/// Suggestion operations for auto-fixes
#[derive(Debug, Clone, Serialize)]
pub enum CompilerSuggestionOperation {
    InsertBefore,
    InsertAfter,
    Remove,
    Replace,
}

/// A compiler suggestion for fixing an error
#[derive(Debug, Clone, Serialize)]
pub struct CompilerSuggestion {
    pub op: CompilerSuggestionOperation,
    pub range: (usize, usize),
    pub description: String,
    pub text: Option<String>, // None for Remove operations
}

/// Source location (matches Babel's SourceLocation format)
/// This is the HIR source location, separate from AST's BaseNode location.
/// GeneratedSource is represented as None.
///
/// Locations carry the full provenance of the source construct they came from:
/// start/end line and column, byte index, and the originating filename. Codegen
/// copies this straight onto the Babel AST nodes it materializes so that Babel
/// can emit accurate source maps for the compiled output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub start: Position,
    pub end: Position,
    /// The source file this location came from, as reported by the parser's
    /// `sourceFilename` option. Skipped during serialization to keep the
    /// existing logger/diagnostic payload shapes unchanged.
    #[serde(default, skip_serializing)]
    pub filename: Option<SourceFilename>,
}

impl SourceLocation {
    /// Create a location without filename provenance. Prefer carrying the
    /// filename through from the original AST node whenever one is available.
    pub fn new(start: Position, end: Position) -> Self {
        Self {
            start,
            end,
            filename: None,
        }
    }

    /// Create a location spanning `start`..`end` with the given filename.
    pub fn with_filename(start: Position, end: Position, filename: Option<SourceFilename>) -> Self {
        Self {
            start,
            end,
            filename,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub column: u32,
    /// Byte offset in the source file. Preserved for logger event serialization.
    #[serde(default, skip_serializing)]
    pub index: Option<u32>,
}

/// Sentinel value for generated/synthetic source locations
pub const GENERATED_SOURCE: Option<SourceLocation> = None;

/// Detail for a diagnostic
#[derive(Debug, Clone, Serialize)]
pub enum CompilerDiagnosticDetail {
    Error {
        loc: Option<SourceLocation>,
        message: Option<String>,
        /// The identifier name from the AST source location, if this error
        /// points to an identifier node. Preserved for logger event serialization
        /// to match Babel's SourceLocation.identifierName field.
        #[serde(skip)]
        identifier_name: Option<String>,
    },
    Hint {
        message: String,
    },
}

/// A single compiler diagnostic (new-style)
#[derive(Debug, Clone)]
pub struct CompilerDiagnostic {
    pub category: ErrorCategory,
    pub reason: String,
    pub description: Option<String>,
    pub details: Vec<CompilerDiagnosticDetail>,
    pub suggestions: Option<Vec<CompilerSuggestion>>,
}

impl CompilerDiagnostic {
    pub fn new(
        category: ErrorCategory,
        reason: impl Into<String>,
        description: Option<String>,
    ) -> Self {
        Self {
            category,
            reason: reason.into(),
            description,
            details: Vec::new(),
            suggestions: None,
        }
    }

    pub fn severity(&self) -> ErrorSeverity {
        self.category.severity()
    }

    pub fn logged_severity(&self) -> ErrorSeverity {
        self.category.logged_severity()
    }

    pub fn with_detail(mut self, detail: CompilerDiagnosticDetail) -> Self {
        self.details.push(detail);
        self
    }

    /// Create a Todo diagnostic (matches TS `CompilerError.throwTodo()`).
    pub fn todo(reason: impl Into<String>, loc: Option<SourceLocation>) -> Self {
        let reason = reason.into();
        let mut diag = Self::new(ErrorCategory::Todo, reason.clone(), None);
        diag.details.push(CompilerDiagnosticDetail::Error {
            loc,
            message: Some(reason),
            identifier_name: None,
        });
        diag
    }

    /// Create a diagnostic from a CompilerErrorDetail.
    pub fn from_detail(detail: CompilerErrorDetail) -> Self {
        Self::new(
            detail.category,
            detail.reason.clone(),
            detail.description.clone(),
        )
        .with_detail(CompilerDiagnosticDetail::Error {
            loc: detail.loc,
            message: Some(detail.reason),
            identifier_name: None,
        })
    }

    pub fn primary_location(&self) -> Option<&SourceLocation> {
        self.details.iter().find_map(|d| match d {
            CompilerDiagnosticDetail::Error { loc, .. } => loc.as_ref(), // identifier_name covered by ..
            _ => None,
        })
    }
}

/// Legacy-style error detail (matches CompilerErrorDetail in TS)
#[derive(Debug, Clone, Serialize)]
pub struct CompilerErrorDetail {
    pub category: ErrorCategory,
    pub reason: String,
    pub description: Option<String>,
    pub loc: Option<SourceLocation>,
    pub suggestions: Option<Vec<CompilerSuggestion>>,
}

impl CompilerErrorDetail {
    pub fn new(category: ErrorCategory, reason: impl Into<String>) -> Self {
        Self {
            category,
            reason: reason.into(),
            description: None,
            loc: None,
            suggestions: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_loc(mut self, loc: Option<SourceLocation>) -> Self {
        self.loc = loc;
        self
    }

    pub fn severity(&self) -> ErrorSeverity {
        self.category.severity()
    }

    pub fn logged_severity(&self) -> ErrorSeverity {
        self.category.logged_severity()
    }
}

/// Aggregate compiler error - can contain multiple diagnostics.
/// This is the main error type thrown/returned by the compiler.
#[derive(Debug, Clone)]
pub struct CompilerError {
    pub details: Vec<CompilerErrorOrDiagnostic>,
    /// When false, this error was accumulated on the Environment via
    /// `record_error()` / `record_diagnostic()` and returned at the end
    /// of the pipeline. In TS, `CompileUnexpectedThrow` is only emitted
    /// for errors that are **thrown** (not accumulated). Defaults to `true`
    /// because errors created directly (e.g., via `?` from a pass) are
    /// analogous to thrown errors in the TS code.
    pub is_thrown: bool,
}

/// Either a new-style diagnostic or legacy error detail
#[derive(Debug, Clone)]
pub enum CompilerErrorOrDiagnostic {
    Diagnostic(CompilerDiagnostic),
    ErrorDetail(CompilerErrorDetail),
}

impl CompilerErrorOrDiagnostic {
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Diagnostic(d) => d.severity(),
            Self::ErrorDetail(d) => d.severity(),
        }
    }

    pub fn logged_severity(&self) -> ErrorSeverity {
        match self {
            Self::Diagnostic(d) => d.logged_severity(),
            Self::ErrorDetail(d) => d.logged_severity(),
        }
    }
}

impl CompilerError {
    pub fn new() -> Self {
        Self {
            details: Vec::new(),
            is_thrown: true,
        }
    }

    pub fn push_diagnostic(&mut self, diagnostic: CompilerDiagnostic) {
        if diagnostic.severity() != ErrorSeverity::Off {
            self.details
                .push(CompilerErrorOrDiagnostic::Diagnostic(diagnostic));
        }
    }

    pub fn push_error_detail(&mut self, detail: CompilerErrorDetail) {
        if detail.severity() != ErrorSeverity::Off {
            self.details
                .push(CompilerErrorOrDiagnostic::ErrorDetail(detail));
        }
    }

    pub fn has_errors(&self) -> bool {
        self.details
            .iter()
            .any(|d| d.severity() == ErrorSeverity::Error)
    }

    pub fn has_any_errors(&self) -> bool {
        !self.details.is_empty()
    }

    /// Check if any error detail has Invariant category.
    pub fn has_invariant_errors(&self) -> bool {
        self.details.iter().any(|d| {
            let cat = match d {
                CompilerErrorOrDiagnostic::Diagnostic(d) => d.category,
                CompilerErrorOrDiagnostic::ErrorDetail(d) => d.category,
            };
            cat == ErrorCategory::Invariant
        })
    }

    pub fn merge(&mut self, other: CompilerError) {
        self.details.extend(other.details);
    }

    /// Check if all error details are non-invariant.
    /// In TS, this is used to determine if an error thrown during compilation
    /// should be logged as CompileUnexpectedThrow.
    pub fn is_all_non_invariant(&self) -> bool {
        self.details.iter().all(|d| {
            let cat = match d {
                CompilerErrorOrDiagnostic::Diagnostic(d) => d.category,
                CompilerErrorOrDiagnostic::ErrorDetail(d) => d.category,
            };
            cat != ErrorCategory::Invariant
        })
    }

    /// Format as a string matching the TS `CompilerError.toString()` output.
    /// Used for the `data` field of `CompileUnexpectedThrow` events.
    ///
    /// Format per detail: `"Category: reason. Description. (line:column)"`
    /// Multiple details are joined with `"\n\n"`.
    pub fn to_string_for_event(&self) -> String {
        self.details
            .iter()
            .map(|d| {
                let (category, reason, description, loc) = match d {
                    CompilerErrorOrDiagnostic::Diagnostic(d) => {
                        let loc = d.primary_location().cloned();
                        (d.category, &d.reason, &d.description, loc)
                    }
                    CompilerErrorOrDiagnostic::ErrorDetail(d) => {
                        (d.category, &d.reason, &d.description, d.loc)
                    }
                };
                let mut buf = format!("{}: {}", format_category_heading(category), reason);
                if let Some(desc) = description {
                    buf.push_str(&format!(". {}.", desc));
                }
                if let Some(loc) = loc {
                    buf.push_str(&format!(" ({}:{})", loc.start.line, loc.start.column));
                }
                buf
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Default for CompilerError {
    fn default() -> Self {
        Self::new()
    }
}

/// Allow `?` to convert a `CompilerError` into a `CompilerDiagnostic`
/// when the enclosing function returns `Result<T, CompilerDiagnostic>`.
///
/// This typically happens when `record_error()` returns `Err(CompilerError)`
/// for an Invariant error, and the calling function already returns
/// `Result<T, CompilerDiagnostic>`. The conversion extracts the first
/// error detail from the aggregate error.
impl From<CompilerError> for CompilerDiagnostic {
    fn from(err: CompilerError) -> Self {
        if let Some(first) = err.details.into_iter().next() {
            match first {
                CompilerErrorOrDiagnostic::Diagnostic(d) => d,
                CompilerErrorOrDiagnostic::ErrorDetail(d) => CompilerDiagnostic::from_detail(d),
            }
        } else {
            CompilerDiagnostic::new(ErrorCategory::Invariant, "Unknown compiler error", None)
        }
    }
}

impl From<CompilerDiagnostic> for CompilerError {
    fn from(diagnostic: CompilerDiagnostic) -> Self {
        let mut error = CompilerError::new();
        // Todo diagnostics should produce ErrorDetail (flat loc format), matching
        // the TS behavior where CompilerError.throwTodo() creates a CompilerErrorDetail
        // with loc directly on it, not a CompilerDiagnostic with sub-details.
        if diagnostic.category == ErrorCategory::Todo {
            let loc = diagnostic.primary_location().cloned();
            error.push_error_detail(CompilerErrorDetail {
                category: diagnostic.category,
                reason: diagnostic.reason,
                description: diagnostic.description,
                loc,
                suggestions: diagnostic.suggestions,
            });
        } else {
            error.push_diagnostic(diagnostic);
        }
        error
    }
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for detail in &self.details {
            match detail {
                CompilerErrorOrDiagnostic::Diagnostic(d) => {
                    write!(f, "{}: {}", format_category_heading(d.category), d.reason)?;
                    if let Some(desc) = &d.description {
                        write!(f, ". {}.", desc)?;
                    }
                }
                CompilerErrorOrDiagnostic::ErrorDetail(d) => {
                    write!(f, "{}: {}", format_category_heading(d.category), d.reason)?;
                    if let Some(desc) = &d.description {
                        write!(f, ". {}.", desc)?;
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::error::Error for CompilerError {}

pub fn format_category_heading(category: ErrorCategory) -> &'static str {
    match category {
        ErrorCategory::EffectDependencies
        | ErrorCategory::IncompatibleLibrary
        | ErrorCategory::PreserveManualMemo
        | ErrorCategory::UnsupportedSyntax => "Compilation Skipped",
        ErrorCategory::Invariant => "Invariant",
        ErrorCategory::Todo => "Todo",
        _ => "Error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(line: u32, column: u32) -> Position {
        Position {
            line,
            column,
            index: None,
        }
    }

    #[test]
    fn interning_returns_the_same_pointer_for_equal_names() {
        let a = SourceFilename::new("src/Interned.jsx");
        let b = SourceFilename::new("src/Interned.jsx");

        assert_eq!(a, b);
        assert!(std::ptr::eq(a.as_str(), b.as_str()));
    }

    #[test]
    fn distinct_names_stay_distinct() {
        let a = SourceFilename::new("src/One.jsx");
        let b = SourceFilename::new("src/Two.jsx");

        assert_ne!(a, b);
        assert_eq!(a.as_str(), "src/One.jsx");
        assert_eq!(b.as_str(), "src/Two.jsx");
    }

    /// `SourceLocation` is used as a map key to deduplicate diagnostics (see
    /// `validate_hooks_usage`), and `filename` participates in `Eq`/`Hash`.
    /// Two locations at the same line/column in *different* files are therefore
    /// distinct keys, which is what makes that dedup correct across files.
    #[test]
    fn filename_participates_in_equality_and_hashing() {
        let same_position = || (position(3, 7), position(3, 19));

        let (start, end) = same_position();
        let in_a = SourceLocation::with_filename(start, end, Some(SourceFilename::new("a.jsx")));
        let (start, end) = same_position();
        let in_b = SourceLocation::with_filename(start, end, Some(SourceFilename::new("b.jsx")));
        let (start, end) = same_position();
        let unattributed = SourceLocation::new(start, end);

        assert_ne!(in_a, in_b);
        assert_ne!(in_a, unattributed);

        let mut set = FxHashSet::default();
        set.insert(in_a);
        set.insert(in_b);
        set.insert(unattributed);
        assert_eq!(set.len(), 3);

        let (start, end) = same_position();
        let in_a_again =
            SourceLocation::with_filename(start, end, Some(SourceFilename::new("a.jsx")));
        assert_eq!(in_a, in_a_again);
        assert!(!set.insert(in_a_again));
    }

    /// `filename` is deserialized but deliberately not serialized, so that
    /// logger and diagnostic payload shapes are unchanged.
    #[test]
    fn filename_is_not_serialized_but_is_accepted_on_input() {
        let loc = SourceLocation::with_filename(
            position(1, 0),
            position(1, 4),
            Some(SourceFilename::new("src/Skipped.jsx")),
        );

        let json = serde_json::to_string(&loc).expect("serializes");
        assert!(!json.contains("filename"), "unexpected payload: {json}");

        let parsed: SourceLocation = serde_json::from_str(
            r#"{"start":{"line":1,"column":0},"end":{"line":1,"column":4},"filename":"src/Parsed.jsx"}"#,
        )
        .expect("deserializes");
        assert_eq!(
            parsed.filename.map(|f| f.as_str().to_string()),
            Some("src/Parsed.jsx".to_string())
        );
    }
}
