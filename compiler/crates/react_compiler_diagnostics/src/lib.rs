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

/// The unit a frontend measures source offsets in.
///
/// There is no single right answer, which is why this is explicit. JavaScript
/// indexes strings in UTF-16 code units, so Babel's `loc.start.index`, ESLint
/// fix ranges, and the `range` of [`CompilerSuggestion`] are all UTF-16. Rust's
/// `str` is indexed in UTF-8 bytes, so a Rust-native parser reports those
/// instead: swc's `swc_ecma_react_compiler` bridge fills [`Position::index`]
/// with `BytePos` values directly.
///
/// The two coincide only for ASCII. Interpreting one as the other silently
/// yields the wrong slice for any source containing non-ASCII text, and panics
/// outright when an offset lands inside a multibyte character.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SourceOffsetEncoding {
    /// UTF-16 code units, as reported by a JavaScript parser. The default,
    /// because the Babel bridge is the frontend that supplies source text.
    #[default]
    Utf16CodeUnits,
    /// UTF-8 bytes, as reported by a Rust-native parser (swc, oxc).
    Utf8Bytes,
}

/// Source text paired with the offset encoding of the frontend that produced
/// the AST for it.
///
/// Offsets are meaningless without knowing their unit, so the two travel
/// together rather than letting a consumer assume.
#[derive(Debug, Clone, Copy)]
pub struct SourceText<'a> {
    text: &'a str,
    encoding: SourceOffsetEncoding,
}

impl<'a> SourceText<'a> {
    pub fn new(text: &'a str, encoding: SourceOffsetEncoding) -> Self {
        Self { text, encoding }
    }

    pub fn text(&self) -> &'a str {
        self.text
    }

    pub fn encoding(&self) -> SourceOffsetEncoding {
        self.encoding
    }
}

/// An offset into source text, in the unit its frontend reports.
///
/// The unit is deliberately *not* baked into this type, because it differs by
/// frontend (see [`SourceOffsetEncoding`]). What the type does enforce is that
/// you cannot index Rust source text with the raw value: resolving requires a
/// [`SourceText`], which carries the encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceOffset(u32);

impl SourceOffset {
    pub fn new(offset: u32) -> Self {
        Self(offset)
    }

    /// The raw offset, in whatever unit the frontend used. Use this only when
    /// handing the value back to a consumer that shares that unit; never to
    /// index a Rust `str`.
    pub fn get(self) -> u32 {
        self.0
    }

    /// Resolve to a UTF-8 byte offset within `source`.
    ///
    /// Returns `None` if the offset is past the end of the text or does not
    /// land on a character boundary, rather than panicking the way `str` range
    /// indexing would.
    pub fn to_byte_offset(self, source: SourceText<'_>) -> Option<usize> {
        let target = self.0 as usize;
        match source.encoding {
            SourceOffsetEncoding::Utf8Bytes => (target <= source.text.len()
                && source.text.is_char_boundary(target))
            .then_some(target),
            SourceOffsetEncoding::Utf16CodeUnits => {
                let mut utf16 = 0usize;
                for (byte_idx, ch) in source.text.char_indices() {
                    match utf16.cmp(&target) {
                        std::cmp::Ordering::Equal => return Some(byte_idx),
                        // Stepped past the target: it pointed into a surrogate pair.
                        std::cmp::Ordering::Greater => return None,
                        std::cmp::Ordering::Less => {}
                    }
                    utf16 += ch.len_utf16();
                }
                (utf16 == target).then_some(source.text.len())
            }
        }
    }

    /// Build from a UTF-8 byte offset into `source`, expressed in that
    /// source's encoding.
    ///
    /// This is what a Rust-native frontend needs when it must report an offset
    /// in a unit other than its own. Returns `None` if the offset is out of
    /// bounds or is not a character boundary.
    pub fn from_byte_offset(byte_offset: usize, source: SourceText<'_>) -> Option<Self> {
        if byte_offset > source.text.len() || !source.text.is_char_boundary(byte_offset) {
            return None;
        }
        Some(Self(match source.encoding {
            SourceOffsetEncoding::Utf8Bytes => byte_offset as u32,
            SourceOffsetEncoding::Utf16CodeUnits => {
                source.text[..byte_offset].encode_utf16().count() as u32
            }
        }))
    }
}

impl std::fmt::Display for SourceOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Source location (matches Babel's SourceLocation format)
/// This is the HIR source location, separate from AST's BaseNode location.
/// GeneratedSource is represented as None.
///
/// Locations carry the full provenance of the source construct they came from:
/// start/end line and column, offset, and the originating filename. Codegen
/// copies this straight onto the AST nodes it materializes so that the printer
/// can emit accurate source maps for the compiled output.
///
/// # Implementing a non-Babel frontend
///
/// The representation is printer-agnostic, but it is richer than what a
/// Rust-native parser reports. swc (`Span { lo, hi }`) and oxc
/// (`Span { start, end }`) carry only byte offsets, with no line, column, or
/// filename, so such a frontend must resolve line/column itself and declare its
/// offset unit via [`SourceOffsetEncoding`]. `None` continues to mean
/// "generated", which every printer must render as an unmapped node.
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

    /// The text this location spans within `source`.
    ///
    /// Handles the offset-unit conversion the raw positions require, so callers
    /// never index the source text directly. Returns `None` when either
    /// endpoint is absent, the range is inverted, or an offset does not land on
    /// a character boundary.
    pub fn slice<'a>(&self, source: SourceText<'a>) -> Option<&'a str> {
        let start = self.start.index?;
        let end = self.end.index?;
        if start > end {
            return None;
        }
        let start = start.to_byte_offset(source)?;
        let end = end.to_byte_offset(source)?;
        source.text().get(start..end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub column: u32,
    /// Offset of this position in the source file, in the unit the frontend
    /// reports. See [`SourceOffsetEncoding`] before using this to index source
    /// text.
    #[serde(default, skip_serializing)]
    pub index: Option<SourceOffset>,
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

    fn span(start: u32, end: u32) -> SourceLocation {
        SourceLocation::new(
            Position {
                line: 1,
                column: start,
                index: Some(SourceOffset::new(start)),
            },
            Position {
                line: 1,
                column: end,
                index: Some(SourceOffset::new(end)),
            },
        )
    }

    /// Source as a JS frontend (Babel) reports it.
    fn utf16(text: &str) -> SourceText<'_> {
        SourceText::new(text, SourceOffsetEncoding::Utf16CodeUnits)
    }

    /// Source as a Rust-native frontend (swc, oxc) reports it.
    fn utf8(text: &str) -> SourceText<'_> {
        SourceText::new(text, SourceOffsetEncoding::Utf8Bytes)
    }

    /// The units coincide for ASCII, which is why treating one as the other
    /// survives every ASCII test fixture.
    #[test]
    fn both_encodings_agree_for_ascii() {
        let code = "const useEffect = 1;";
        assert_eq!(span(6, 15).slice(utf16(code)), Some("useEffect"));
        assert_eq!(span(6, 15).slice(utf8(code)), Some("useEffect"));
        assert_eq!(SourceOffset::new(6).to_byte_offset(utf16(code)), Some(6));
    }

    /// A single non-ASCII character shifts every following byte offset, so the
    /// same numeric offset means different things to the two frontends. Reading
    /// a UTF-16 offset as bytes would yield " useEffec" here.
    #[test]
    fn encodings_diverge_for_non_ascii_source() {
        // "é" is one UTF-16 code unit and two UTF-8 bytes.
        let code = "const é = 1; useEffect();";
        let byte_start = code.find("useEffect").expect("present");
        assert_eq!(byte_start, 14);

        assert_eq!(
            SourceOffset::from_byte_offset(byte_start, utf16(code)),
            Some(SourceOffset::new(13))
        );
        assert_eq!(
            SourceOffset::from_byte_offset(byte_start, utf8(code)),
            Some(SourceOffset::new(14))
        );

        assert_eq!(span(13, 22).slice(utf16(code)), Some("useEffect"));
        assert_eq!(span(14, 23).slice(utf8(code)), Some("useEffect"));
        // Each frontend's offsets are wrong when read in the other's unit.
        assert_eq!(span(13, 22).slice(utf8(code)), Some(" useEffec"));
    }

    /// Astral characters are two UTF-16 code units and four UTF-8 bytes, so an
    /// offset can point into the middle of a surrogate pair. That must be
    /// `None` rather than a panic.
    #[test]
    fn surrogate_halves_and_interior_bytes_are_rejected() {
        let code = "const a = '😀'; useEffect();";
        let byte_start = code.find("useEffect").expect("present");

        let offset = SourceOffset::from_byte_offset(byte_start, utf16(code)).expect("converts");
        assert_eq!(offset.to_byte_offset(utf16(code)), Some(byte_start));

        // The emoji starts at UTF-16 offset 11; offset 12 is its trailing
        // surrogate, which is not a character boundary in UTF-8.
        assert_eq!(SourceOffset::new(12).to_byte_offset(utf16(code)), None);
        // Likewise byte 12 lands inside the emoji's 4-byte encoding.
        assert_eq!(SourceOffset::new(12).to_byte_offset(utf8(code)), None);
    }

    /// Out-of-range and inverted inputs return `None` instead of panicking the
    /// way `&code[start..end]` would.
    #[test]
    fn out_of_range_and_inverted_offsets_are_rejected() {
        let code = "const a = 1;";
        let len = code.len() as u32;

        assert_eq!(
            SourceOffset::new(len).to_byte_offset(utf16(code)),
            Some(code.len())
        );
        assert_eq!(
            SourceOffset::new(len).to_byte_offset(utf8(code)),
            Some(code.len())
        );
        assert_eq!(SourceOffset::new(len + 1).to_byte_offset(utf16(code)), None);
        assert_eq!(SourceOffset::new(len + 1).to_byte_offset(utf8(code)), None);
        assert_eq!(span(9, 3).slice(utf16(code)), None);
        assert_eq!(span(0, 999).slice(utf16(code)), None);
        assert_eq!(SourceOffset::from_byte_offset(999, utf16(code)), None);
    }

    /// A Rust-native frontend reports UTF-8 byte spans natively, so resolving
    /// them must be an identity that still validates boundaries.
    #[test]
    fn utf8_byte_spans_round_trip_in_both_encodings() {
        let code = "const 日本 = 1; useEffect();";
        for (byte_idx, _) in code.char_indices() {
            for source in [utf16(code), utf8(code)] {
                let offset = SourceOffset::from_byte_offset(byte_idx, source).expect("converts");
                assert_eq!(offset.to_byte_offset(source), Some(byte_idx));
            }
        }
    }
}
