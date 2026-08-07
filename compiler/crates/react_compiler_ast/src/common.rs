use react_compiler_diagnostics::{
    Position as HirPosition, SourceFilename as HirSourceFilename,
    SourceLocation as HirSourceLocation, SourceOffset,
};
use serde::Deserialize;
use serde::Serialize;

/// An AST subtree the compiler does not model with typed nodes (type
/// annotations, class bodies, parser extras). Wraps JSON text: serialization
/// is verbatim pass-through and deserialization streams the subtree into text
/// without retaining a `serde_json::Value` tree. Consumers that inspect these
/// subtrees parse on demand via [`RawNode::parse_value`]; paths that do so
/// repeatedly per traversal pay a parse each time, so cache the parsed Value
/// at the call site if it shows up in profiles.
///
/// Deserialize is hand-implemented with a transcode rather than capturing a
/// `RawValue` directly: most nodes sit under `#[serde(tag = "type")]` enums,
/// whose content buffering breaks `RawValue`'s text-borrowing capture.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct RawNode(pub Box<serde_json::value::RawValue>);

impl<'de> serde::Deserialize<'de> for RawNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut buf = Vec::new();
        let mut ser = serde_json::Serializer::new(&mut buf);
        serde_transcode::transcode(deserializer, &mut ser).map_err(serde::de::Error::custom)?;
        let text = String::from_utf8(buf).map_err(serde::de::Error::custom)?;
        serde_json::value::RawValue::from_string(text)
            .map(RawNode)
            .map_err(serde::de::Error::custom)
    }
}

impl RawNode {
    pub fn from_value(value: &serde_json::Value) -> Self {
        RawNode(
            serde_json::value::RawValue::from_string(value.to_string())
                .expect("serde_json::Value always serializes to valid JSON"),
        )
    }

    pub fn null() -> Self {
        RawNode(
            serde_json::value::RawValue::from_string("null".to_string())
                .expect("null is valid JSON"),
        )
    }

    /// The raw JSON text of this subtree.
    pub fn get(&self) -> &str {
        self.0.get()
    }

    /// Parse the subtree into a `serde_json::Value` for structural inspection.
    /// RawNode text is valid JSON by construction, so failure here means a
    /// broken invariant, not bad input; fail loudly rather than degrade.
    pub fn parse_value(&self) -> serde_json::Value {
        from_json_str_unbounded(self.0.get()).expect("RawNode holds valid JSON by construction")
    }

    /// The node's `"type"` field, without parsing the whole subtree into a Value.
    pub fn type_name(&self) -> Option<String> {
        #[derive(Deserialize)]
        struct TypeProbe {
            #[serde(rename = "type")]
            type_name: Option<String>,
        }
        from_json_str_unbounded::<TypeProbe>(self.0.get())
            .ok()
            .and_then(|p| p.type_name)
    }
}

/// Parse JSON text with serde_json's recursion limit disabled. Every internal
/// reparse of [`RawNode`] text must go through this: the napi entrypoint
/// deserializes arbitrarily deep ASTs with the limit disabled (on a 64MB
/// stack), and the tolerant statement path's reparses must not quietly
/// reintroduce the default limit.
pub fn from_json_str_unbounded<'de, T: serde::Deserialize<'de>>(
    s: &'de str,
) -> serde_json::Result<T> {
    let mut deserializer = serde_json::Deserializer::from_str(s);
    deserializer.disable_recursion_limit();
    T::deserialize(&mut deserializer)
}

/// Custom deserializer that distinguishes "field absent" from "field: null".
/// - JSON field absent → `None` (via `#[serde(default)]`)
/// - JSON field `null` → `Some(RawNode("null"))`
/// - JSON field with value → `Some(raw value)`
///
/// Use with `#[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "nullable_value")]`
pub fn nullable_value<'de, D>(deserializer: D) -> Result<Option<RawNode>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    RawNode::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub column: u32,
    /// Offset of this position in the source file, in the unit the frontend
    /// reports (UTF-16 code units from a JS parser, UTF-8 bytes from swc/oxc).
    /// See [`SourceOffset`] before using this to index source text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<SourceOffset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub start: Position,
    pub end: Position,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "identifierName"
    )]
    pub identifier_name: Option<String>,
}

impl SourceLocation {
    /// Convert this Babel AST location into the HIR representation, preserving
    /// line, column, UTF-16 offset, and the originating filename.
    ///
    /// This is the ingest direction: the compiler is driven as a Babel plugin,
    /// so locations start life on the Babel AST that Babel's parser produced.
    /// This is the single entrypoint for AST -> HIR conversion; do not hand-roll
    /// partial conversions that drop `index` or `filename`, because whatever is
    /// lost here cannot be recovered by [`SourceLocation::from_hir`] on the way
    /// back out.
    pub fn to_hir(&self) -> HirSourceLocation {
        HirSourceLocation {
            start: HirPosition {
                line: self.start.line,
                column: self.start.column,
                index: self.start.index,
            },
            end: HirPosition {
                line: self.end.line,
                column: self.end.column,
                index: self.end.index,
            },
            filename: self.filename.as_deref().map(HirSourceFilename::new),
        }
    }

    /// Materialize a Babel AST location from an HIR location.
    ///
    /// This is the emit direction, and the one that determines source map
    /// quality. The compiler hands its compiled Babel AST back to Babel, which
    /// splices it into the program and runs its own generator; that generator
    /// derives the source map purely from the `loc` on each node. So a node
    /// materialized by codegen only appears in the source map if it goes
    /// through here with the location of the source construct it represents.
    ///
    /// Every codegen site that assigns a location to a materialized node must
    /// use this so that provenance is never silently truncated.
    pub fn from_hir(loc: &HirSourceLocation) -> Self {
        Self {
            start: Position {
                line: loc.start.line,
                column: loc.start.column,
                index: loc.start.index,
            },
            end: Position {
                line: loc.end.line,
                column: loc.end.column,
                index: loc.end.index,
            },
            filename: loc.filename.map(|f| f.as_str().to_string()),
            identifier_name: None,
        }
    }
}

/// Convert an optional AST location into an optional HIR location.
pub fn ast_loc_to_hir(loc: Option<&SourceLocation>) -> Option<HirSourceLocation> {
    loc.map(SourceLocation::to_hir)
}

/// Convert an optional HIR location into an optional AST location.
///
/// `None` (i.e. `GeneratedSource`) stays `None`: compiler-only scaffolding must
/// remain unmapped so that generated frames never blame user code.
pub fn hir_loc_to_ast(loc: Option<HirSourceLocation>) -> Option<SourceLocation> {
    loc.as_ref().map(SourceLocation::from_hir)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Comment {
    CommentBlock(CommentData),
    CommentLine(CommentData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentData {
    pub value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loc: Option<SourceLocation>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaseNode {
    // NOTE: When creating AST nodes for code generation output, use
    // `BaseNode::typed("NodeTypeName")` instead of `BaseNode::default()`
    // to ensure the "type" field is emitted during serialization.
    /// The node type string (e.g. "BlockStatement").
    /// When deserialized through a `#[serde(tag = "type")]` enum, the enum
    /// consumes the "type" field so this defaults to None. When deserialized
    /// directly, this captures the "type" field for round-trip fidelity.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loc: Option<SourceLocation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<(u32, u32)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<RawNode>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "leadingComments"
    )]
    pub leading_comments: Option<Vec<Comment>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "innerComments"
    )]
    pub inner_comments: Option<Vec<Comment>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "trailingComments"
    )]
    pub trailing_comments: Option<Vec<Comment>>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "_nodeId")]
    pub node_id: Option<u32>,
}

impl BaseNode {
    /// Create a BaseNode with the given type name.
    /// Use this when creating AST nodes for code generation to ensure the
    /// `"type"` field is present in serialized output.
    pub fn typed(type_name: &str) -> Self {
        Self {
            node_type: Some(type_name.to_string()),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ast_loc() -> SourceLocation {
        SourceLocation {
            start: Position {
                line: 12,
                column: 4,
                index: Some(SourceOffset::new(211)),
            },
            end: Position {
                line: 12,
                column: 27,
                index: Some(SourceOffset::new(234)),
            },
            filename: Some("src/Component.jsx".to_string()),
            identifier_name: None,
        }
    }

    /// Whatever `to_hir` drops cannot be recovered on the way back out, so it
    /// must carry every field. Dropping `index` or `filename` here is silent:
    /// line and column still look right, so codegen output and snapshot
    /// fixtures are unchanged while source map provenance is quietly wrong.
    #[test]
    fn to_hir_preserves_every_field() {
        let hir = ast_loc().to_hir();

        assert_eq!(hir.start.line, 12);
        assert_eq!(hir.start.column, 4);
        assert_eq!(hir.start.index, Some(SourceOffset::new(211)));
        assert_eq!(hir.end.line, 12);
        assert_eq!(hir.end.column, 27);
        assert_eq!(hir.end.index, Some(SourceOffset::new(234)));
        assert_eq!(
            hir.filename.map(|f| f.as_str().to_string()),
            Some("src/Component.jsx".to_string())
        );
    }

    #[test]
    fn ast_to_hir_to_ast_round_trips() {
        let original = ast_loc();
        let result = SourceLocation::from_hir(&original.to_hir());

        assert_eq!(result.start.line, original.start.line);
        assert_eq!(result.start.column, original.start.column);
        assert_eq!(result.start.index, original.start.index);
        assert_eq!(result.end.line, original.end.line);
        assert_eq!(result.end.column, original.end.column);
        assert_eq!(result.end.index, original.end.index);
        assert_eq!(result.filename, original.filename);
    }

    #[test]
    fn locations_without_index_or_filename_round_trip_as_none() {
        let sparse = SourceLocation {
            start: Position {
                line: 1,
                column: 0,
                index: None,
            },
            end: Position {
                line: 1,
                column: 5,
                index: None,
            },
            filename: None,
            identifier_name: None,
        };

        let result = SourceLocation::from_hir(&sparse.to_hir());

        assert_eq!(result.start.index, None);
        assert_eq!(result.end.index, None);
        assert_eq!(result.filename, None);
    }

    /// `None` means `GeneratedSource`. Compiler-only scaffolding relies on this
    /// staying `None` so that generated frames never blame user code.
    #[test]
    fn generated_source_stays_unmapped_in_both_directions() {
        assert!(ast_loc_to_hir(None).is_none());
        assert!(hir_loc_to_ast(None).is_none());
    }
}
