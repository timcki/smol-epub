//! Decode a corpus of real EPUB JPEGs and check that the smol-epub
//! decoder either produces a sensible result or rejects the file
//! cleanly without an "AC index overflow" warning fallback.
//!
//! Fixtures under tests/fixtures/ were extracted from EPUBs that
//! triggered the AC overflow path on-device.

use smol_epub::jpeg::{
    ChunkReader, DeflateReader, SliceReader, decode_jpeg, peek_jpeg_dimensions,
};

fn check(name: &str, bytes: &[u8]) -> Result<(u16, u16), String> {
    let (w, h) = peek_jpeg_dimensions(SliceReader::new(bytes))
        .map_err(|e| format!("{name}: peek: {e}"))?;
    let res = decode_jpeg(SliceReader::new(bytes), 600, 800);
    match res {
        Ok(img) => {
            assert!(img.width > 0 && img.height > 0, "{name}: zero output dims");
            assert_eq!(img.data.len(), img.stride * img.height as usize);
            Ok((w, h))
        }
        Err(e) => Err(format!("{name}: decode: {e}")),
    }
}

// Mirror the on-device call path: SD reads via 4 KB chunks through a closure.
fn check_streaming(name: &str, bytes: &[u8]) -> Result<(u16, u16), String> {
    let read_fn = |off: u32, buf: &mut [u8]| -> Result<usize, &'static str> {
        let off = off as usize;
        if off >= bytes.len() {
            return Ok(0);
        }
        let n = buf.len().min(bytes.len() - off);
        buf[..n].copy_from_slice(&bytes[off..off + n]);
        Ok(n)
    };
    let reader = ChunkReader::new(read_fn, 0, bytes.len() as u32);
    match decode_jpeg(reader, 600, 800) {
        Ok(img) => {
            assert!(img.width > 0 && img.height > 0, "{name}: zero output dims");
            Ok((img.width, img.height))
        }
        Err(e) => Err(format!("{name}: stream-decode: {e}")),
    }
}

// Mirror the on-device call path for a DEFLATE-compressed ZIP entry:
// raw JPEG bytes get DEFLATE-encoded; reads come from the compressed
// blob via the streaming closure, exactly like ZIP-stored entries with
// METHOD_DEFLATED.
fn check_deflate_streaming(name: &str, bytes: &[u8]) -> Result<(u16, u16, Vec<u8>), String> {
    use miniz_oxide::deflate::compress_to_vec;
    let compressed = compress_to_vec(bytes, 6);
    let read_fn = |off: u32, buf: &mut [u8]| -> Result<usize, &'static str> {
        let off = off as usize;
        if off >= compressed.len() {
            return Ok(0);
        }
        let n = buf.len().min(compressed.len() - off);
        buf[..n].copy_from_slice(&compressed[off..off + n]);
        Ok(n)
    };
    let reader = DeflateReader::new(read_fn, 0, compressed.len() as u32)
        .map_err(|e| format!("{name}: deflate-init: {e}"))?;
    match decode_jpeg(reader, 600, 800) {
        Ok(img) => Ok((img.width, img.height, img.data)),
        Err(e) => Err(format!("{name}: deflate-decode: {e}")),
    }
}

#[test]
fn baseline_leviathan_cover_decodes() {
    let bytes = include_bytes!("fixtures/leviathan_cover.jpeg");
    let r = check("leviathan_cover", bytes);
    eprintln!("leviathan_cover (slice): {r:?}");
    r.unwrap();
    let s = check_streaming("leviathan_cover", bytes);
    eprintln!("leviathan_cover (stream): {s:?}");
    s.unwrap();
}

// Decoding a DEFLATE-compressed JPEG must produce the same image as
// decoding the raw JPEG. Before the rewind fix, a small-scan_start
// JPEG decoded via the deflate path produced visually corrupt output
// because the post-header rewind walked past the 32 KB DEFLATE window.
#[test]
fn deflate_streaming_matches_slice_for_baseline_jpegs() {
    let cases: &[(&str, &[u8])] = &[
        ("leviathan_cover", include_bytes!("fixtures/leviathan_cover.jpeg")),
        ("artemis_cover", include_bytes!("fixtures/artemis_cover.jpg")),
        ("quickstart_cover", include_bytes!("fixtures/quickstart_cover.jpg")),
        ("piotr_inline", include_bytes!("fixtures/piotr_inline_00007.jpg")),
    ];
    for (name, bytes) in cases {
        let slice = decode_jpeg(SliceReader::new(bytes), 600, 800)
            .unwrap_or_else(|e| panic!("{name}: slice decode failed: {e}"));
        let (dw, dh, ddata) =
            check_deflate_streaming(name, bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (slice.width, slice.height),
            (dw, dh),
            "{name}: deflate dims diverge from slice"
        );
        assert_eq!(
            slice.data, ddata,
            "{name}: deflate decode produced different pixels than slice decode"
        );
    }
}

#[test]
fn baseline_artemis_cover_decodes() {
    let bytes = include_bytes!("fixtures/artemis_cover.jpg");
    let r = check("artemis_cover", bytes);
    eprintln!("artemis_cover: {r:?}");
    r.unwrap();
}

#[test]
fn baseline_quickstart_cover_decodes() {
    let bytes = include_bytes!("fixtures/quickstart_cover.jpg");
    let r = check("quickstart_cover", bytes);
    eprintln!("quickstart_cover: {r:?}");
    r.unwrap();
}

#[test]
fn baseline_piotr_inline_decodes() {
    let bytes = include_bytes!("fixtures/piotr_inline_00007.jpg");
    let r = check("piotr_inline", bytes);
    eprintln!("piotr_inline: {r:?}");
    r.unwrap();
}

// progressive — must be rejected cleanly with a specific error,
// not silently produce a degraded image via the partial first-scan path.

fn assert_progressive_rejected(bytes: &[u8]) {
    let err = decode_jpeg(SliceReader::new(bytes), 600, 800)
        .expect_err("progressive must be rejected");
    assert!(
        err.contains("progressive"),
        "expected progressive-rejection error, got: {err}"
    );
}

#[test]
fn progressive_moonbound_logo_rejected() {
    assert_progressive_rejected(include_bytes!("fixtures/moonbound_logo.jpg"));
}

#[test]
fn progressive_moonbound_pg143_rejected() {
    assert_progressive_rejected(include_bytes!("fixtures/moonbound_pg143.jpg"));
}

#[test]
fn progressive_moonbound_newsletter_rejected() {
    assert_progressive_rejected(include_bytes!("fixtures/moonbound_newsletter.jpg"));
}

