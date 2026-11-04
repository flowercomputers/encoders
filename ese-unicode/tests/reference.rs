//! Re-derives every table entry from the reference crates.

use ese_unicode::{class, class_of, normalize};
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::char::decompose_canonical;

fn strip(c: char) -> String {
    let mut s = String::new();
    decompose_canonical(c, |d| {
        if get_general_category(d) != GeneralCategory::NonspacingMark {
            s.push(d);
        }
    });
    s
}

#[test]
fn every_code_point_matches_the_reference_crates() {
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        let cat = get_general_category(c);
        let k = class_of(c);
        assert_eq!(
            k & class::CONTROL != 0,
            matches!(
                cat,
                GeneralCategory::Control | GeneralCategory::Format | GeneralCategory::PrivateUse
            ),
            "{cp:#x} control"
        );
        assert_eq!(
            k & class::WHITESPACE != 0,
            c.is_whitespace(),
            "{cp:#x} whitespace"
        );
        assert_eq!(
            k & class::PUNCTUATION != 0,
            matches!(
                cat,
                GeneralCategory::ConnectorPunctuation
                    | GeneralCategory::DashPunctuation
                    | GeneralCategory::OpenPunctuation
                    | GeneralCategory::ClosePunctuation
                    | GeneralCategory::InitialPunctuation
                    | GeneralCategory::FinalPunctuation
                    | GeneralCategory::OtherPunctuation
            ),
            "{cp:#x} punctuation"
        );
        assert_eq!(
            k & class::NONSPACING_MARK != 0,
            cat == GeneralCategory::NonspacingMark,
            "{cp:#x} Mn"
        );
        assert_eq!(
            k & class::HANGUL_SYLLABLE != 0,
            (0xAC00..=0xD7A3).contains(&cp),
            "{cp:#x} hangul"
        );

        let ident = c.to_string();
        let st = strip(c);
        let lo: String = c.to_lowercase().collect();
        let sl: String = st.chars().flat_map(char::to_lowercase).collect();
        assert_eq!(&*normalize(c, false, false), ident, "{cp:#x} identity");
        assert_eq!(&*normalize(c, true, false), st, "{cp:#x} strip");
        assert_eq!(&*normalize(c, false, true), lo, "{cp:#x} lower");
        assert_eq!(&*normalize(c, true, true), sl, "{cp:#x} strip+lower");
    }
}

#[test]
fn cjk_ranges() {
    assert!(class_of('你') & class::CJK != 0);
    assert!(class_of('\u{F900}') & class::CJK != 0);
    assert!(class_of('a') & class::CJK == 0);
    assert!(class_of('\u{3000}') & class::CJK == 0);
}

#[test]
fn crate_version_is_the_unicode_version() {
    let (major, minor, _) = ese_unicode::UNICODE_VERSION;
    assert_eq!(env!("CARGO_PKG_VERSION_MAJOR"), major.to_string());
    assert_eq!(env!("CARGO_PKG_VERSION_MINOR"), minor.to_string());
    // and the reference crates agree with the tables they generated
    assert_eq!(
        unicode_normalization::UNICODE_VERSION,
        ese_unicode::UNICODE_VERSION
    );
}
