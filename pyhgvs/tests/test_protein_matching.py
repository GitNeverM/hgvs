"""
Tests for protein HGVS name parsing, formatting, normalization, and matching.

Covers:
- Amino acid conversion utilities (1-letter ↔ 3-letter)
- Protein HGVS parsing with 1-letter and 3-letter amino acid codes
- Configurable protein formatting (use_3letter flag)
- HGVS name normalization (normalize_hgvs_name)
- Semantic equivalence comparison (hgvs_names_equal, HGVSName.equivalent)
- Exception hierarchy and backward compatibility
- cDNA HGVS equivalence comparison
"""

from __future__ import unicode_literals

import pytest

import pyhgvs
from pyhgvs import (
    HGVSName,
    HGVSError,
    HGVSParseError,
    HGVSFormattingError,
    HGVSTranscriptError,
    HGVSNormalizationError,
    HGVSInvalidAminoAcidError,
    InvalidHGVSName,
    aa1_to_aa3,
    aa3_to_aa1,
    normalize_aa_allele,
    normalize_hgvs_name,
    hgvs_names_equal,
)


# ---------------------------------------------------------------------------
# Amino acid conversion utilities
# ---------------------------------------------------------------------------

class TestAminoAcidConversion:

    def test_aa1_to_aa3_standard_residues(self):
        """All 20 standard amino acids convert correctly."""
        mapping = {
            'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu',
            'F': 'Phe', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
            'K': 'Lys', 'L': 'Leu', 'M': 'Met', 'N': 'Asn',
            'P': 'Pro', 'Q': 'Gln', 'R': 'Arg', 'S': 'Ser',
            'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr',
        }
        for aa1, aa3 in mapping.items():
            assert aa1_to_aa3(aa1) == aa3, "Failed for %s" % aa1

    def test_aa3_to_aa1_standard_residues(self):
        """All 20 standard amino acids convert back correctly."""
        mapping = {
            'Ala': 'A', 'Cys': 'C', 'Asp': 'D', 'Glu': 'E',
            'Phe': 'F', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
            'Lys': 'K', 'Leu': 'L', 'Met': 'M', 'Asn': 'N',
            'Pro': 'P', 'Gln': 'Q', 'Arg': 'R', 'Ser': 'S',
            'Thr': 'T', 'Val': 'V', 'Trp': 'W', 'Tyr': 'Y',
        }
        for aa3, aa1 in mapping.items():
            assert aa3_to_aa1(aa3) == aa1, "Failed for %s" % aa3

    def test_stop_codon_aa1_to_aa3(self):
        assert aa1_to_aa3('*') == 'Ter'

    def test_stop_codon_aa3_to_aa1(self):
        assert aa3_to_aa1('Ter') == '*'

    def test_unknown_aa1_raises(self):
        with pytest.raises(HGVSInvalidAminoAcidError):
            aa1_to_aa3('Z')

    def test_unknown_aa3_raises(self):
        with pytest.raises(HGVSInvalidAminoAcidError):
            aa3_to_aa1('Foo')

    def test_normalize_aa_allele_1letter(self):
        """1-letter codes are normalised to 3-letter."""
        assert normalize_aa_allele('R', use_3letter=True) == 'Arg'
        assert normalize_aa_allele('H', use_3letter=True) == 'His'
        assert normalize_aa_allele('*', use_3letter=True) == 'Ter'

    def test_normalize_aa_allele_3letter(self):
        """3-letter codes round-trip unchanged."""
        assert normalize_aa_allele('Arg', use_3letter=True) == 'Arg'
        assert normalize_aa_allele('His', use_3letter=True) == 'His'
        assert normalize_aa_allele('Ter', use_3letter=True) == 'Ter'

    def test_normalize_aa_allele_to_1letter(self):
        """3-letter codes convert to 1-letter when requested."""
        assert normalize_aa_allele('Arg', use_3letter=False) == 'R'
        assert normalize_aa_allele('His', use_3letter=False) == 'H'
        assert normalize_aa_allele('Ter', use_3letter=False) == '*'

    def test_normalize_aa_allele_stop_variants(self):
        """Various stop-codon representations all normalise correctly."""
        for raw in ('*', 'Ter', 'Trm', 'Stop'):
            assert normalize_aa_allele(raw, use_3letter=True) == 'Ter', raw
            assert normalize_aa_allele(raw, use_3letter=False) == '*', raw

    def test_normalize_aa_allele_concatenated(self):
        """Concatenated 3-letter sequences round-trip."""
        assert normalize_aa_allele('GluSer', use_3letter=True) == 'GluSer'


# ---------------------------------------------------------------------------
# Protein HGVS parsing — three-letter notation (existing behaviour)
# ---------------------------------------------------------------------------

class TestParseProtein3Letter:

    def test_substitution(self):
        n = HGVSName('p.Arg132His')
        assert n.kind == 'p'
        assert n.start == 132
        assert n.end == 132
        assert n.ref_allele == 'Arg'
        assert n.alt_allele == 'His'
        assert n.mutation_type == '>'

    def test_no_change(self):
        n = HGVSName('p.Glu1161=')
        assert n.ref_allele == 'Glu'
        assert n.pep_extra == '='

    def test_frameshift_range(self):
        n = HGVSName('p.Glu1000_Ser1003?fs')
        assert n.mutation_type == 'delins'
        assert n.start == 1000
        assert n.end == 1003
        assert n.ref_allele == 'Glu'
        assert n.ref2_allele == 'Ser'
        assert n.pep_extra == '?fs'

    def test_with_transcript_prefix(self):
        n = HGVSName('NM_004380.2:p.Arg132His')
        assert n.transcript == 'NM_004380.2'
        assert n.start == 132

    def test_with_gene_and_transcript_prefix(self):
        n = HGVSName('NM_004380.2(IDH1):p.Arg132His')
        assert n.transcript == 'NM_004380.2'
        assert n.gene == 'IDH1'


# ---------------------------------------------------------------------------
# Protein HGVS parsing — one-letter notation (new behaviour)
# ---------------------------------------------------------------------------

class TestParseProtein1Letter:

    def test_simple_substitution(self):
        n = HGVSName('p.R132H')
        assert n.kind == 'p'
        assert n.start == 132
        assert n.end == 132
        assert n.ref_allele == 'R'
        assert n.alt_allele == 'H'
        assert n.mutation_type == '>'

    def test_no_change_1letter(self):
        n = HGVSName('p.E1161=')
        assert n.ref_allele == 'E'
        assert n.pep_extra == '='

    def test_stop_codon_1letter_as_alt(self):
        n = HGVSName('p.Arg132*')
        assert n.ref_allele == 'Arg'
        assert n.alt_allele == '*'

    def test_stop_codon_1letter_as_ref(self):
        n = HGVSName('p.*132Leu')
        assert n.ref_allele == '*'
        assert n.alt_allele == 'Leu'

    def test_various_1letter_substitutions(self):
        cases = [
            ('p.V600E', 'V', 'E', 600),
            ('p.G12D', 'G', 'D', 12),
            ('p.K27M', 'K', 'M', 27),
            ('p.S249C', 'S', 'C', 249),
        ]
        for hgvs_str, ref, alt, pos in cases:
            n = HGVSName(hgvs_str)
            assert n.ref_allele == ref, hgvs_str
            assert n.alt_allele == alt, hgvs_str
            assert n.start == pos, hgvs_str


# ---------------------------------------------------------------------------
# Protein HGVS formatting
# ---------------------------------------------------------------------------

class TestFormatProtein:

    def test_format_3letter_default(self):
        """Default formatting preserves stored allele notation."""
        n = HGVSName('p.Arg132His')
        assert n.format() == 'p.Arg132His'

    def test_format_1letter_when_stored_as_3letter(self):
        """3-letter alleles can be output as 1-letter."""
        n = HGVSName('p.Arg132His')
        assert n.format(use_3letter=False) == 'p.R132H'

    def test_format_3letter_when_stored_as_1letter(self):
        """1-letter alleles are converted to 3-letter for output."""
        n = HGVSName('p.R132H')
        assert n.format(use_3letter=True) == 'p.Arg132His'

    def test_format_1letter_round_trip(self):
        """1-letter form round-trips."""
        n = HGVSName('p.R132H')
        assert n.format(use_3letter=False) == 'p.R132H'

    def test_format_stop_codon_3letter(self):
        n = HGVSName('p.Arg132*')
        assert n.format(use_3letter=True) == 'p.Arg132Ter'

    def test_format_stop_codon_1letter(self):
        n = HGVSName('p.Arg132*')
        assert n.format(use_3letter=False) == 'p.R132*'

    def test_format_no_change(self):
        n = HGVSName('p.Glu1000=')
        assert n.format() == 'p.Glu1000='

    def test_format_frameshift(self):
        n = HGVSName('p.Glu1000_Ser1003?fs')
        assert n.format() == 'p.Glu1000_Ser1003?fs'

    def test_format_with_prefix(self):
        n = HGVSName('NM_004380.2:p.Arg132His')
        assert n.format() == 'NM_004380.2:p.Arg132His'
        assert n.format(use_prefix=False) == 'p.Arg132His'


# ---------------------------------------------------------------------------
# HGVS normalisation
# ---------------------------------------------------------------------------

class TestNormalizeHgvsName:

    def test_normalize_1letter_to_3letter(self):
        assert normalize_hgvs_name('p.R132H') == 'p.Arg132His'

    def test_normalize_3letter_unchanged(self):
        assert normalize_hgvs_name('p.Arg132His') == 'p.Arg132His'

    def test_normalize_to_1letter(self):
        assert normalize_hgvs_name('p.Arg132His', use_3letter=False) == 'p.R132H'

    def test_normalize_stop_1letter(self):
        assert normalize_hgvs_name('p.Arg132*') == 'p.Arg132Ter'

    def test_normalize_cdna_unchanged(self):
        """cDNA names pass through normalisation unchanged."""
        assert normalize_hgvs_name('c.395G>A') == 'c.395G>A'

    def test_normalize_with_transcript_prefix(self):
        result = normalize_hgvs_name('NM_004380.2:p.R132H')
        assert result == 'NM_004380.2:p.Arg132His'

    def test_normalize_invalid_raises(self):
        with pytest.raises(HGVSParseError):
            normalize_hgvs_name('p.ZZZ132')


# ---------------------------------------------------------------------------
# Semantic equivalence
# ---------------------------------------------------------------------------

class TestHgvsNamesEqual:

    # --- Protein equivalence ---

    def test_1letter_equals_3letter(self):
        assert hgvs_names_equal('p.R132H', 'p.Arg132His')

    def test_3letter_equals_3letter(self):
        assert hgvs_names_equal('p.Arg132His', 'p.Arg132His')

    def test_1letter_equals_1letter(self):
        assert hgvs_names_equal('p.R132H', 'p.R132H')

    def test_stop_codon_equivalence(self):
        assert hgvs_names_equal('p.Arg132*', 'p.Arg132Ter')

    def test_stop_codon_as_ref(self):
        assert hgvs_names_equal('p.*132Leu', 'p.Ter132Leu')

    def test_different_position_not_equal(self):
        assert not hgvs_names_equal('p.R132H', 'p.R133H')

    def test_different_ref_not_equal(self):
        assert not hgvs_names_equal('p.R132H', 'p.K132H')

    def test_different_alt_not_equal(self):
        assert not hgvs_names_equal('p.R132H', 'p.R132C')

    def test_different_kind_not_equal(self):
        assert not hgvs_names_equal('p.R132H', 'c.395G>A')

    def test_ignores_transcript_prefix(self):
        """Transcript prefix is ignored for equivalence."""
        assert hgvs_names_equal('NM_004380.2:p.R132H', 'p.Arg132His')
        assert hgvs_names_equal(
            'NM_004380.2(IDH1):p.Arg132His', 'NM_004380.3:p.R132H')

    def test_accepts_hgvsname_objects(self):
        n1 = HGVSName('p.R132H')
        n2 = HGVSName('p.Arg132His')
        assert hgvs_names_equal(n1, n2)
        assert n1.equivalent(n2)

    def test_common_oncology_variants(self):
        """Real-world equivalences used in clinical databases."""
        equivalences = [
            ('p.V600E', 'p.Val600Glu'),
            ('p.G12D', 'p.Gly12Asp'),
            ('p.G12V', 'p.Gly12Val'),
            ('p.K27M', 'p.Lys27Met'),
        ]
        for a, b in equivalences:
            assert hgvs_names_equal(a, b), "Expected %r == %r" % (a, b)

    def test_nonstandard_form_raises_parse_error(self):
        """Non-HGVS strings (e.g. histone notation 'H3.3K27M') raise HGVSParseError."""
        with pytest.raises(HGVSParseError):
            # 'p.H3.3K27M' is not valid HGVS — the '.' after 'p' splits the
            # kind at 'H3', which is not a recognised kind.
            HGVSName('p.H3.3K27M')

    # --- cDNA equivalence ---

    def test_cdna_identical(self):
        assert hgvs_names_equal('c.395G>A', 'c.395G>A')

    def test_cdna_not_equal_diff_pos(self):
        assert not hgvs_names_equal('c.395G>A', 'c.396G>A')

    def test_cdna_not_equal_diff_alt(self):
        assert not hgvs_names_equal('c.395G>A', 'c.395G>C')


# ---------------------------------------------------------------------------
# HGVSName.normalize()
# ---------------------------------------------------------------------------

class TestHgvsNameNormalize:

    def test_normalize_returns_new_object_for_protein(self):
        n = HGVSName('p.R132H')
        norm = n.normalize()
        assert norm is not n
        assert norm.ref_allele == 'Arg'
        assert norm.alt_allele == 'His'

    def test_normalize_returns_same_object_for_cdna(self):
        """Non-protein names are returned as-is."""
        n = HGVSName('c.395G>A')
        norm = n.normalize()
        assert norm is n

    def test_normalize_3letter_unchanged(self):
        n = HGVSName('p.Arg132His')
        norm = n.normalize()
        assert norm.ref_allele == 'Arg'
        assert norm.alt_allele == 'His'

    def test_normalize_stop_codon(self):
        n = HGVSName('p.Arg132*')
        norm = n.normalize()
        assert norm.alt_allele == 'Ter'


# ---------------------------------------------------------------------------
# Exception hierarchy and backward compatibility
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:

    def test_hgvs_error_is_value_error(self):
        assert issubclass(HGVSError, ValueError)

    def test_hgvs_parse_error_is_hgvs_error(self):
        assert issubclass(HGVSParseError, HGVSError)

    def test_invalid_hgvs_name_is_alias(self):
        """InvalidHGVSName is the same object as HGVSParseError."""
        assert InvalidHGVSName is HGVSParseError

    def test_catch_as_invalid_hgvs_name(self):
        """Existing code that catches InvalidHGVSName still works."""
        with pytest.raises(InvalidHGVSName):
            HGVSName('p.ZZZ')

    def test_catch_as_hgvs_parse_error(self):
        with pytest.raises(HGVSParseError):
            HGVSName('p.ZZZ')

    def test_catch_as_value_error(self):
        with pytest.raises(ValueError):
            HGVSName('p.ZZZ')

    def test_hgvs_formatting_error_is_hgvs_error(self):
        assert issubclass(HGVSFormattingError, HGVSError)

    def test_hgvs_transcript_error_is_hgvs_error(self):
        assert issubclass(HGVSTranscriptError, HGVSError)

    def test_hgvs_normalization_error_is_hgvs_error(self):
        assert issubclass(HGVSNormalizationError, HGVSError)

    def test_invalid_amino_acid_error(self):
        with pytest.raises(HGVSInvalidAminoAcidError):
            aa1_to_aa3('Z')

    def test_invalid_amino_acid_is_hgvs_error(self):
        assert issubclass(HGVSInvalidAminoAcidError, HGVSError)

    def test_parse_unknown_kind_raises_parse_error(self):
        with pytest.raises(HGVSParseError):
            HGVSName('x.123A>C')

    def test_parse_no_dot_raises_parse_error(self):
        with pytest.raises(HGVSParseError):
            HGVSName('c123A')

    def test_repr_does_not_raise_for_protein(self):
        """repr() should not raise even for protein names."""
        n = HGVSName('p.Arg132His')
        r = repr(n)
        assert 'HGVSName' in r


# ---------------------------------------------------------------------------
# HGVS stable protein categories — Frameshift
# (https://hgvs-nomenclature.org/stable/recommendations/protein/frameshift/)
# ---------------------------------------------------------------------------

class TestFrameshiftParsing:
    """Tests for HGVS protein frameshift expressions."""

    def test_simple_frameshift_3letter(self):
        """p.Arg97fs — simplest frameshift, no new AA or ter position."""
        n = HGVSName('p.Arg97fs')
        assert n.mutation_type == 'fs'
        assert n.ref_allele == 'Arg'
        assert n.start == 97
        assert n.end == 97
        assert n.alt_allele == ''
        assert n.pep_extra == 'fs'

    def test_simple_frameshift_1letter(self):
        """p.R97fs — 1-letter code."""
        n = HGVSName('p.R97fs')
        assert n.mutation_type == 'fs'
        assert n.ref_allele == 'R'
        assert n.start == 97
        assert n.pep_extra == 'fs'

    def test_frameshift_with_new_aa_and_ter_3letter(self):
        """p.Arg97ProfsTer23 — new AA + Ter + position."""
        n = HGVSName('p.Arg97ProfsTer23')
        assert n.mutation_type == 'fs'
        assert n.ref_allele == 'Arg'
        assert n.start == 97
        assert n.alt_allele == 'Pro'
        assert n.pep_extra == 'fsTer23'

    def test_frameshift_with_star_notation(self):
        """p.Arg97Profs*23 — uses * instead of Ter."""
        n = HGVSName('p.Arg97Profs*23')
        assert n.mutation_type == 'fs'
        assert n.ref_allele == 'Arg'
        assert n.alt_allele == 'Pro'
        # pep_extra normalises * → Ter during parse
        assert n.pep_extra == 'fsTer23'

    def test_frameshift_with_unknown_ter_position(self):
        """p.Ile327Argfs*? — uncertain termination position."""
        n = HGVSName('p.Ile327Argfs*?')
        assert n.mutation_type == 'fs'
        assert n.ref_allele == 'Ile'
        assert n.start == 327
        assert n.alt_allele == 'Arg'
        assert n.pep_extra == 'fsTer?'

    def test_frameshift_predicted_form_simple(self):
        """p.(Arg123fs) — predicted consequence, outer parens stripped."""
        n = HGVSName('p.(Arg123fs)')
        assert n.mutation_type == 'fs'
        assert n.ref_allele == 'Arg'
        assert n.start == 123
        assert n.pep_extra == 'fs'

    def test_frameshift_predicted_form_with_ter(self):
        """p.(Arg123LysfsTer34) — predicted frameshift with new AA."""
        n = HGVSName('p.(Arg123LysfsTer34)')
        assert n.mutation_type == 'fs'
        assert n.ref_allele == 'Arg'
        assert n.start == 123
        assert n.alt_allele == 'Lys'
        assert n.pep_extra == 'fsTer34'

    def test_frameshift_format_simple(self):
        n = HGVSName('p.Arg97fs')
        assert n.format() == 'p.Arg97fs'

    def test_frameshift_format_with_ter_3letter(self):
        n = HGVSName('p.Arg97ProfsTer23')
        assert n.format(use_3letter=True) == 'p.Arg97ProfsTer23'

    def test_frameshift_format_with_ter_1letter(self):
        """Formatting with use_3letter=False converts Ter → *."""
        n = HGVSName('p.Arg97ProfsTer23')
        assert n.format(use_3letter=False) == 'p.R97Pfs*23'

    def test_frameshift_with_star_formats_as_ter(self):
        """Input p.Arg97Profs*23 normalises to Ter in 3-letter output."""
        n = HGVSName('p.Arg97Profs*23')
        assert n.format(use_3letter=True) == 'p.Arg97ProfsTer23'

    def test_frameshift_equivalence_star_vs_ter(self):
        """p.Arg97Profs*23 and p.Arg97ProfsTer23 are semantically equal."""
        assert hgvs_names_equal('p.Arg97Profs*23', 'p.Arg97ProfsTer23')

    def test_frameshift_equivalence_1letter_vs_3letter(self):
        """p.R97Pfs*23 and p.Arg97ProfsTer23 are semantically equal."""
        assert hgvs_names_equal('p.R97Pfs*23', 'p.Arg97ProfsTer23')

    def test_frameshift_legacy_range_unchanged(self):
        """Legacy range frameshift p.Glu1161_Ser1164?fs still works."""
        n = HGVSName('p.Glu1000_Ser1003?fs')
        assert n.mutation_type == 'delins'
        assert n.start == 1000
        assert n.end == 1003
        assert n.pep_extra == '?fs'
        assert n.format() == 'p.Glu1000_Ser1003?fs'


# ---------------------------------------------------------------------------
# HGVS stable protein categories — Deletion
# ---------------------------------------------------------------------------

class TestProteinDeletion:
    """Tests for HGVS protein deletion expressions."""

    def test_single_deletion_3letter(self):
        """p.Lys23del — single-residue deletion."""
        n = HGVSName('p.Lys23del')
        assert n.mutation_type == 'del'
        assert n.ref_allele == 'Lys'
        assert n.start == 23
        assert n.end == 23
        assert n.alt_allele == ''

    def test_single_deletion_1letter(self):
        """p.K23del — 1-letter notation."""
        n = HGVSName('p.K23del')
        assert n.mutation_type == 'del'
        assert n.ref_allele == 'K'
        assert n.start == 23

    def test_range_deletion_3letter(self):
        """p.Lys23_Val25del — range deletion."""
        n = HGVSName('p.Lys23_Val25del')
        assert n.mutation_type == 'del'
        assert n.ref_allele == 'Lys'
        assert n.ref2_allele == 'Val'
        assert n.start == 23
        assert n.end == 25

    def test_range_deletion_1letter(self):
        """p.K23_V25del — range deletion 1-letter."""
        n = HGVSName('p.K23_V25del')
        assert n.mutation_type == 'del'
        assert n.ref_allele == 'K'
        assert n.ref2_allele == 'V'
        assert n.start == 23
        assert n.end == 25

    def test_deletion_format_single(self):
        n = HGVSName('p.Lys23del')
        assert n.format() == 'p.Lys23del'

    def test_deletion_format_range(self):
        n = HGVSName('p.Lys23_Val25del')
        assert n.format() == 'p.Lys23_Val25del'

    def test_deletion_format_1letter_to_3letter(self):
        n = HGVSName('p.K23del')
        assert n.format(use_3letter=True) == 'p.Lys23del'

    def test_deletion_predicted_form(self):
        """p.(Lys23del) — predicted form parens are stripped."""
        n = HGVSName('p.(Lys23del)')
        assert n.mutation_type == 'del'
        assert n.ref_allele == 'Lys'
        assert n.start == 23


# ---------------------------------------------------------------------------
# HGVS stable protein categories — Duplication
# ---------------------------------------------------------------------------

class TestProteinDuplication:
    """Tests for HGVS protein duplication expressions."""

    def test_single_duplication_3letter(self):
        """p.Lys23dup — single-residue duplication."""
        n = HGVSName('p.Lys23dup')
        assert n.mutation_type == 'dup'
        assert n.ref_allele == 'Lys'
        assert n.start == 23
        assert n.end == 23
        # alt_allele == ref_allele for dup
        assert n.alt_allele == 'Lys'

    def test_single_duplication_1letter(self):
        n = HGVSName('p.K23dup')
        assert n.mutation_type == 'dup'
        assert n.ref_allele == 'K'

    def test_range_duplication_3letter(self):
        """p.Lys23_Val25dup — range duplication."""
        n = HGVSName('p.Lys23_Val25dup')
        assert n.mutation_type == 'dup'
        assert n.start == 23
        assert n.end == 25
        assert n.ref_allele == 'Lys'
        assert n.ref2_allele == 'Val'

    def test_duplication_format_single(self):
        n = HGVSName('p.Lys23dup')
        assert n.format() == 'p.Lys23dup'

    def test_duplication_format_range(self):
        n = HGVSName('p.Lys23_Val25dup')
        assert n.format() == 'p.Lys23_Val25dup'

    def test_duplication_format_1letter_to_3letter(self):
        n = HGVSName('p.K23dup')
        assert n.format(use_3letter=True) == 'p.Lys23dup'

    def test_duplication_predicted_form(self):
        """p.(Lys23dup) — predicted form parens are stripped."""
        n = HGVSName('p.(Lys23dup)')
        assert n.mutation_type == 'dup'
        assert n.start == 23


# ---------------------------------------------------------------------------
# HGVS stable protein categories — Insertion
# ---------------------------------------------------------------------------

class TestProteinInsertion:
    """Tests for HGVS protein insertion expressions."""

    def test_insertion_3letter(self):
        """p.Lys23_Leu24insArgSerGln — insertion between two residues."""
        n = HGVSName('p.Lys23_Leu24insArgSerGln')
        assert n.mutation_type == 'ins'
        assert n.ref_allele == 'Lys'
        assert n.ref2_allele == 'Leu'
        assert n.start == 23
        assert n.end == 24
        assert n.alt_allele == 'ArgSerGln'

    def test_insertion_single_aa(self):
        """p.Lys23_Leu24insArg — insertion of a single amino acid."""
        n = HGVSName('p.Lys23_Leu24insArg')
        assert n.mutation_type == 'ins'
        assert n.alt_allele == 'Arg'

    def test_insertion_1letter_alt(self):
        """p.K23_L24insRSQ — 1-letter alt codes."""
        n = HGVSName('p.K23_L24insRSQ')
        assert n.mutation_type == 'ins'
        assert n.ref_allele == 'K'
        assert n.ref2_allele == 'L'
        assert n.alt_allele == 'RSQ'

    def test_insertion_stop_codon(self):
        """p.Lys23_Leu24ins* — insert stop codon."""
        n = HGVSName('p.Lys23_Leu24ins*')
        assert n.mutation_type == 'ins'
        assert n.alt_allele == '*'

    def test_insertion_unknown_count(self):
        """p.Lys23_Leu24ins[5] — insert unknown sequence of 5 residues."""
        n = HGVSName('p.Lys23_Leu24ins[5]')
        assert n.mutation_type == 'ins'
        assert n.alt_allele == '[5]'

    def test_insertion_format(self):
        n = HGVSName('p.Lys23_Leu24insArgSerGln')
        assert n.format() == 'p.Lys23_Leu24insArgSerGln'

    def test_insertion_format_1letter(self):
        n = HGVSName('p.Lys23_Leu24insArgSerGln')
        assert n.format(use_3letter=False) == 'p.K23_L24insRSQ'


# ---------------------------------------------------------------------------
# HGVS stable protein categories — Deletion-Insertion (delins)
# ---------------------------------------------------------------------------

class TestProteinDelIns:
    """Tests for HGVS protein deletion-insertion expressions."""

    def test_single_delins_3letter(self):
        """p.Cys28delinsTrpVal — single-residue delins."""
        n = HGVSName('p.Cys28delinsTrpVal')
        assert n.mutation_type == 'delins'
        assert n.ref_allele == 'Cys'
        assert n.start == 28
        assert n.end == 28
        assert n.alt_allele == 'TrpVal'

    def test_single_delins_single_alt(self):
        """p.Cys28delinsTrp."""
        n = HGVSName('p.Cys28delinsTrp')
        assert n.mutation_type == 'delins'
        assert n.alt_allele == 'Trp'

    def test_range_delins_3letter(self):
        """p.Cys28_Lys29delinsTrp — range delins."""
        n = HGVSName('p.Cys28_Lys29delinsTrp')
        assert n.mutation_type == 'delins'
        assert n.ref_allele == 'Cys'
        assert n.ref2_allele == 'Lys'
        assert n.start == 28
        assert n.end == 29
        assert n.alt_allele == 'Trp'

    def test_delins_1letter(self):
        n = HGVSName('p.C28delinsWV')
        assert n.mutation_type == 'delins'
        assert n.ref_allele == 'C'
        assert n.alt_allele == 'WV'

    def test_delins_format_single(self):
        n = HGVSName('p.Cys28delinsTrpVal')
        assert n.format() == 'p.Cys28delinsTrpVal'

    def test_delins_format_range(self):
        n = HGVSName('p.Cys28_Lys29delinsTrp')
        assert n.format() == 'p.Cys28_Lys29delinsTrp'

    def test_delins_format_1letter(self):
        n = HGVSName('p.Cys28delinsTrpVal')
        assert n.format(use_3letter=False) == 'p.C28delinsWV'

    def test_delins_predicted_form(self):
        """p.(Cys28delinsTrpVal) — predicted form parens are stripped."""
        n = HGVSName('p.(Cys28delinsTrpVal)')
        assert n.mutation_type == 'delins'
        assert n.alt_allele == 'TrpVal'


# ---------------------------------------------------------------------------
# HGVS stable protein categories — Extension
# ---------------------------------------------------------------------------

class TestProteinExtension:
    """Tests for HGVS protein extension expressions (N-term and C-term)."""

    def test_nterm_extension_3letter(self):
        """p.Met1ext-5 — N-terminal extension."""
        n = HGVSName('p.Met1ext-5')
        assert n.mutation_type == 'ext'
        assert n.ref_allele == 'Met'
        assert n.start == 1
        assert n.alt_allele == ''
        assert n.pep_extra == 'ext-5'

    def test_nterm_extension_unknown(self):
        """p.Met1ext? — N-terminal extension of unknown length."""
        n = HGVSName('p.Met1ext?')
        assert n.mutation_type == 'ext'
        assert n.pep_extra == 'ext?'

    def test_cterm_extension_stop_3letter(self):
        """p.*110GlnextTer17 — C-terminal extension from stop codon."""
        n = HGVSName('p.*110GlnextTer17')
        assert n.mutation_type == 'ext'
        assert n.ref_allele == '*'
        assert n.start == 110
        assert n.alt_allele == 'Gln'
        assert n.pep_extra == 'extTer17'

    def test_cterm_extension_ter_notation(self):
        """p.Ter110GlnextTer17 — C-terminal extension using Ter."""
        n = HGVSName('p.Ter110GlnextTer17')
        assert n.mutation_type == 'ext'
        assert n.ref_allele == 'Ter'
        assert n.start == 110
        assert n.alt_allele == 'Gln'
        assert n.pep_extra == 'extTer17'

    def test_extension_format_nterm(self):
        n = HGVSName('p.Met1ext-5')
        assert n.format() == 'p.Met1ext-5'

    def test_extension_format_cterm_3letter(self):
        n = HGVSName('p.*110GlnextTer17')
        assert n.format(use_3letter=True) == 'p.Ter110GlnextTer17'

    def test_extension_format_cterm_1letter(self):
        """With use_3letter=False, Ter → * in pep_extra."""
        n = HGVSName('p.*110GlnextTer17')
        result = n.format(use_3letter=False)
        assert 'ext*17' in result


# ---------------------------------------------------------------------------
# HGVS stable protein categories — Repeated Sequences
# ---------------------------------------------------------------------------

class TestProteinRepeatedSequences:
    """Tests for HGVS protein repeated-sequence expressions."""

    def test_single_repeat_3letter(self):
        """p.Ala2[10] — 10 copies of Ala at position 2."""
        n = HGVSName('p.Ala2[10]')
        assert n.mutation_type == 'rep'
        assert n.ref_allele == 'Ala'
        assert n.start == 2
        assert n.end == 2
        assert n.pep_extra == '[10]'

    def test_single_repeat_1letter(self):
        n = HGVSName('p.A2[10]')
        assert n.mutation_type == 'rep'
        assert n.ref_allele == 'A'
        assert n.pep_extra == '[10]'

    def test_repeat_unknown_count(self):
        """p.Ala2[?] — unknown repeat count."""
        n = HGVSName('p.Ala2[?]')
        assert n.mutation_type == 'rep'
        assert n.pep_extra == '[?]'

    def test_range_repeat(self):
        """p.Ala2_Pro5[10] — repeat over a range."""
        n = HGVSName('p.Ala2_Pro5[10]')
        assert n.mutation_type == 'rep'
        assert n.ref_allele == 'Ala'
        assert n.ref2_allele == 'Pro'
        assert n.start == 2
        assert n.end == 5
        assert n.pep_extra == '[10]'

    def test_repeat_format_single(self):
        n = HGVSName('p.Ala2[10]')
        assert n.format() == 'p.Ala2[10]'

    def test_repeat_format_range(self):
        n = HGVSName('p.Ala2_Pro5[10]')
        assert n.format() == 'p.Ala2_Pro5[10]'

    def test_repeat_format_1letter_to_3letter(self):
        n = HGVSName('p.A2[10]')
        assert n.format(use_3letter=True) == 'p.Ala2[10]'


# ---------------------------------------------------------------------------
# Predicted-form bracket stripping — cross-category
# ---------------------------------------------------------------------------

class TestPredictedForms:
    """Outer parentheses p.(...) are stripped and the inner form is parsed."""

    def test_predicted_substitution(self):
        n = HGVSName('p.(Arg54Ser)')
        assert n.mutation_type == '>'
        assert n.ref_allele == 'Arg'
        assert n.alt_allele == 'Ser'
        assert n.start == 54

    def test_predicted_no_change(self):
        n = HGVSName('p.(Glu1161=)')
        assert n.ref_allele == 'Glu'
        assert n.pep_extra == '='

    def test_predicted_deletion_single(self):
        n = HGVSName('p.(Lys23del)')
        assert n.mutation_type == 'del'
        assert n.start == 23

    def test_predicted_duplication(self):
        n = HGVSName('p.(Lys23dup)')
        assert n.mutation_type == 'dup'

    def test_predicted_delins(self):
        n = HGVSName('p.(Cys28delinsTrpVal)')
        assert n.mutation_type == 'delins'
        assert n.alt_allele == 'TrpVal'

    def test_predicted_frameshift_simple(self):
        n = HGVSName('p.(Arg123fs)')
        assert n.mutation_type == 'fs'
        assert n.pep_extra == 'fs'

    def test_predicted_frameshift_with_ter(self):
        n = HGVSName('p.(Arg123LysfsTer34)')
        assert n.mutation_type == 'fs'
        assert n.alt_allele == 'Lys'
        assert n.pep_extra == 'fsTer34'


# ---------------------------------------------------------------------------
# Format / round-trip for new protein mutation types
# ---------------------------------------------------------------------------

class TestNewProteinTypeFormatting:
    """Round-trip parse → format for all new HGVS protein categories."""

    @pytest.mark.parametrize('hgvs_str', [
        # Frameshift
        'p.Arg97fs',
        'p.Arg97ProfsTer23',
        'p.Ile327Argfs',
        # Deletion
        'p.Lys23del',
        'p.Lys23_Val25del',
        # Duplication
        'p.Lys23dup',
        'p.Lys23_Val25dup',
        # Insertion
        'p.Lys23_Leu24insArg',
        'p.Lys23_Leu24insArgSerGln',
        # DelIns
        'p.Cys28delinsTrpVal',
        'p.Cys28_Lys29delinsTrp',
        # Extension
        'p.Met1ext-5',
        # Repeated sequences
        'p.Ala2[10]',
        'p.Ala2_Pro5[10]',
    ])
    def test_round_trip(self, hgvs_str):
        """Parse then format should reproduce the original string."""
        n = HGVSName(hgvs_str)
        formatted = n.format()
        assert formatted == hgvs_str, (
            "Round-trip failed for %r: got %r" % (hgvs_str, formatted))

    @pytest.mark.parametrize('hgvs_str', [
        'p.Arg97fs',
        'p.Lys23del',
        'p.Lys23dup',
        'p.Lys23_Leu24insArg',
        'p.Cys28delinsTrpVal',
        'p.Ala2[10]',
    ])
    def test_does_not_raise(self, hgvs_str):
        """Parsing should not raise for any listed HGVS expression."""
        HGVSName(hgvs_str)   # must not raise


# ---------------------------------------------------------------------------
# HGVS stable: Substitution extended tests
# ---------------------------------------------------------------------------

class TestSubstitutionExtended:
    """Extra substitution cases from HGVS stable recommendations."""

    def test_nonsense_mutation_stop_alt(self):
        """p.Arg213* — nonsense / stop-gain."""
        n = HGVSName('p.Arg213*')
        assert n.mutation_type == '>'
        assert n.ref_allele == 'Arg'
        assert n.alt_allele == '*'
        assert n.start == 213

    def test_stop_codon_as_ref(self):
        """p.*45Ser — stop-loss mutation."""
        n = HGVSName('p.*45Ser')
        assert n.mutation_type == '>'
        assert n.ref_allele == '*'
        assert n.alt_allele == 'Ser'

    def test_synonymous_no_change_ter_notation(self):
        """p.Ter1161= — no-change using Ter for stop codon."""
        n = HGVSName('p.Ter1161=')
        assert n.ref_allele == 'Ter'
        assert n.pep_extra == '='

    def test_1letter_stop_as_ref_with_new_aa(self):
        """p.*110Gln — stop → Gln."""
        n = HGVSName('p.*110Gln')
        assert n.mutation_type == '>'
        assert n.ref_allele == '*'
        assert n.alt_allele == 'Gln'
