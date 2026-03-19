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
# HGVS protein Substitution (per HGVS stable recommendations)
# https://hgvs-nomenclature.org/stable/recommendations/protein/substitution/
# ---------------------------------------------------------------------------

class TestProteinSubstitution:
    """Substitution: one amino acid replaced by another (including stop codon)."""

    def test_missense_3letter(self):
        """Standard missense: Trp24Cys."""
        n = HGVSName('p.Trp24Cys')
        assert n.mutation_type == '>'
        assert n.start == 24
        assert n.ref_allele == 'Trp'
        assert n.alt_allele == 'Cys'

    def test_missense_1letter(self):
        """Standard missense 1-letter: W24C."""
        n = HGVSName('p.W24C')
        assert n.mutation_type == '>'
        assert n.start == 24
        assert n.ref_allele == 'W'
        assert n.alt_allele == 'C'

    def test_nonsense_ter_3letter(self):
        """Nonsense: Trp24Ter (stop codon in 3-letter form)."""
        n = HGVSName('p.Trp24Ter')
        assert n.ref_allele == 'Trp'
        assert n.alt_allele == 'Ter'
        assert n.mutation_type == '>'

    def test_nonsense_star_1letter(self):
        """Nonsense: Arg132* (stop codon as asterisk)."""
        n = HGVSName('p.Arg132*')
        assert n.ref_allele == 'Arg'
        assert n.alt_allele == '*'
        assert n.mutation_type == '>'

    def test_star_normalises_to_ter(self):
        """'*' is normalised to 'Ter' in 3-letter mode."""
        n = HGVSName('p.Arg132*')
        assert n.format(use_3letter=True) == 'p.Arg132Ter'
        assert n.format(use_3letter=False) == 'p.R132*'

    def test_stop_as_ref(self):
        """Stop codon as reference allele: *132Leu or Ter132Leu."""
        n = HGVSName('p.*132Leu')
        assert n.ref_allele == '*'
        assert n.alt_allele == 'Leu'

    def test_predicted_substitution(self):
        """Predicted consequence: p.(Trp24Cys)."""
        n = HGVSName('p.(Trp24Cys)')
        assert n.predicted is True
        assert n.ref_allele == 'Trp'
        assert n.alt_allele == 'Cys'
        assert n.format() == 'p.(Trp24Cys)'

    def test_predicted_1letter(self):
        """Predicted 1-letter: p.(R132H) - formats as 3-letter by default."""
        n = HGVSName('p.(R132H)')
        assert n.predicted is True
        assert n.ref_allele == 'R'
        assert n.alt_allele == 'H'
        # default use_3letter=True converts to 3-letter representation
        assert n.format(use_3letter=True) == 'p.(Arg132His)'
        assert n.format(use_3letter=False) == 'p.(R132H)'

    def test_no_change_3letter(self):
        """Synonymous (no change): p.Glu1161=."""
        n = HGVSName('p.Glu1161=')
        assert n.ref_allele == 'Glu'
        assert n.pep_extra == '='
        assert n.format() == 'p.Glu1161='

    def test_no_change_1letter(self):
        """Synonymous 1-letter: p.E1161=."""
        n = HGVSName('p.E1161=')
        assert n.ref_allele == 'E'
        assert n.pep_extra == '='

    def test_format_roundtrip_3letter(self):
        """3-letter substitution round-trips."""
        assert HGVSName('p.Arg132His').format() == 'p.Arg132His'

    def test_format_roundtrip_1letter(self):
        """1-letter substitution round-trips."""
        assert HGVSName('p.R132H').format(use_3letter=False) == 'p.R132H'

    def test_format_1letter_to_3letter(self):
        """1-letter form converts to 3-letter on formatting."""
        assert HGVSName('p.R132H').format(use_3letter=True) == 'p.Arg132His'

    def test_common_oncology_variants(self):
        """Well-known clinical variants parse and format correctly."""
        cases = [
            ('p.V600E', 'V', 'E', 600),
            ('p.G12D', 'G', 'D', 12),
            ('p.K27M', 'K', 'M', 27),
            ('p.Arg132His', 'Arg', 'His', 132),
        ]
        for hgvs_str, ref, alt, pos in cases:
            n = HGVSName(hgvs_str)
            assert n.ref_allele == ref, hgvs_str
            assert n.alt_allele == alt, hgvs_str
            assert n.start == pos, hgvs_str

    def test_equivalence_1letter_vs_3letter(self):
        """1-letter and 3-letter substitutions are semantically equal."""
        assert hgvs_names_equal('p.R132H', 'p.Arg132His')
        assert hgvs_names_equal('p.Arg132*', 'p.Arg132Ter')


# ---------------------------------------------------------------------------
# HGVS protein Deletion (per HGVS stable recommendations)
# https://hgvs-nomenclature.org/stable/recommendations/protein/deletion/
# ---------------------------------------------------------------------------

class TestProteinDeletion:
    """Deletion: one or more amino acids are removed from the sequence."""

    def test_single_residue_3letter(self):
        """Single residue deletion: p.Trp24del."""
        n = HGVSName('p.Trp24del')
        assert n.mutation_type == 'del'
        assert n.start == 24
        assert n.end == 24
        assert n.ref_allele == 'Trp'

    def test_single_residue_1letter(self):
        """Single residue deletion 1-letter: p.W24del."""
        n = HGVSName('p.W24del')
        assert n.mutation_type == 'del'
        assert n.start == 24
        assert n.ref_allele == 'W'

    def test_range_deletion_3letter(self):
        """Range deletion: p.Trp24_Ala26del."""
        n = HGVSName('p.Trp24_Ala26del')
        assert n.mutation_type == 'del'
        assert n.start == 24
        assert n.end == 26
        assert n.ref_allele == 'Trp'
        assert n.ref2_allele == 'Ala'

    def test_range_deletion_1letter(self):
        """Range deletion 1-letter: p.W24_A26del."""
        n = HGVSName('p.W24_A26del')
        assert n.mutation_type == 'del'
        assert n.start == 24
        assert n.end == 26

    def test_format_single_residue_del(self):
        """Single residue deletion formats correctly."""
        assert HGVSName('p.Trp24del').format() == 'p.Trp24del'

    def test_format_range_del(self):
        """Range deletion formats correctly."""
        assert HGVSName('p.Trp24_Ala26del').format() == 'p.Trp24_Ala26del'

    def test_format_1letter_del_as_3letter(self):
        """1-letter deletion converts to 3-letter."""
        assert HGVSName('p.W24del').format(use_3letter=True) == 'p.Trp24del'

    def test_format_1letter_del_round_trip(self):
        """1-letter deletion round-trips."""
        assert HGVSName('p.W24del').format(use_3letter=False) == 'p.W24del'

    def test_format_range_del_1letter(self):
        """Range 1-letter deletion converts to 3-letter."""
        assert HGVSName('p.W24_A26del').format(use_3letter=True) == 'p.Trp24_Ala26del'

    def test_predicted_single_del(self):
        """Predicted single deletion: p.(Trp24del)."""
        n = HGVSName('p.(Trp24del)')
        assert n.predicted is True
        assert n.mutation_type == 'del'
        assert n.format() == 'p.(Trp24del)'

    def test_predicted_range_del(self):
        """Predicted range deletion: p.(Trp24_Ala26del)."""
        n = HGVSName('p.(Trp24_Ala26del)')
        assert n.predicted is True
        assert n.mutation_type == 'del'
        assert n.format() == 'p.(Trp24_Ala26del)'

    def test_with_transcript_prefix(self):
        """Deletion with transcript prefix parses correctly."""
        n = HGVSName('NM_003997.1:p.Trp24del')
        assert n.transcript == 'NM_003997.1'
        assert n.mutation_type == 'del'
        assert n.start == 24


# ---------------------------------------------------------------------------
# HGVS protein Duplication (per HGVS stable recommendations)
# https://hgvs-nomenclature.org/stable/recommendations/protein/duplication/
# ---------------------------------------------------------------------------

class TestProteinDuplication:
    """Duplication: one or more amino acids are duplicated in-frame."""

    def test_single_residue_3letter(self):
        """Single residue duplication: p.Val7dup."""
        n = HGVSName('p.Val7dup')
        assert n.mutation_type == 'dup'
        assert n.start == 7
        assert n.end == 7
        assert n.ref_allele == 'Val'

    def test_single_residue_1letter(self):
        """Single residue duplication 1-letter: p.V7dup."""
        n = HGVSName('p.V7dup')
        assert n.mutation_type == 'dup'
        assert n.start == 7
        assert n.ref_allele == 'V'

    def test_range_duplication_3letter(self):
        """Range duplication: p.Lys23_Val25dup."""
        n = HGVSName('p.Lys23_Val25dup')
        assert n.mutation_type == 'dup'
        assert n.start == 23
        assert n.end == 25
        assert n.ref_allele == 'Lys'
        assert n.ref2_allele == 'Val'

    def test_range_duplication_1letter(self):
        """Range duplication 1-letter: p.K23_V25dup."""
        n = HGVSName('p.K23_V25dup')
        assert n.mutation_type == 'dup'
        assert n.start == 23
        assert n.end == 25

    def test_format_single_residue_dup(self):
        """Single residue duplication formats correctly."""
        assert HGVSName('p.Val7dup').format() == 'p.Val7dup'

    def test_format_range_dup(self):
        """Range duplication formats correctly."""
        assert HGVSName('p.Lys23_Val25dup').format() == 'p.Lys23_Val25dup'

    def test_format_1letter_dup_as_3letter(self):
        """1-letter duplication converts to 3-letter."""
        assert HGVSName('p.V7dup').format(use_3letter=True) == 'p.Val7dup'

    def test_format_1letter_dup_round_trip(self):
        """1-letter duplication round-trips."""
        assert HGVSName('p.V7dup').format(use_3letter=False) == 'p.V7dup'

    def test_format_range_dup_1letter(self):
        """Range 1-letter duplication converts to 3-letter."""
        assert HGVSName('p.K23_V25dup').format(use_3letter=True) == 'p.Lys23_Val25dup'

    def test_predicted_single_dup(self):
        """Predicted single duplication: p.(Val7dup)."""
        n = HGVSName('p.(Val7dup)')
        assert n.predicted is True
        assert n.mutation_type == 'dup'
        assert n.format() == 'p.(Val7dup)'

    def test_predicted_range_dup(self):
        """Predicted range duplication: p.(Lys23_Val25dup)."""
        n = HGVSName('p.(Lys23_Val25dup)')
        assert n.predicted is True
        assert n.mutation_type == 'dup'
        assert n.format() == 'p.(Lys23_Val25dup)'

    def test_with_transcript_prefix(self):
        """Duplication with transcript prefix parses correctly."""
        n = HGVSName('NM_003997.1:p.Val7dup')
        assert n.transcript == 'NM_003997.1'
        assert n.mutation_type == 'dup'
        assert n.start == 7


# ---------------------------------------------------------------------------
# HGVS protein Insertion (per HGVS stable recommendations)
# https://hgvs-nomenclature.org/stable/recommendations/protein/insertion/
# ---------------------------------------------------------------------------

class TestProteinInsertion:
    """Insertion: one or more amino acids inserted between two existing residues."""

    def test_single_aa_insert_3letter(self):
        """Single amino acid insertion: p.His4_Gln5insAla."""
        n = HGVSName('p.His4_Gln5insAla')
        assert n.mutation_type == 'ins'
        assert n.start == 4
        assert n.end == 5
        assert n.ref_allele == 'His'
        assert n.ref2_allele == 'Gln'
        assert n.alt_allele == 'Ala'

    def test_single_aa_insert_1letter(self):
        """Single amino acid insertion 1-letter: p.H4_Q5insA."""
        n = HGVSName('p.H4_Q5insA')
        assert n.mutation_type == 'ins'
        assert n.start == 4
        assert n.end == 5
        assert n.ref_allele == 'H'
        assert n.ref2_allele == 'Q'
        assert n.alt_allele == 'A'

    def test_multi_aa_insert_3letter(self):
        """Multiple amino acid insertion: p.Lys2_Gly3insGlnSerLys."""
        n = HGVSName('p.Lys2_Gly3insGlnSerLys')
        assert n.mutation_type == 'ins'
        assert n.start == 2
        assert n.end == 3
        assert n.alt_allele == 'GlnSerLys'

    def test_insert_with_ter(self):
        """Insertion ending with Ter (stop-gaining): p.His4_Gln5insAlaTer."""
        n = HGVSName('p.His4_Gln5insAlaTer')
        assert n.mutation_type == 'ins'
        assert n.alt_allele == 'AlaTer'

    def test_format_single_ins(self):
        """Single insertion formats correctly."""
        assert HGVSName('p.His4_Gln5insAla').format() == 'p.His4_Gln5insAla'

    def test_format_multi_ins(self):
        """Multi-insertion formats correctly."""
        assert HGVSName('p.Lys2_Gly3insGlnSerLys').format() == 'p.Lys2_Gly3insGlnSerLys'

    def test_format_1letter_ins_as_3letter(self):
        """1-letter insertion converts to 3-letter."""
        assert HGVSName('p.H4_Q5insA').format(use_3letter=True) == 'p.His4_Gln5insAla'

    def test_format_1letter_ins_round_trip(self):
        """1-letter insertion round-trips."""
        assert HGVSName('p.H4_Q5insA').format(use_3letter=False) == 'p.H4_Q5insA'

    def test_predicted_ins(self):
        """Predicted insertion: p.(His4_Gln5insAla)."""
        n = HGVSName('p.(His4_Gln5insAla)')
        assert n.predicted is True
        assert n.mutation_type == 'ins'
        assert n.format() == 'p.(His4_Gln5insAla)'

    def test_with_transcript_prefix(self):
        """Insertion with transcript prefix parses correctly."""
        n = HGVSName('NM_004371.2:p.His4_Gln5insAla')
        assert n.transcript == 'NM_004371.2'
        assert n.mutation_type == 'ins'
        assert n.alt_allele == 'Ala'


# ---------------------------------------------------------------------------
# HGVS protein Deletion-Insertion (per HGVS stable recommendations)
# https://hgvs-nomenclature.org/stable/recommendations/protein/delins/
# ---------------------------------------------------------------------------

class TestProteinDeletionInsertion:
    """Deletion-Insertion: one/more AAs replaced by a different sequence."""

    def test_single_delins_3letter(self):
        """Single residue delins: p.Asn47delinsSerSerTer."""
        n = HGVSName('p.Asn47delinsSerSerTer')
        assert n.mutation_type == 'delins'
        assert n.start == 47
        assert n.end == 47
        assert n.ref_allele == 'Asn'
        assert n.alt_allele == 'SerSerTer'

    def test_single_delins_to_single_aa(self):
        """Single-to-single delins (distinct from substitution due to keyword)."""
        n = HGVSName('p.Asn47delinsGlu')
        assert n.mutation_type == 'delins'
        assert n.start == 47
        assert n.alt_allele == 'Glu'

    def test_single_delins_1letter(self):
        """Single residue delins 1-letter: p.N47delinsST."""
        n = HGVSName('p.N47delinsST')
        assert n.mutation_type == 'delins'
        assert n.start == 47
        assert n.ref_allele == 'N'
        assert n.alt_allele == 'ST'

    def test_single_delins_with_stop(self):
        """Single delins ending in stop: p.N47delinsST*."""
        n = HGVSName('p.N47delinsST*')
        assert n.mutation_type == 'delins'
        assert n.alt_allele == 'ST*'

    def test_range_delins_3letter(self):
        """Range delins: p.Glu125_Ala132delinsGlyLeuHis."""
        n = HGVSName('p.Glu125_Ala132delinsGlyLeuHis')
        assert n.mutation_type == 'delins'
        assert n.start == 125
        assert n.end == 132
        assert n.ref_allele == 'Glu'
        assert n.ref2_allele == 'Ala'
        assert n.alt_allele == 'GlyLeuHis'

    def test_range_delins_1letter(self):
        """Range delins 1-letter: p.E125_A132delinsSTV."""
        n = HGVSName('p.E125_A132delinsSTV')
        assert n.mutation_type == 'delins'
        assert n.start == 125
        assert n.end == 132
        assert n.alt_allele == 'STV'

    def test_format_single_delins(self):
        """Single delins formats correctly."""
        assert HGVSName('p.Asn47delinsSerSerTer').format() == 'p.Asn47delinsSerSerTer'

    def test_format_range_delins(self):
        """Range delins formats correctly."""
        n = HGVSName('p.Glu125_Ala132delinsGlyLeuHis')
        assert n.format() == 'p.Glu125_Ala132delinsGlyLeuHis'

    def test_format_1letter_delins_as_3letter(self):
        """1-letter single delins converts to 3-letter."""
        assert HGVSName('p.N47delinsST').format(use_3letter=True) == 'p.Asn47delinsSerThr'

    def test_format_1letter_delins_round_trip(self):
        """1-letter single delins round-trips."""
        assert HGVSName('p.N47delinsST').format(use_3letter=False) == 'p.N47delinsST'

    def test_predicted_single_delins(self):
        """Predicted single delins: p.(Asn47delinsSerSerTer)."""
        n = HGVSName('p.(Asn47delinsSerSerTer)')
        assert n.predicted is True
        assert n.mutation_type == 'delins'
        assert n.format() == 'p.(Asn47delinsSerSerTer)'

    def test_predicted_range_delins(self):
        """Predicted range delins: p.(Glu125_Ala132delinsGlyLeuHis)."""
        n = HGVSName('p.(Glu125_Ala132delinsGlyLeuHis)')
        assert n.predicted is True
        assert n.mutation_type == 'delins'
        assert n.format() == 'p.(Glu125_Ala132delinsGlyLeuHis)'

    def test_with_transcript_prefix(self):
        """Delins with transcript prefix parses correctly."""
        n = HGVSName('NM_004371.2:p.Asn47delinsSerSerTer')
        assert n.transcript == 'NM_004371.2'
        assert n.mutation_type == 'delins'
        assert n.alt_allele == 'SerSerTer'


# ---------------------------------------------------------------------------
# Predicted-form tests (cross-category)
# ---------------------------------------------------------------------------

class TestProteinPredicted:
    """Predicted (parenthesized) forms across all supported mutation types."""

    def test_predicted_substitution(self):
        n = HGVSName('p.(Trp24Cys)')
        assert n.predicted is True
        assert n.mutation_type == '>'
        assert n.format() == 'p.(Trp24Cys)'

    def test_predicted_deletion_single(self):
        n = HGVSName('p.(Trp24del)')
        assert n.predicted is True
        assert n.mutation_type == 'del'
        assert n.format() == 'p.(Trp24del)'

    def test_predicted_deletion_range(self):
        n = HGVSName('p.(Trp24_Ala26del)')
        assert n.predicted is True
        assert n.format() == 'p.(Trp24_Ala26del)'

    def test_predicted_duplication_single(self):
        n = HGVSName('p.(Val7dup)')
        assert n.predicted is True
        assert n.mutation_type == 'dup'
        assert n.format() == 'p.(Val7dup)'

    def test_predicted_duplication_range(self):
        n = HGVSName('p.(Lys23_Val25dup)')
        assert n.predicted is True
        assert n.format() == 'p.(Lys23_Val25dup)'

    def test_predicted_insertion(self):
        n = HGVSName('p.(His4_Gln5insAla)')
        assert n.predicted is True
        assert n.mutation_type == 'ins'
        assert n.format() == 'p.(His4_Gln5insAla)'

    def test_predicted_delins_single(self):
        n = HGVSName('p.(Asn47delinsSerSerTer)')
        assert n.predicted is True
        assert n.mutation_type == 'delins'
        assert n.format() == 'p.(Asn47delinsSerSerTer)'

    def test_predicted_delins_range(self):
        n = HGVSName('p.(Glu125_Ala132delinsGlyLeuHis)')
        assert n.predicted is True
        assert n.format() == 'p.(Glu125_Ala132delinsGlyLeuHis)'

    def test_non_predicted_flag_false(self):
        """Non-parenthesized names should have predicted=False."""
        for hgvs_str in ('p.Trp24Cys', 'p.Trp24del', 'p.Val7dup',
                         'p.His4_Gln5insAla', 'p.Asn47delinsSerSerTer'):
            n = HGVSName(hgvs_str)
            assert n.predicted is False, hgvs_str


# ---------------------------------------------------------------------------
# normalize_aa_allele: multi-1-letter sequence support
# ---------------------------------------------------------------------------

class TestNormalizeAaAlleleMulti:
    """normalize_aa_allele should handle multi-AA 1-letter sequences."""

    def test_multi_1letter_to_3letter(self):
        """'ST' → 'SerThr'."""
        assert normalize_aa_allele('ST', use_3letter=True) == 'SerThr'

    def test_multi_1letter_with_stop_to_3letter(self):
        """'ST*' → 'SerThrTer'."""
        assert normalize_aa_allele('ST*', use_3letter=True) == 'SerThrTer'

    def test_multi_1letter_round_trip(self):
        """Multi 1-letter stays as-is in 1-letter mode."""
        assert normalize_aa_allele('ST', use_3letter=False) == 'ST'

    def test_multi_3letter_seq_unchanged(self):
        """Existing multi-3-letter sequences still work."""
        assert normalize_aa_allele('GlnSerLys', use_3letter=True) == 'GlnSerLys'

    def test_multi_3letter_to_1letter(self):
        """Multi-3-letter converts to 1-letter."""
        assert normalize_aa_allele('GlnSerLys', use_3letter=False) == 'QSK'


# ---------------------------------------------------------------------------
# Regression: existing behavior is preserved
# ---------------------------------------------------------------------------

class TestProteinRegressions:
    """Ensure all previously working patterns still work correctly."""

    def test_substitution_3letter(self):
        n = HGVSName('p.Arg132His')
        assert n.kind == 'p'
        assert n.start == 132
        assert n.ref_allele == 'Arg'
        assert n.alt_allele == 'His'
        assert n.mutation_type == '>'

    def test_no_change(self):
        n = HGVSName('p.Glu1161=')
        assert n.ref_allele == 'Glu'
        assert n.pep_extra == '='
        assert n.format() == 'p.Glu1161='

    def test_frameshift_range_legacy(self):
        """Old-style range/frameshift notation still parses."""
        n = HGVSName('p.Glu1000_Ser1003?fs')
        assert n.mutation_type == 'delins'
        assert n.start == 1000
        assert n.end == 1003
        assert n.pep_extra == '?fs'
        assert n.format() == 'p.Glu1000_Ser1003?fs'

    def test_substitution_1letter(self):
        n = HGVSName('p.R132H')
        assert n.start == 132
        assert n.ref_allele == 'R'
        assert n.alt_allele == 'H'

    def test_stop_codon_as_alt(self):
        n = HGVSName('p.Arg132*')
        assert n.alt_allele == '*'
        assert n.format() == 'p.Arg132Ter'

    def test_stop_codon_as_ref(self):
        n = HGVSName('p.*132Leu')
        assert n.ref_allele == '*'
        assert n.alt_allele == 'Leu'

    def test_with_transcript_prefix(self):
        n = HGVSName('NM_004380.2:p.Arg132His')
        assert n.transcript == 'NM_004380.2'
        assert n.start == 132

    def test_format_3letter_default(self):
        assert HGVSName('p.Arg132His').format() == 'p.Arg132His'

    def test_format_1letter(self):
        assert HGVSName('p.Arg132His').format(use_3letter=False) == 'p.R132H'

    def test_normalize_1letter_to_3letter(self):
        assert normalize_hgvs_name('p.R132H') == 'p.Arg132His'

    def test_hgvs_equal_1letter_3letter(self):
        assert hgvs_names_equal('p.R132H', 'p.Arg132His')
