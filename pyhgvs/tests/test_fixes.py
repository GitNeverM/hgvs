"""
Tests for HGVS fixes: dup representation, inversion parsing,
Ensembl transcript lookup, and get_ref_alt correctness.
"""
from __future__ import unicode_literals

import io
import pytest

from .. import (
    CDNACoord,
    CDNA_STOP_CODON,
    HGVSName,
    HGVSParseError,
)
from ..utils import TranscriptLookup, read_transcripts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(name):
    """Return an HGVSName parsed from *name*."""
    return HGVSName(name)


# ---------------------------------------------------------------------------
# 1. Duplication: internal representation (alt_allele == ref_allele)
# ---------------------------------------------------------------------------

class TestDupRepresentation:
    """The alt_allele for dup variants equals ref_allele (not ref*2)."""

    def test_cdna_single_base_dup_alt_equals_ref(self):
        h = _parse('BRCA1:c.101dupA')
        assert h.mutation_type == 'dup'
        assert h.ref_allele == 'A'
        assert h.alt_allele == 'A'

    def test_cdna_range_dup_alt_equals_ref(self):
        h = _parse('BRCA1:c.1000_1002dupATG')
        assert h.mutation_type == 'dup'
        assert h.ref_allele == 'ATG'
        assert h.alt_allele == 'ATG'

    def test_genomic_single_base_dup_alt_equals_ref(self):
        h = _parse('BRCA1:g.101dupA')
        assert h.mutation_type == 'dup'
        assert h.ref_allele == 'A'
        assert h.alt_allele == 'A'

    def test_genomic_range_dup_alt_equals_ref(self):
        h = _parse('BRCA1:g.1000_1002dupATG')
        assert h.mutation_type == 'dup'
        assert h.ref_allele == 'ATG'
        assert h.alt_allele == 'ATG'

    def test_cdna_dup_no_sequence_alt_empty(self):
        """dup without explicit sequence: both alleles are empty."""
        h = _parse('BRCA1:c.101dup')
        assert h.mutation_type == 'dup'
        assert h.ref_allele == ''
        assert h.alt_allele == ''

    def test_cdna_range_dup_no_sequence_alt_empty(self):
        h = _parse('BRCA1:c.1000_1002dup')
        assert h.mutation_type == 'dup'
        assert h.ref_allele == ''
        assert h.alt_allele == ''


# ---------------------------------------------------------------------------
# 2. get_ref_alt() for dup variants
# ---------------------------------------------------------------------------

class TestGetRefAlt:
    """get_ref_alt() correctly exposes dup as an insertion."""

    def test_single_base_dup_forward(self):
        h = _parse('BRCA1:c.101dupA')
        ref, alt = h.get_ref_alt(is_forward_strand=True)
        assert ref == ''
        assert alt == 'A'

    def test_range_dup_forward(self):
        h = _parse('BRCA1:c.1000_1002dupATG')
        ref, alt = h.get_ref_alt(is_forward_strand=True)
        assert ref == ''
        assert alt == 'ATG'

    def test_range_dup_reverse_strand(self):
        """On the reverse strand the duplicated sequence is rev-complemented."""
        h = _parse('BRCA1:c.1000_1002dupATG')
        ref, alt = h.get_ref_alt(is_forward_strand=False)
        assert ref == ''
        assert alt == 'CAT'  # revcomp('ATG')

    def test_dup_no_sequence(self):
        """Without explicit sequence, both ref and alt are empty."""
        h = _parse('BRCA1:c.101dup')
        ref, alt = h.get_ref_alt(is_forward_strand=True)
        assert ref == ''
        assert alt == ''

    def test_genomic_range_dup(self):
        h = _parse('BRCA1:g.1000_1002dupATG')
        ref, alt = h.get_ref_alt(is_forward_strand=True)
        assert ref == ''
        assert alt == 'ATG'

    def test_del_unchanged(self):
        """Non-dup variants are not altered by dup logic."""
        h = _parse('BRCA1:c.101delA')
        ref, alt = h.get_ref_alt(is_forward_strand=True)
        assert ref == 'A'
        assert alt == ''

    def test_snp_unchanged(self):
        h = _parse('BRCA1:c.101A>C')
        ref, alt = h.get_ref_alt(is_forward_strand=True)
        assert ref == 'A'
        assert alt == 'C'


# ---------------------------------------------------------------------------
# 3. Format/round-trip for dup
# ---------------------------------------------------------------------------

class TestDupFormat:
    """Formatting dup variants produces the correct HGVS string."""

    def test_format_cdna_single_dup(self):
        h = _parse('BRCA1:c.101dupA')
        assert h.format() == 'BRCA1:c.101dupA'

    def test_format_cdna_range_dup(self):
        h = _parse('BRCA1:c.1000_1002dupATG')
        assert h.format() == 'BRCA1:c.1000_1002dupATG'

    def test_format_genomic_single_dup(self):
        h = _parse('BRCA1:g.101dupA')
        assert h.format() == 'BRCA1:g.101dupA'

    def test_format_genomic_range_dup(self):
        h = _parse('BRCA1:g.1000_1002dupATG')
        assert h.format() == 'BRCA1:g.1000_1002dupATG'


# ---------------------------------------------------------------------------
# 4. Inversion parsing (new pattern)
# ---------------------------------------------------------------------------

class TestInversionParsing:
    """c.Xdup_Yinv and g.X_Yinv are parsed correctly."""

    def test_cdna_inversion_parsed(self):
        h = _parse('BRCA1:c.1000_1002inv')
        assert h.mutation_type == 'inv'
        assert h.kind == 'c'
        assert h.cdna_start == CDNACoord(1000)
        assert h.cdna_end == CDNACoord(1002)

    def test_cdna_inversion_formatted(self):
        h = _parse('BRCA1:c.1000_1002inv')
        assert h.format() == 'BRCA1:c.1000_1002inv'

    def test_genomic_inversion_parsed(self):
        h = _parse('BRCA1:g.1000_1002inv')
        assert h.mutation_type == 'inv'
        assert h.kind == 'g'
        assert h.start == 1000
        assert h.end == 1002

    def test_genomic_inversion_formatted(self):
        h = _parse('BRCA1:g.1000_1002inv')
        assert h.format() == 'BRCA1:g.1000_1002inv'

    def test_cdna_intronic_inversion(self):
        h = _parse('NM_007294.3:c.1000+5_1000+10inv')
        assert h.mutation_type == 'inv'
        assert h.cdna_start == CDNACoord(1000, 5)
        assert h.cdna_end == CDNACoord(1000, 10)


# ---------------------------------------------------------------------------
# 5. cDNA / genomic dup-without-sequence parsing
# ---------------------------------------------------------------------------

class TestDupNoSequence:
    """dup variants without an explicit allele sequence parse correctly."""

    def test_cdna_single_dup_no_seq(self):
        h = _parse('BRCA1:c.101dup')
        assert h.mutation_type == 'dup'
        assert h.cdna_start == CDNACoord(101)
        assert h.cdna_end == CDNACoord(101)
        assert h.ref_allele == ''
        assert h.alt_allele == ''

    def test_cdna_range_dup_no_seq(self):
        h = _parse('BRCA1:c.1000_1002dup')
        assert h.mutation_type == 'dup'
        assert h.cdna_start == CDNACoord(1000)
        assert h.cdna_end == CDNACoord(1002)
        assert h.ref_allele == ''

    def test_genomic_single_dup_no_seq(self):
        h = _parse('BRCA1:g.101dup')
        assert h.mutation_type == 'dup'
        assert h.start == 101
        assert h.end == 101
        assert h.ref_allele == ''

    def test_genomic_range_dup_no_seq(self):
        h = _parse('BRCA1:g.1000_1002dup')
        assert h.mutation_type == 'dup'
        assert h.start == 1000
        assert h.end == 1002
        assert h.ref_allele == ''

    def test_intronic_dup_no_seq(self):
        h = _parse('NM_007294.3:c.1000+5_1000+6dup')
        assert h.mutation_type == 'dup'
        assert h.cdna_start == CDNACoord(1000, 5)
        assert h.cdna_end == CDNACoord(1000, 6)


# ---------------------------------------------------------------------------
# 6. HGVS c. and g. variant parsing completeness
# ---------------------------------------------------------------------------

class TestCdnaVariantParsing:
    """Comprehensive tests for cDNA allele patterns."""

    @pytest.mark.parametrize('hgvs_str,expected', [
        # Substitutions
        ('BRCA1:c.1A>G',      {'mutation_type': '>', 'ref_allele': 'A', 'alt_allele': 'G'}),
        ('BRCA1:c.-1A>G',     {'mutation_type': '>', 'cdna_start': CDNACoord(-1)}),
        ('BRCA1:c.*1A>G',     {'mutation_type': '>', 'cdna_start': CDNACoord(1, landmark=CDNA_STOP_CODON)}),
        ('BRCA1:c.1+1A>G',    {'mutation_type': '>', 'cdna_start': CDNACoord(1, 1)}),
        ('BRCA1:c.1-1A>G',    {'mutation_type': '>', 'cdna_start': CDNACoord(1, -1)}),
        # Deletions
        ('BRCA1:c.1delA',     {'mutation_type': 'del', 'ref_allele': 'A'}),
        ('BRCA1:c.1del',      {'mutation_type': 'del', 'ref_allele': ''}),
        ('BRCA1:c.1_3delATG', {'mutation_type': 'del', 'ref_allele': 'ATG'}),
        ('BRCA1:c.1_3del',    {'mutation_type': 'del', 'ref_allele': ''}),
        # Insertions
        ('BRCA1:c.1_2insATG', {'mutation_type': 'ins', 'alt_allele': 'ATG'}),
        # Duplications with sequence
        ('BRCA1:c.1dupA',     {'mutation_type': 'dup', 'ref_allele': 'A', 'alt_allele': 'A'}),
        ('BRCA1:c.1_3dupATG', {'mutation_type': 'dup', 'ref_allele': 'ATG'}),
        # Duplications without sequence
        ('BRCA1:c.1dup',      {'mutation_type': 'dup', 'ref_allele': ''}),
        ('BRCA1:c.1_3dup',    {'mutation_type': 'dup', 'ref_allele': ''}),
        # DelIns
        ('BRCA1:c.1delAinsG', {'mutation_type': 'delins', 'ref_allele': 'A', 'alt_allele': 'G'}),
        ('BRCA1:c.1delinsG',  {'mutation_type': 'delins', 'alt_allele': 'G'}),
        # No change
        ('BRCA1:c.1A=',       {'mutation_type': '=', 'ref_allele': 'A'}),
        # Inversion
        ('BRCA1:c.1_3inv',    {'mutation_type': 'inv'}),
    ])
    def test_cdna_variant(self, hgvs_str, expected):
        h = _parse(hgvs_str)
        for key, val in expected.items():
            assert getattr(h, key) == val, (
                "For %r, expected %s=%r but got %r" % (hgvs_str, key, val, getattr(h, key)))


class TestGenomicVariantParsing:
    """Comprehensive tests for genomic allele patterns."""

    @pytest.mark.parametrize('hgvs_str,expected', [
        # Substitution
        ('BRCA1:g.100A>G',     {'mutation_type': '>', 'ref_allele': 'A', 'alt_allele': 'G'}),
        # Deletion
        ('BRCA1:g.100delA',    {'mutation_type': 'del', 'ref_allele': 'A'}),
        ('BRCA1:g.100del',     {'mutation_type': 'del'}),
        ('BRCA1:g.100_102del', {'mutation_type': 'del', 'start': 100, 'end': 102}),
        # Insertion
        ('BRCA1:g.100_101insATG', {'mutation_type': 'ins', 'alt_allele': 'ATG'}),
        # Duplication with/without sequence
        ('BRCA1:g.100dupA',    {'mutation_type': 'dup', 'ref_allele': 'A', 'alt_allele': 'A'}),
        ('BRCA1:g.100dup',     {'mutation_type': 'dup', 'ref_allele': ''}),
        ('BRCA1:g.100_102dupATG', {'mutation_type': 'dup', 'ref_allele': 'ATG', 'alt_allele': 'ATG'}),
        ('BRCA1:g.100_102dup', {'mutation_type': 'dup'}),
        # Inversion
        ('BRCA1:g.100_102inv', {'mutation_type': 'inv', 'start': 100, 'end': 102}),
        # DelIns
        ('BRCA1:g.100delAinsG', {'mutation_type': 'delins'}),
        ('BRCA1:g.100delinsG',  {'mutation_type': 'delins', 'alt_allele': 'G'}),
        # No change
        ('BRCA1:g.100A=',       {'mutation_type': '=', 'ref_allele': 'A'}),
    ])
    def test_genomic_variant(self, hgvs_str, expected):
        h = _parse(hgvs_str)
        for key, val in expected.items():
            assert getattr(h, key) == val, (
                "For %r, expected %s=%r but got %r" % (hgvs_str, key, val, getattr(h, key)))


class TestProteinVariantParsing:
    """Tests for protein allele patterns."""

    @pytest.mark.parametrize('hgvs_str,expected', [
        # No change
        ('NM_004380.2:p.Arg132=',
         {'mutation_type': '>', 'ref_allele': 'Arg', 'start': 132, 'pep_extra': '='}),
        ('NM_004380.2:p.R132=',
         {'mutation_type': '>', 'ref_allele': 'R', 'start': 132, 'pep_extra': '='}),
        # Missense
        ('NM_004380.2:p.Arg132His',
         {'mutation_type': '>', 'ref_allele': 'Arg', 'alt_allele': 'His', 'start': 132}),
        ('NM_004380.2:p.R132H',
         {'mutation_type': '>', 'ref_allele': 'R', 'alt_allele': 'H', 'start': 132}),
        # Nonsense (stop codon)
        ('NM_004380.2:p.Arg132Ter',
         {'mutation_type': '>', 'ref_allele': 'Arg', 'alt_allele': 'Ter', 'start': 132}),
        ('NM_004380.2:p.Arg132*',
         {'mutation_type': '>', 'ref_allele': 'Arg', 'alt_allele': '*', 'start': 132}),
        # Frameshift range
        ('NM_004380.2:p.Glu1161_Ser1164?fs',
         {'mutation_type': 'delins', 'start': 1161, 'end': 1164, 'pep_extra': '?fs'}),
        # Frameshift single (now correctly parsed as mutation_type='fs')
        ('NM_004380.2:p.Glu1161fs',
         {'mutation_type': 'fs', 'ref_allele': 'Glu', 'start': 1161,
          'fs_new_aa': '', 'fs_stop': None}),
    ])
    def test_protein_variant(self, hgvs_str, expected):
        h = _parse(hgvs_str)
        for key, val in expected.items():
            assert getattr(h, key) == val, (
                "For %r, expected %s=%r but got %r" % (hgvs_str, key, val, getattr(h, key)))


# ---------------------------------------------------------------------------
# 7. Ensembl transcript lookup in TranscriptLookup
# ---------------------------------------------------------------------------

# Minimal RefGene (GenePred-extension) format line for the test transcript.
# Fields: bin, name, chrom, strand, txStart, txEnd, cdsStart, cdsEnd,
#         exonCount, exonStarts, exonEnds, score, name2, cdsStartStat,
#         cdsEndStat, exonFrames.
# This represents NM_007294.3 (BRCA1) on chr17 minus strand.
_REFGENE_NM = (
    '899\tNM_007294.3\tchr17\t-\t41196311\t41277500\t41197694\t41276113\t23\t'
    '41196311,41199659,41201137,41203079,41209068,41215349,41215890,'
    '41219624,41222944,41226347,41228504,41234420,41242960,41243451,'
    '41247862,41249260,41251791,41256138,41256884,41258472,41267742,'
    '41276033,41277287,\t'
    '41197819,41199720,41201211,41203134,41209152,41215390,41215968,'
    '41219712,41223255,41226538,41228631,41234592,41243049,41246877,'
    '41247939,41249306,41251897,41256278,41256973,41258550,41267796,'
    '41276132,41277500,\t'
    '0\tBRCA1\tcmpl\tcmpl\t1,0,1,0,0,1,1,0,1,2,1,0,1,1,2,1,0,1,2,2,2,0,-1,'
)

# Minimal MANE summary TSV (header + one data row) mapping NM_007294.3 →
# ENST00000357654.9 as MANE Select.
_MANE_SUMMARY = (
    '#NCBI_GeneID\tEnsembl_Gene\tHGNC_ID\tsymbol\tname\t'
    'RefSeq_nuc\tRefSeq_prot\tEnsembl_nuc\tEnsembl_prot\tMANE_status\n'
    '672\tENSG00000012048\tHGNC:1100\tBRCA1\tBRCA1 DNA repair...\t'
    'NM_007294.3\tNP_009225.1\tENST00000357654.9\tENSP00000350283.3\tMANE Select\n'
)


@pytest.fixture
def store_with_mane():
    store = TranscriptLookup()
    store.load_refgene(io.StringIO(_REFGENE_NM))
    store.load_mane_summary(io.StringIO(_MANE_SUMMARY))
    return store


class TestEnsemblLookup:
    """TranscriptLookup supports ENST accession retrieval."""

    def test_refseq_lookup_still_works(self, store_with_mane):
        tx = store_with_mane.get('NM_007294.3')
        assert tx is not None
        assert tx.name == 'NM_007294'

    def test_refseq_versioned_lookup(self, store_with_mane):
        tx = store_with_mane.get('NM_007294.3')
        assert tx is not None
        assert tx.full_name == 'NM_007294.3'

    def test_ensembl_versioned_lookup(self, store_with_mane):
        tx = store_with_mane.get('ENST00000357654.9')
        assert tx is not None
        assert tx.name == 'NM_007294'

    def test_ensembl_bare_lookup(self, store_with_mane):
        tx = store_with_mane.get('ENST00000357654')
        assert tx is not None
        assert tx.name == 'NM_007294'

    def test_get_by_ensembl_versioned(self, store_with_mane):
        tx = store_with_mane.get_by_ensembl('ENST00000357654.9')
        assert tx is not None
        assert tx.name == 'NM_007294'

    def test_get_by_ensembl_bare(self, store_with_mane):
        tx = store_with_mane.get_by_ensembl('ENST00000357654')
        assert tx is not None
        assert tx.name == 'NM_007294'

    def test_ensembl_transcript_attribute_set(self, store_with_mane):
        tx = store_with_mane.get('NM_007294.3')
        assert getattr(tx, 'ensembl_transcript', None) == 'ENST00000357654.9'

    def test_is_mane_select_set(self, store_with_mane):
        tx = store_with_mane.get('NM_007294.3')
        assert getattr(tx, 'is_mane_select', False) is True

    def test_unknown_enst_returns_none(self, store_with_mane):
        tx = store_with_mane.get('ENST99999999999')
        assert tx is None

    def test_enst_in_contains(self, store_with_mane):
        assert 'ENST00000357654' in store_with_mane
        assert 'ENST00000357654.9' in store_with_mane

    def test_enst_not_in_contains(self, store_with_mane):
        assert 'ENST99999999' not in store_with_mane

    def test_get_mane_select(self, store_with_mane):
        tx = store_with_mane.get_mane_select('BRCA1')
        assert tx is not None
        assert tx.name == 'NM_007294'


class TestEnsemblWithoutMane:
    """Without load_mane_summary, ENST lookups return None."""

    def test_enst_returns_none_without_mane(self):
        store = TranscriptLookup()
        store.load_refgene(io.StringIO(_REFGENE_NM))
        assert store.get('ENST00000357654') is None


# ---------------------------------------------------------------------------
# 8. Ensembl prefix in HGVSName.parse_prefix
# ---------------------------------------------------------------------------

class TestEnsemblPrefix:
    """ENST accessions are recognised as transcript prefixes."""

    def test_enst_transcript_prefix(self):
        h = _parse('ENST00000357654:c.2207A>C')
        assert h.transcript == 'ENST00000357654'
        assert h.kind == 'c'

    def test_enst_versioned_transcript_prefix(self):
        h = _parse('ENST00000357654.9:c.2207A>C')
        assert h.transcript == 'ENST00000357654.9'
        assert h.kind == 'c'


# ---------------------------------------------------------------------------
# 9. Misc edge cases
# ---------------------------------------------------------------------------

class TestMiscParsing:
    """Additional edge-case tests."""

    def test_cdna_after_stop_codon(self):
        h = _parse('NM_000492.3:c.*3A>C')
        assert h.cdna_start == CDNACoord(3, 0, CDNA_STOP_CODON)
        assert h.mutation_type == '>'

    def test_intronic_sub(self):
        h = _parse('NM_007294.3:c.1000+5A>G')
        assert h.cdna_start == CDNACoord(1000, 5)
        assert h.mutation_type == '>'

    def test_invalid_hgvs_raises(self):
        with pytest.raises(HGVSParseError):
            _parse('BRCA1:x.101A>G')

    def test_parse_hgvs_name_repr_does_not_crash(self):
        """repr() should not raise even for unusual variants."""
        h = _parse('BRCA1:c.101dupA')
        r = repr(h)
        assert 'BRCA1' in r
