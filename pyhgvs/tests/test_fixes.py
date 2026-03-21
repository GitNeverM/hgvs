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
    get_vcf_allele,
    normalize_variant,
    parse_hgvs_name,
)
from ..utils import TranscriptLookup, make_transcript, read_transcripts
from .genome import MockGenome


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


# ---------------------------------------------------------------------------
# 10. get_gene_transcripts
# ---------------------------------------------------------------------------

# RefGene lines for three transcripts of BRCA1 with different spans so that
# sorting by length is testable.  txStart / txEnd values are chosen to give
# distinct spans:
#   NM_007294.3  span = 41277500 - 41196311 = 81189   (medium)
#   NM_007295.4  span = 40000000 - 39900000 = 100000  (longest)
#   NM_007296.3  span = 50000000 - 49990000 = 10000   (shortest)
# All assigned to gene BRCA1.

def _make_refgene_line(tx_id, chrom, strand, tx_start, tx_end, gene,
                       cds_start=None, cds_end=None):
    """Build a minimal but syntactically valid GenePred-extension line."""
    if cds_start is None:
        cds_start = tx_start + 1000
    if cds_end is None:
        cds_end = tx_end - 1000
    # One exon covering the whole transcript
    exon_starts = '%d,' % tx_start
    exon_ends = '%d,' % tx_end
    return (
        '0\t%(id)s\t%(chrom)s\t%(strand)s\t%(start)d\t%(end)d\t'
        '%(cds_start)d\t%(cds_end)d\t1\t%(exon_starts)s\t%(exon_ends)s\t'
        '0\t%(gene)s\tcmpl\tcmpl\t0,' % dict(
            id=tx_id, chrom=chrom, strand=strand,
            start=tx_start, end=tx_end,
            cds_start=cds_start, cds_end=cds_end,
            exon_starts=exon_starts, exon_ends=exon_ends,
            gene=gene)
    )


# Three BRCA1 isoforms with deliberately different spans
_REFGENE_THREE = '\n'.join([
    # NM_007294.3  span = 81189 (medium)
    _make_refgene_line('NM_007294.3', 'chr17', '-', 41196311, 41277500, 'BRCA1'),
    # NM_007295.4  span = 100000 (longest)
    _make_refgene_line('NM_007295.4', 'chr17', '-', 39900000, 40000000, 'BRCA1'),
    # NM_007296.3  span = 10000 (shortest)
    _make_refgene_line('NM_007296.3', 'chr17', '-', 49990000, 50000000, 'BRCA1'),
    # A different gene – should not appear in BRCA1 results
    _make_refgene_line('NM_000059.3', 'chr13', '+', 32889644, 32973809, 'BRCA2'),
])
# MANE summary: NM_007294.3 is MANE Select, NM_007295.4 is MANE Plus Clinical
_MANE_THREE = (
    '#NCBI_GeneID\tEnsembl_Gene\tHGNC_ID\tsymbol\tname\t'
    'RefSeq_nuc\tRefSeq_prot\tEnsembl_nuc\tEnsembl_prot\tMANE_status\n'
    '672\tENSG00000012048\tHGNC:1100\tBRCA1\tBRCA1 select\t'
    'NM_007294.3\tNP_009225.1\tENST00000357654.9\tENSP00000350283.3\tMANE Select\n'
    '672\tENSG00000012048\tHGNC:1100\tBRCA1\tBRCA1 plus clinical\t'
    'NM_007295.4\tNP_009226.2\tENST00000493795.5\tENSP00000417438.2\tMANE Plus Clinical\n'
)


@pytest.fixture
def store_three():
    """TranscriptLookup with three BRCA1 isoforms and MANE annotations."""
    store = TranscriptLookup()
    store.load_refgene(io.StringIO(_REFGENE_THREE))
    store.load_mane_summary(io.StringIO(_MANE_THREE))
    return store


class TestGetGeneTranscripts:
    """Tests for TranscriptLookup.get_gene_transcripts()."""

    # 1. Basic: gene exists → returns all transcripts for that gene
    def test_returns_all_transcripts_for_gene(self, store_three):
        txs = store_three.get_gene_transcripts('BRCA1')
        assert len(txs) == 3

    # 2. Returned objects are Transcript instances by default
    def test_returns_transcript_objects_by_default(self, store_three):
        from ..models import Transcript
        txs = store_three.get_gene_transcripts('BRCA1')
        assert all(isinstance(tx, Transcript) for tx in txs)

    # 3. sort_policy="mane": MANE Select is first
    def test_mane_policy_select_is_first(self, store_three):
        txs = store_three.get_gene_transcripts('BRCA1', sort_policy='mane')
        assert getattr(txs[0], 'is_mane_select', False) is True

    # 4. sort_policy="mane": MANE Plus Clinical is second
    def test_mane_policy_plus_clinical_is_second(self, store_three):
        txs = store_three.get_gene_transcripts('BRCA1', sort_policy='mane')
        assert getattr(txs[1], 'is_mane_plus_clinical', False) is True

    # 5. sort_policy="mane": last transcript is neither MANE Select nor Plus
    def test_mane_policy_rest_is_last(self, store_three):
        txs = store_three.get_gene_transcripts('BRCA1', sort_policy='mane')
        assert not getattr(txs[2], 'is_mane_select', False)
        assert not getattr(txs[2], 'is_mane_plus_clinical', False)

    # 6. sort_policy="longest": sorted by span descending
    def test_longest_policy_descending_span(self, store_three):
        txs = store_three.get_gene_transcripts('BRCA1', sort_policy='longest')
        spans = [
            tx.tx_position.chrom_stop - tx.tx_position.chrom_start
            for tx in txs
        ]
        assert spans == sorted(spans, reverse=True)

    # 7. sort_policy="random": all transcripts returned (no ordering guarantee)
    def test_random_policy_returns_all(self, store_three):
        txs = store_three.get_gene_transcripts('BRCA1', sort_policy='random')
        assert len(txs) == 3

    # 8. return_id=True → list of str (full_name)
    def test_return_id_true_returns_strings(self, store_three):
        ids = store_three.get_gene_transcripts('BRCA1', return_id=True)
        assert all(isinstance(i, str) for i in ids)
        assert len(ids) == 3

    # 9. return_id=True values match full_name (versioned accession)
    def test_return_id_values_are_full_names(self, store_three):
        ids = store_three.get_gene_transcripts('BRCA1', return_id=True)
        expected = {'NM_007294.3', 'NM_007295.4', 'NM_007296.3'}
        assert set(ids) == expected

    # 10. Unknown gene → empty list (no exception)
    def test_unknown_gene_returns_empty_list(self, store_three):
        result = store_three.get_gene_transcripts('UNKNOWN_GENE_XYZ')
        assert result == []

    # 11. Invalid sort_policy → ValueError
    def test_invalid_sort_policy_raises_value_error(self, store_three):
        with pytest.raises(ValueError):
            store_three.get_gene_transcripts('BRCA1', sort_policy='invalid')

    # 12. Results for one gene do not include transcripts from another gene
    def test_does_not_include_other_genes(self, store_three):
        txs = store_three.get_gene_transcripts('BRCA1')
        genes = {tx.gene.name for tx in txs}
        assert genes == {'BRCA1'}

    # 13. mane sort – within same tier, longer transcript comes first.
    #     NM_007296.3 is "other" (tier 2) span=10000; to test intra-tier
    #     ordering we need two non-MANE transcripts.  We build a fresh store.
    def test_mane_policy_intra_tier_longest_first(self):
        refgene = '\n'.join([
            # short non-MANE
            _make_refgene_line('NM_000001.1', 'chr1', '+', 1000, 2000, 'GENE1'),
            # long non-MANE
            _make_refgene_line('NM_000002.1', 'chr1', '+', 1000, 5000, 'GENE1'),
        ])
        store = TranscriptLookup()
        store.load_refgene(io.StringIO(refgene))
        txs = store.get_gene_transcripts('GENE1', sort_policy='mane')
        # Both are "other" tier; longer (span=4000) must come first
        spans = [
            tx.tx_position.chrom_stop - tx.tx_position.chrom_start
            for tx in txs
        ]
        assert spans[0] >= spans[1]


# ---------------------------------------------------------------------------
# 11. normalize_variant: block substitution / MNV padding fix
# ---------------------------------------------------------------------------

class TestNormalizeVariantBlockSubstitution:
    """
    normalize_variant must NOT add a 1 bp anchor when all alleles are
    non-empty after trimming (VCF spec: complex substitutions do not require
    padding).  True insertions/deletions (empty allele) must still be padded.
    """

    def _make_pos(self, chrom_start, chrom_stop):
        from ..models import Position
        return Position(
            chrom='chr14',
            chrom_start=chrom_start,
            chrom_stop=chrom_stop,
            is_forward_strand=True,
        )

    def test_equal_length_delins_no_extra_pad(self):
        """
        Equal-length delins (CGT->TCT) after trimming common suffix 'T'
        gives CG->TC.  No 1 bp anchor should be added because both alleles
        are non-empty.

        This is the regression test for the bug reported with
        NM_006888.6:c.259_261delinsTCT which was producing an incorrect
        extra-padded result.
        """
        from ..variants import NormalizedVariant
        # Simulate get_vcf_allele output: anchor 'C' prepended by
        # get_vcf_coords/get_vcf_allele, giving ref='CCGT', alt='CTCT'.
        pos = self._make_pos(100, 104)
        nv = NormalizedVariant(pos, 'CCGT', ['CTCT'],
                               seq_5p='AAAAC', seq_3p='GGGG')
        # After trim prefix 'C' -> ['CGT','TCT'], trim suffix 'T' -> ['CG','TC'].
        # No 1 bp pad (both alleles non-empty).
        assert nv.alleles == ['CG', 'TC'], (
            "Equal-length delins should yield minimal block substitution, "
            "got %r" % nv.alleles
        )
        assert '1bp pad' not in nv.log

    def test_block_substitution_two_chars_no_pad(self):
        """CG->TC block substitution should not be padded."""
        from ..variants import NormalizedVariant
        pos = self._make_pos(100, 102)
        nv = NormalizedVariant(pos, 'CG', ['TC'],
                               seq_5p='AAAAC', seq_3p='GGGG')
        assert nv.alleles == ['CG', 'TC']
        assert '1bp pad' not in nv.log

    def test_snp_not_padded(self):
        """SNPs must never be padded (unchanged by this fix, regression)."""
        from ..variants import NormalizedVariant
        pos = self._make_pos(100, 101)
        nv = NormalizedVariant(pos, 'C', ['T'],
                               seq_5p='AAAAC', seq_3p='GGGG')
        assert nv.alleles == ['C', 'T']
        assert '1bp pad' not in nv.log

    def test_pure_insertion_still_padded(self):
        """
        A true insertion (ref is empty) must still receive the 1 bp anchor
        base (VCF requirement for indels).
        """
        from ..variants import NormalizedVariant
        pos = self._make_pos(100, 100)
        nv = NormalizedVariant(pos, '', ['TCT'],
                               seq_5p='AAAAC', seq_3p='GGGG')
        # Should be padded: alleles start with same base.
        assert nv.alleles[0] == nv.alleles[1][0], (
            "Insertion anchor base should be the same for ref and alt")
        assert '1bp pad' in nv.log

    def test_pure_deletion_still_padded(self):
        """
        A true deletion (alt is empty after trimming) must still receive the
        1 bp anchor base (VCF requirement for indels).
        """
        from ..variants import NormalizedVariant
        pos = self._make_pos(100, 103)
        nv = NormalizedVariant(pos, 'CGT', [''],
                               seq_5p='AAAAC', seq_3p='GGGG')
        assert '1bp pad' in nv.log
        assert len(nv.alleles[1]) == 1, (
            "After padding, deleted allele should be a single anchor base")

    def test_unequal_delins_still_correct(self):
        """
        A non-equal-length delins (e.g. 3 del, 1 ins) correctly represents
        the event without spurious extra padding.

        Simulate anchor+ref='CCGT', anchor+alt='CT' (3-base ref, 1-base ins,
        both including the 'C' anchor prepended by get_vcf_coords/get_vcf_allele).
        Trimming proceeds: prefix 'C' -> ['CGT','T'] -> suffix 'T' ->
        ['CG', ''].  alt becomes empty, so empty_seq=True and 1 bp padding
        IS applied.  After padding both alleles are non-empty.
        """
        from ..variants import NormalizedVariant
        pos = self._make_pos(100, 104)
        nv = NormalizedVariant(pos, 'CCGT', ['CT'],
                               seq_5p='AAAAC', seq_3p='GGGG')
        assert '' not in nv.alleles, "No allele should remain empty after padding"
        assert '1bp pad' in nv.log

    def test_triallelic_block_substitution_no_pad(self):
        """
        A triallelic variant where all post-trim alleles are non-empty must
        not be padded (regression: the old uniq_starts>1 condition padded
        even valid multi-allelic block substitutions).

        'TGGC' vs 'TGGA' vs 'TGAC': trim 'TG' prefix (added by
        get_vcf_coords/get_vcf_allele) -> 'GC','GA','AC'.  None is empty
        so no 1 bp anchor is added.
        """
        from ..variants import NormalizedVariant
        pos = self._make_pos(100, 104)
        nv = NormalizedVariant(pos, 'TGGC', ['TGGA', 'TGAC'],
                               seq_5p='AAAATG', seq_3p='GGGG')
        assert '1bp pad' not in nv.log
        # All alleles non-empty.
        assert all(a for a in nv.alleles)


# ---------------------------------------------------------------------------
# 12. get_vcf_allele: intermediate output tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared fixture helpers for end-to-end and get_vcf_allele tests
# ---------------------------------------------------------------------------

def _make_fwd_transcript(tx_id='NM_T001.1', chrom='chr1',
                         tx_start=0, tx_end=500,
                         cds_start=0, cds_end=500,
                         gene='TESTGENE'):
    """Return a simple single-exon forward-strand Transcript."""
    return make_transcript({
        'id': tx_id,
        'chrom': chrom,
        'strand': '+',
        'start': tx_start,
        'end': tx_end,
        'cds_start': cds_start,
        'cds_end': cds_end,
        'gene_name': gene,
        'exons': [(tx_start, tx_end)],
        'exon_frames': [0],
    })


def _make_rev_transcript(tx_id='NM_T002.1', chrom='chr1',
                         tx_start=0, tx_end=500,
                         cds_start=0, cds_end=500,
                         gene='TESTGENE'):
    """Return a simple single-exon reverse-strand Transcript."""
    return make_transcript({
        'id': tx_id,
        'chrom': chrom,
        'strand': '-',
        'start': tx_start,
        'end': tx_end,
        'cds_start': cds_start,
        'cds_end': cds_end,
        'gene_name': gene,
        'exons': [(tx_start, tx_end)],
        'exon_frames': [0],
    })


def _genome(*args):
    """Build a MockGenome from alternating key/value arguments.

    Usage::

        _genome(
            ('chr1', 257, 261), 'CCGT',
            ('chr1', 258, 261), 'CGT',
        )

    Each key is a (chrom, 0based_start, 0based_end_exclusive) tuple matching
    the internal format used by MockGenome / MockChromosome.
    """
    it = iter(args)
    lookup = {k: v for k, v in zip(it, it)}
    return MockGenome(lookup=lookup, default_seq='N')


class TestGetVcfAllele:
    """
    Unit tests for get_vcf_allele() verifying the intermediate ref/alt before
    normalize_variant() is called.

    Key design:
      * For equal-length delins / block substitutions:
        get_vcf_allele MUST NOT prepend an anchor base.  The event boundaries
        are returned as-is so normalize_variant can decide whether anchoring is
        needed after trimming.
      * For pure del / ins / dup:
        get_vcf_allele MUST still prepend the anchor base (ref[0]) to the alt
        allele, since the alt (or ref) is inherently empty.
    """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _tx():
        """Forward-strand transcript: c.N  →  genomic N (1-based, cds_start=0)."""
        return _make_fwd_transcript()

    @staticmethod
    def _genome_for(anchor_base, ref_seq, chrom='chr1', anchor_pos=258):
        """
        Build a MockGenome where:
          * 1-based [anchor_pos, anchor_pos+len(ref_seq)] = anchor_base + ref_seq
          * 1-based [anchor_pos+1, anchor_pos+len(ref_seq)] = ref_seq  (no anchor)
        get_genomic_sequence calls genome[chrom][start-1:end] (0-based).
        So 1-based range [258, 261] → 0-based [257, 261).
        """
        end_1based = anchor_pos + len(ref_seq)
        with_anchor_key = (chrom, anchor_pos - 1, end_1based)
        without_anchor_key = (chrom, anchor_pos, end_1based)
        return _genome(
            with_anchor_key, anchor_base + ref_seq,
            without_anchor_key, ref_seq,
        )

    # ------------------------------------------------------------------
    # 1. Equal-length delins: NO anchor in get_vcf_allele output
    # ------------------------------------------------------------------

    def test_equal_length_delins_no_anchor_in_ref(self):
        """
        c.259_261delinsTCT: equal-length (3→3) block substitution.
        After the fix, get_vcf_allele MUST NOT prepend an anchor to ref.
        ref should equal the raw genomic sequence CGT (no extra base).
        """
        tx = self._tx()
        # c.259 → genomic 259 (1-based).  anchor = genomic 258 = 'C'.
        genome = self._genome_for('C', 'CGT')
        hgvs = HGVSName('NM_T001.1:c.259_261delinsTCT')
        chrom, start, end, ref, alt = get_vcf_allele(hgvs, genome, tx)
        assert ref == 'CGT', (
            "For equal-length delins, ref must be the raw 3-base sequence "
            "without a VCF anchor; got %r" % ref)

    def test_equal_length_delins_no_anchor_in_alt(self):
        """
        c.259_261delinsTCT: alt must equal the stated alt sequence 'TCT',
        not be prepended with an anchor base.
        """
        tx = self._tx()
        genome = self._genome_for('C', 'CGT')
        hgvs = HGVSName('NM_T001.1:c.259_261delinsTCT')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, tx)
        assert alt == 'TCT', (
            "For equal-length delins, alt must equal the stated sequence "
            "without an anchor base; got %r" % alt)

    def test_equal_length_delins_start_not_shifted(self):
        """
        c.259_261delinsTCT: the returned start position must be 259 (the
        actual genomic start of the event), not 258 (anchor position).
        """
        tx = self._tx()
        genome = self._genome_for('C', 'CGT')
        hgvs = HGVSName('NM_T001.1:c.259_261delinsTCT')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, tx)
        assert start == 259, (
            "Start must be the event start (259), not the anchor position; "
            "got %d" % start)

    # ------------------------------------------------------------------
    # 2. Deletion-dominant delins: also no premature anchor
    # ------------------------------------------------------------------

    def test_deletion_dominant_delins_no_premature_anchor(self):
        """
        c.259_261delinsT: 3-base range replaced by 1 base.
        get_vcf_allele must NOT add an anchor.  The anchor will be added
        by normalize_variant → _1bp_pad when trimming empties the alt.
        """
        tx = self._tx()
        genome = self._genome_for('C', 'CGT')
        hgvs = HGVSName('NM_T001.1:c.259_261delinsT')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, tx)
        assert ref == 'CGT' and alt == 'T', (
            "Deletion-dominant delins: expected ref='CGT', alt='T'; "
            "got ref=%r, alt=%r" % (ref, alt))

    # ------------------------------------------------------------------
    # 3. Pure deletion: anchor IS required in get_vcf_allele output
    # ------------------------------------------------------------------

    def test_pure_del_has_anchor_in_alt(self):
        """
        c.259_261del: pure deletion — alt is empty, VCF requires anchor.
        get_vcf_allele must still prepend ref[0] to alt.
        """
        tx = self._tx()
        genome = self._genome_for('C', 'CGT')
        hgvs = HGVSName('NM_T001.1:c.259_261del')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, tx)
        assert len(ref) == 4 and alt == ref[0], (
            "Pure del: ref must be anchor+deleted_seq (4 chars), "
            "alt must be anchor only; got ref=%r, alt=%r" % (ref, alt))

    def test_pure_del_start_is_anchor_position(self):
        """
        c.259_261del: start must be 258 (the anchor position), not 259.
        """
        tx = self._tx()
        genome = self._genome_for('C', 'CGT')
        hgvs = HGVSName('NM_T001.1:c.259_261del')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, tx)
        assert start == 258, (
            "Pure del: start must be anchor position (258); got %d" % start)

    # ------------------------------------------------------------------
    # 4. Pure insertion: anchor IS required
    # ------------------------------------------------------------------

    def test_pure_ins_has_anchor_in_alt(self):
        """
        c.259_260insACG: pure insertion — ref is a single anchor base.
        get_vcf_allele must prepend ref[0] to alt.
        """
        tx = self._tx()
        # For c.259_260ins, get_vcf_coords returns an empty interval plus the
        # anchor.  The anchor is at position 259 (genome['chr1'][258:259]).
        genome = _genome(
            ('chr1', 258, 259), 'C',   # anchor only for ins
            ('chr1', 258, 261), 'CGT',  # broader range just in case
        )
        hgvs = HGVSName('NM_T001.1:c.259_260insACG')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, tx)
        assert alt.startswith(ref[0]), (
            "Pure ins: alt must start with the anchor base ref[0]; "
            "got ref=%r, alt=%r" % (ref, alt))

    # ------------------------------------------------------------------
    # 5. genomic delins (no transcript): same behaviour
    # ------------------------------------------------------------------

    def test_genomic_equal_length_delins_no_anchor(self):
        """
        chr1:g.259_261delinsTCT: genomic HGVS, no transcript needed.
        get_vcf_allele must NOT add an anchor for the equal-length case.
        """
        genome = _genome(
            ('chr1', 257, 261), 'CCGT',  # anchor(258) + ref(259-261) in 1-based
            ('chr1', 258, 261), 'CGT',
        )
        hgvs = HGVSName('chr1:g.259_261delinsTCT')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, transcript=None)
        assert ref == 'CGT' and alt == 'TCT', (
            "Genomic equal-length delins: no anchor expected; "
            "got ref=%r, alt=%r" % (ref, alt))

    # ------------------------------------------------------------------
    # 6. Reverse-strand cDNA delins: anchor logic still correct
    # ------------------------------------------------------------------

    def test_rev_strand_equal_length_delins_no_anchor(self):
        """
        Reverse-strand transcript: c.259_261delinsTCT.
        get_ref_alt revcomps the alt to the forward strand; the result must
        still not have a spurious anchor prepended.
        """
        tx = _make_rev_transcript()
        # For a 500-bp reverse-strand transcript:
        # c.259 → genomic 500 - 259 + 1 = 242 (1-based)
        # c.261 → genomic 500 - 261 + 1 = 240
        # After get_coords swap: start=240, end=242
        # get_vcf_coords: start=239, end=242
        # get_genomic_sequence(239, 242) → genome['chr1'][238:242] = 4 chars
        genome = _genome(
            ('chr1', 238, 242), 'CCGT',   # anchor(239) + ref(240-242) in 1-based
            ('chr1', 239, 242), 'CGT',
        )
        hgvs = HGVSName('NM_T002.1:c.259_261delinsTCT')
        _, start, _, ref, alt = get_vcf_allele(hgvs, genome, tx)
        # alt on forward strand = revcomp('TCT') = 'AGA'; no anchor.
        assert ref == 'CGT' and alt == 'AGA', (
            "Rev-strand equal-length delins: expected ref='CGT', alt='AGA'; "
            "got ref=%r, alt=%r" % (ref, alt))


# ---------------------------------------------------------------------------
# 13. parse_hgvs_name: full chain end-to-end tests
# ---------------------------------------------------------------------------

class TestParseHgvsNameEndToEnd:
    """
    Full-chain regression tests for parse_hgvs_name() exercising
    get_vcf_allele() → normalize_variant() with mock transcript and genome.

    The mock transcript uses a simple single-exon forward-strand setup where
    cDNA c.N maps to genomic position N (since cds_start=0 and tx starts at 0).
    """

    @staticmethod
    def _tx():
        return _make_fwd_transcript()

    @staticmethod
    def _genome_for(anchor_base, ref_seq, chrom='chr1', anchor_pos=258):
        end_1based = anchor_pos + len(ref_seq)
        lookup = {
            (chrom, anchor_pos - 1, end_1based): anchor_base + ref_seq,
            (chrom, anchor_pos, end_1based): ref_seq,
        }
        return MockGenome(lookup=lookup, default_seq='N')

    # ------------------------------------------------------------------
    # Main regression: NM_006888.6:c.259_261delinsTCT equivalent
    # ------------------------------------------------------------------

    def test_equal_length_delins_minimal_vcf(self):
        """
        Main regression test.

        c.259_261delinsTCT: CGT → TCT.
        Common suffix 'T' trims to CG → TC.
        Expected VCF: ('chr1', 259, 'CG', 'TC').

        This is the equivalent of the bug reported for
        NM_006888.6:c.259_261delinsTCT which previously produced
        the incorrect padded result ('chr14', 90403941, 'CCG', 'CTC').
        """
        tx = self._tx()
        genome = self._genome_for('C', 'CGT')

        def get_tx(name):
            return tx

        result = parse_hgvs_name(
            'NM_T001.1:c.259_261delinsTCT', genome,
            get_transcript=get_tx, normalize=True)
        assert result == ('chr1', 259, 'CG', 'TC'), (
            "Equal-length delins must produce minimal VCF (no extra anchor); "
            "got %r" % (result,))

    def test_equal_length_delins_no_common_trim(self):
        """
        c.259_261delinsAGA: CGT → AGA (no common prefix or suffix).
        Expected VCF: ('chr1', 259, 'CGT', 'AGA').
        """
        tx = self._tx()
        genome = self._genome_for('C', 'CGT')

        result = parse_hgvs_name(
            'NM_T001.1:c.259_261delinsAGA', genome,
            get_transcript=lambda n: tx, normalize=True)
        assert result == ('chr1', 259, 'CGT', 'AGA'), (
            "delins with no common trim must return the full 3-base result; "
            "got %r" % (result,))

    def test_deletion_dominant_delins_is_anchored(self):
        """
        c.259_261delinsT: CGT → T (deletion-dominant).
        After trimming the common suffix 'T': CG → ''; anchor is added.
        Expected VCF: position 258, anchored deletion representation.
        """
        tx = self._tx()
        # normalize_variant will need 5' flanking to fetch the anchor.
        # The anchor is at genomic position 258 = 'C'.
        # We provide enough sequence around the site.
        lookup = {
            ('chr1', 257, 261): 'CCGT',  # anchor=C, ref=CGT
            ('chr1', 258, 261): 'CGT',
            # 5' flanking for _1bp_pad (30 bases before position 258)
            ('chr1', 228, 258): 'C' * 30,
        }
        genome = MockGenome(lookup=lookup, default_seq='N')

        result = parse_hgvs_name(
            'NM_T001.1:c.259_261delinsT', genome,
            get_transcript=lambda n: tx, normalize=True)
        # Result must be a deletion; start < 259 (anchored)
        chrom, start, ref, alt = result
        assert chrom == 'chr1', "chrom mismatch"
        assert start < 259, (
            "Deletion-dominant delins must be left-anchored (start < 259); "
            "got start=%d" % start)
        assert len(ref) > len(alt), (
            "Deletion-dominant delins must have len(ref) > len(alt); "
            "got ref=%r, alt=%r" % (ref, alt))

    def test_snp_unchanged(self):
        """
        c.259A>T: simple SNP.  Unaffected by any of the delins fixes.
        Expected VCF: ('chr1', 259, 'C', 'T').  (ref from genome = 'C')
        """
        tx = self._tx()
        genome = _genome(
            ('chr1', 258, 259), 'C',   # genomic 259 = 'C'
            ('chr1', 257, 261), 'CCGT',
        )
        result = parse_hgvs_name(
            'NM_T001.1:c.259C>T', genome,
            get_transcript=lambda n: tx, normalize=True)
        assert result == ('chr1', 259, 'C', 'T'), (
            "SNP result must be unchanged; got %r" % (result,))

    def test_pure_deletion_still_correct(self):
        """
        c.259_261del: pure deletion.  The fix must not break standard dels.
        """
        tx = self._tx()
        lookup = {
            ('chr1', 257, 261): 'CCGT',
            ('chr1', 258, 261): 'CGT',
            ('chr1', 228, 258): 'C' * 30,
        }
        genome = MockGenome(lookup=lookup, default_seq='N')

        result = parse_hgvs_name(
            'NM_T001.1:c.259_261del', genome,
            get_transcript=lambda n: tx, normalize=True)
        chrom, start, ref, alt = result
        assert chrom == 'chr1'
        assert len(ref) > len(alt), (
            "Pure deletion must have len(ref) > len(alt); "
            "got ref=%r, alt=%r" % (ref, alt))
        assert start < 259, "Pure deletion must be left-anchored"
