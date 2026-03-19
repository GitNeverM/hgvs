"""
Methods for manipulating HGVS names

Recommendations for the HGVS naming standard:
http://www.hgvs.org/mutnomen/standards.html

Definition of which transcript to use coding variants:
ftp://ftp.ncbi.nih.gov/refseq/H_sapiens/RefSeqGene/LRG_RefSeqGene


HGVS language currently implemented.

HGVS = ALLELE
     | PREFIX_NAME : ALLELE

PREFIX_NAME = TRANSCRIPT
            | TRANSCRIPT '(' GENE ')'

TRANSCRIPT = TRANSCRIPT_NAME
           | TRANSCRIPT_NAME '.' TRANSCRIPT_VERSION

TRANSCRIPT_VERSION = NUMBER

ALLELE = 'c.' CDNA_ALLELE    # cDNA
       | 'g.' GENOMIC_ALLELE # genomic
       | 'm.' MIT_ALLELE     # mitochondrial sequence
       | 'n.' NC_ALLELE      # non-coding RNA reference sequence
       | 'r.' RNA_ALLELE     # RNA sequence (like r.76a>u)
       | 'p.' PROTEIN_ALLELE # protein sequence (like  p.Lys76Asn)

NC_ALLELE =
RNA_ALLELE =
CDNA_ALLELE = CDNA_COORD SINGLE_BASE_CHANGE
            | CDNA_COORD_RANGE MULTI_BASE_CHANGE

GENOMIC_ALLELE =
MIT_ALLELE = COORD SINGLE_BASE_CHANGE
           | COORD_RANGE MULTI_BASE_CHANGE

SINGLE_BASE_CHANGE = CDNA_ALLELE = CDNA_COORD BASE '='        # no change
                   | CDNA_COORD BASE '>' BASE                 # substitution
                   | CDNA_COORD 'ins' BASE                    # 1bp insertion
                   | CDNA_COORD 'del' BASE                    # 1bp deletion
                   | CDNA_COORD 'dup' BASE                    # 1bp duplication
                   | CDNA_COORD 'ins'                         # 1bp insertion
                   | CDNA_COORD 'del'                         # 1bp deletion
                   | CDNA_COORD 'dup'                         # 1bp duplication
                   | CDNA_COORD 'del' BASE 'ins' BASE         # 1bp indel
                   | CDNA_COORD 'delins' BASE                 # 1bp indel

MULTI_BASE_CHANGE = COORD_RANGE 'del' BASES             # deletion
                  | COORD_RANGE 'ins' BASES             # insertion
                  | COORD_RANGE 'dup' BASES             # duplication
                  | COORD_RANGE 'del'                   # deletion
                  | COORD_RANGE 'dup'                   # duplication
                  | COORD_RANGE 'del' BASES 'ins' BASES # indel
                  | COORD_RANGE 'delins' BASES          # indel


AMINO1 = [GAVLIMFWPSTCYNQDEKRH]

AMINO3 = 'Gly' | 'Ala' | 'Val' | 'Leu' | 'Ile' | 'Met' | 'Phe' | 'Trp' | 'Pro'
       | 'Ser' | 'Thr' | 'Cys' | 'Tyr' | 'Asn' | 'Gln' | 'Asp' | 'Glu' | 'Lys'
       | 'Arg' | 'His'

PROTEIN_ALLELE = AMINO3 COORD '='               # no peptide change
               | AMINO1 COORD '='               # no peptide change
               | AMINO3 COORD AMINO3 PEP_EXTRA  # peptide change
               | AMINO1 COORD AMINO1 PEP_EXTRA  # peptide change
               | AMINO3 COORD '_' AMINO3 COORD PEP_EXTRA        # indel
               | AMINO1 COORD '_' AMINO1 COORD PEP_EXTRA        # indel
               | AMINO3 COORD '_' AMINO3 COORD PEP_EXTRA AMINO3 # indel
               | AMINO1 COORD '_' AMINO1 COORD PEP_EXTRA AMINO1 # indel

# A genomic range:
COORD_RANGE = COORD '_' COORD

# A cDNA range:
CDNA_COORD_RANGE = CDNA_COORD '_' CDNA_COORD

# A cDNA coordinate:
CDNA_COORD = COORD_PREFIX COORD
           | COORD_PREFIX COORD OFFSET_PREFIX OFFSET
COORD_PREFIX = '' | '-' | '*'
COORD = NUMBER
OFFSET_PREFIX = '-' | '+'
OFFSET = NUMBER

# Primatives:
NUMBER = \\d+
BASE = [ACGT]
BASES = BASE+

"""
from __future__ import absolute_import
from __future__ import unicode_literals

import re

from .variants import justify_indel
from .variants import normalize_variant
from .variants import revcomp


CHROM_PREFIX = 'chr'
CDNA_START_CODON = 'cdna_start'
CDNA_STOP_CODON = 'cdna_stop'

# ---------------------------------------------------------------------------
# Amino acid conversion tables
# ---------------------------------------------------------------------------

#: Mapping from single-letter amino acid code to three-letter code.
#: Includes standard 20 amino acids plus stop codon (*) and special codes.
AA1_TO_AA3 = {
    'A': 'Ala',
    'C': 'Cys',
    'D': 'Asp',
    'E': 'Glu',
    'F': 'Phe',
    'G': 'Gly',
    'H': 'His',
    'I': 'Ile',
    'K': 'Lys',
    'L': 'Leu',
    'M': 'Met',
    'N': 'Asn',
    'P': 'Pro',
    'Q': 'Gln',
    'R': 'Arg',
    'S': 'Ser',
    'T': 'Thr',
    'V': 'Val',
    'W': 'Trp',
    'Y': 'Tyr',
    '*': 'Ter',  # stop codon
    'X': 'Xaa',  # unknown/any amino acid
    'U': 'Sec',  # selenocysteine
    'O': 'Pyl',  # pyrrolysine
}

#: Mapping from three-letter amino acid code to single-letter code.
AA3_TO_AA1 = {v: k for k, v in AA1_TO_AA3.items()}
# Add common alternative three-letter stop-codon representations
AA3_TO_AA1['Trm'] = '*'
AA3_TO_AA1['Stop'] = '*'


def aa1_to_aa3(aa1):
    """Convert a single-letter amino acid code (or '*') to its three-letter form.

    Args:
        aa1: Single-letter amino acid code string, e.g. ``'R'`` or ``'*'``.

    Returns:
        Three-letter code string, e.g. ``'Arg'`` or ``'Ter'``.

    Raises:
        HGVSInvalidAminoAcidError: If *aa1* is not a recognised code.
    """
    result = AA1_TO_AA3.get(aa1.upper() if aa1 != '*' else '*')
    if result is None:
        raise HGVSInvalidAminoAcidError(aa1)
    return result


def aa3_to_aa1(aa3):
    """Convert a three-letter amino acid code to its single-letter form.

    Args:
        aa3: Three-letter amino acid code string, e.g. ``'Arg'`` or ``'Ter'``.

    Returns:
        Single-letter code string, e.g. ``'R'`` or ``'*'``.

    Raises:
        HGVSInvalidAminoAcidError: If *aa3* is not a recognised code.
    """
    # Normalise capitalisation: first letter upper, rest lower
    normalised = aa3[0].upper() + aa3[1:].lower() if len(aa3) == 3 else aa3
    result = AA3_TO_AA1.get(normalised)
    if result is None:
        raise HGVSInvalidAminoAcidError(aa3)
    return result


def _is_aa1(token):
    """Return True if *token* is a valid single-letter amino acid code."""
    return len(token) == 1 and token in AA1_TO_AA3


def _is_aa3(token):
    """Return True if *token* looks like a three-letter amino acid code."""
    return len(token) == 3 and token[0].isupper() and token[1:].islower()


def normalize_aa_allele(allele, use_3letter=True):
    """Normalise an amino-acid allele string to a consistent notation.

    Handles single-letter codes, three-letter codes, stop codons (``*`` /
    ``Ter``), and concatenated three-letter sequences (e.g. ``GluSer``).

    Args:
        allele: Raw allele string from an HGVS protein name, e.g. ``'R'``,
            ``'Arg'``, ``'GluSer'``, or ``'*'``.
        use_3letter: If ``True`` (default) return the three-letter form;
            otherwise return single-letter.

    Returns:
        Normalised allele string, or the original string if it cannot be
        recognised (to be lenient for downstream handling).
    """
    if not allele:
        return allele

    # Single-letter stop codon
    if allele == '*':
        return 'Ter' if use_3letter else '*'

    # Three-letter stop codon
    if allele in ('Ter', 'Trm', 'Stop'):
        return 'Ter' if use_3letter else '*'

    # Single-letter amino acid
    if _is_aa1(allele):
        return AA1_TO_AA3[allele] if use_3letter else allele

    # Three-letter amino acid
    if _is_aa3(allele):
        if use_3letter:
            return allele[0].upper() + allele[1:].lower()
        result = AA3_TO_AA1.get(allele[0].upper() + allele[1:].lower())
        return result if result is not None else allele

    # Concatenated three-letter sequence, e.g. 'GluSer'
    if len(allele) % 3 == 0 and allele[0].isupper():
        parts = [allele[i:i + 3] for i in range(0, len(allele), 3)]
        if all(_is_aa3(p) for p in parts):
            if use_3letter:
                return ''.join(p[0].upper() + p[1:].lower() for p in parts)
            converted = [AA3_TO_AA1.get(p[0].upper() + p[1:].lower(), p)
                         for p in parts]
            return ''.join(converted)

    # Unknown — return as-is
    return allele


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class HGVSError(ValueError):
    """Base class for all HGVS-related errors in this library."""


class HGVSInvalidAminoAcidError(HGVSError):
    """Raised when an unrecognised amino acid code is encountered."""

    def __init__(self, code):
        super(HGVSInvalidAminoAcidError, self).__init__(
            "Unrecognised amino acid code: %r" % code)
        self.code = code


class HGVSParseError(HGVSError):
    """Raised when an HGVS string cannot be parsed."""

    def __init__(self, name='', part='name', reason=''):
        if name:
            message = 'Invalid HGVS %s "%s"' % (part, name)
        else:
            message = 'Invalid HGVS %s' % part
        if reason:
            message += ': ' + reason
        super(HGVSParseError, self).__init__(message)
        self.name = name
        self.part = part
        self.reason = reason


class HGVSTranscriptError(HGVSError):
    """Raised when a transcript cannot be resolved."""

    def __init__(self, transcript_name='', reason=''):
        if transcript_name:
            message = 'Transcript not found: %r' % transcript_name
        else:
            message = 'Transcript resolution failed'
        if reason:
            message += ': ' + reason
        super(HGVSTranscriptError, self).__init__(message)
        self.transcript_name = transcript_name
        self.reason = reason


class HGVSFormattingError(HGVSError):
    """Raised when an HGVS name cannot be formatted."""

    def __init__(self, reason=''):
        message = 'HGVS formatting failed'
        if reason:
            message += ': ' + reason
        super(HGVSFormattingError, self).__init__(message)
        self.reason = reason


class HGVSNormalizationError(HGVSError):
    """Raised when HGVS normalization fails."""

    def __init__(self, reason=''):
        message = 'HGVS normalization failed'
        if reason:
            message += ': ' + reason
        super(HGVSNormalizationError, self).__init__(message)
        self.reason = reason


class HGVSRegex(object):
    """
    All regular expression for HGVS names.
    """

    # DNA syntax
    # http://www.hgvs.org/mutnomen/standards.html#nucleotide
    BASE = r"[acgtbdhkmnrsvwyACGTBDHKMNRSVWY]|\d+"
    BASES = r"[acgtbdhkmnrsvwyACGTBDHKMNRSVWY]+|\d+"
    DNA_REF = "(?P<ref>" + BASES + ")"
    DNA_ALT = "(?P<alt>" + BASES + ")"

    # Mutation types
    EQUAL = "(?P<mutation_type>=)"
    SUB = "(?P<mutation_type>>)"
    INS = "(?P<mutation_type>ins)"
    DEL = "(?P<mutation_type>del)"
    DUP = "(?P<mutation_type>dup)"
    INV = "(?P<mutation_type>inv)"

    # Simple coordinate syntax
    COORD_START = r"(?P<start>\d+)"
    COORD_END = r"(?P<end>\d+)"
    COORD_RANGE = COORD_START + "_" + COORD_END

    # cDNA coordinate syntax
    CDNA_COORD = (r"(?P<coord_prefix>|-|\*)(?P<coord>\d+)"
                  r"((?P<offset_prefix>-|\+)(?P<offset>\d+))?")
    CDNA_START = (r"(?P<start>(?P<start_coord_prefix>|-|\*)(?P<start_coord>\d+)"
                  r"((?P<start_offset_prefix>-|\+)(?P<start_offset>\d+))?)")
    CDNA_END = (r"(?P<end>(?P<end_coord_prefix>|-|\*)(?P<end_coord>\d+)"
                r"((?P<end_offset_prefix>-|\+)(?P<end_offset>\d+))?)")
    CDNA_RANGE = CDNA_START + "_" + CDNA_END

    # cDNA allele syntax
    CDNA_ALLELE = [
        # No change
        CDNA_START + DNA_REF + EQUAL,

        # Substitution
        CDNA_START + DNA_REF + SUB + DNA_ALT,

        # 1bp insertion, deletion, duplication
        CDNA_START + INS + DNA_ALT,
        CDNA_START + DEL + DNA_REF,
        CDNA_START + DUP + DNA_REF,
        CDNA_START + DEL,
        CDNA_START + DUP,

        # Insertion, deletion, duplication
        CDNA_RANGE + INS + DNA_ALT,
        CDNA_RANGE + DEL + DNA_REF,
        CDNA_RANGE + DUP + DNA_REF,
        CDNA_RANGE + DEL,
        CDNA_RANGE + DUP,

        # Inversion (range required; sequence is implied to be complement)
        CDNA_RANGE + INV,

        # Indels
        "(?P<delins>" + CDNA_START + 'del' + DNA_REF + 'ins' + DNA_ALT + ")",
        "(?P<delins>" + CDNA_RANGE + 'del' + DNA_REF + 'ins' + DNA_ALT + ")",
        "(?P<delins>" + CDNA_START + 'delins' + DNA_ALT + ")",
        "(?P<delins>" + CDNA_RANGE + 'delins' + DNA_ALT + ")",
    ]

    CDNA_ALLELE_REGEXES = [re.compile("^" + regex + "$")
                           for regex in CDNA_ALLELE]

    # Peptide syntax
    # 3-letter amino acid code (any capitalised 3-char sequence, e.g. Glu, Ser)
    PEP3 = r"(?:[A-Z][a-z]{2})+"
    # 1-letter amino acid code or stop codon (*)
    PEP1 = r"[ACDEFGHIKLMNPQRSTVWYX*]"
    # Combined: tries 3-letter first, then 1-letter
    PEP = "(?:" + PEP3 + "|" + PEP1 + ")"

    # Keep the old PEP name as an alias so subclasses/external code still works
    PEP_REF = "(?P<ref>" + PEP + ")"
    PEP_REF2 = "(?P<ref2>" + PEP + ")"
    PEP_ALT = "(?P<alt>" + PEP + ")"

    # PEP_EXTRA matches the optional suffix that follows a protein allele:
    #   =     synonymous (no change)
    #   ?fs   uncertain frameshift
    #   ?     uncertain consequence
    #   fs    frameshift (without uncertainty marker)
    # The alternatives are ordered longest-first so that '?fs' is preferred
    # over bare '?' when both could match.
    PEP_EXTRA = r"(?P<extra>(?:\?fs|=|\?|fs)?)"

    # ---------------------------------------------------------------------------
    # Extended peptide allele building blocks for HGVS stable recommendations
    # ---------------------------------------------------------------------------

    # Frameshift terminator suffix: Ter23 / *23 / Ter? / *?  (optional)
    # Captured into group 'ter_pos' (just the number or '?', not the Ter/*)
    _PEP_FS_TER = r"(?:(?:Ter|\*)(?P<ter_pos>\d+|\?))?"

    # Multi-amino-acid sequence used in insertion / delins alt alleles.
    # Matches one or more PEP tokens (greedy), or a bare stop codon (*),
    # or an unknown-count bracket such as [5] or [?].
    _PEP_SEQ = r"(?:" + PEP + r")+"
    _PEP_INS_SEQ = r"(?P<alt>" + _PEP_SEQ + r"|\[(?:\d+|\?)\])"
    _PEP_DELINS_SEQ = r"(?P<alt>" + _PEP_SEQ + r")"

    # Extension offset for N-terminal extensions: -5, +5, ?
    _PEP_EXT_OFFSET = r"(?P<ext_offset>-?\d+|\?)"

    # New amino acid and optional new-stop for C-terminal extensions
    _PEP_EXT_AA = r"(?P<ext_aa>" + PEP + r")"
    _PEP_EXT_TER = r"(?:(?:Ter|\*)(?P<ext_ter_pos>\d+|\?))?"

    # Peptide allele syntax — ordered from most-specific to most-general so
    # that longer / more-constrained patterns match before the catch-all ones.
    PEP_ALLELE = [
        # === Frameshift (must precede substitution / no-change) ===
        # With new first amino acid:
        #   p.Arg97ProfsTer23  p.Arg97Profs*23  p.Ile327Argfs*?
        (PEP_REF + COORD_START
         + r"(?P<new_aa>" + PEP + r")fs"
         + _PEP_FS_TER),
        # Without new amino acid:
        #   p.Arg97fs  p.Arg97fsTer23  p.Arg97fs*?
        (PEP_REF + COORD_START + r"fs" + _PEP_FS_TER),

        # === No change (synonymous) ===
        # Example: Glu1161=  /  E1161=
        PEP_REF + COORD_START + PEP_EXTRA,

        # === Substitution (single residue) ===
        # Example: Glu1161Ser  /  R132H  /  Arg132*
        PEP_REF + COORD_START + PEP_ALT + PEP_EXTRA,

        # === Range change (backward-compatible: covers range frameshifts, etc.) ===
        # Example: Glu1161_Ser1164?fs
        ("(?P<delins>"
         + PEP_REF + COORD_START + "_" + PEP_REF2 + COORD_END
         + PEP_EXTRA + ")"),
        ("(?P<delins>"
         + PEP_REF + COORD_START + "_" + PEP_REF2 + COORD_END
         + PEP_ALT + PEP_EXTRA + ")"),

        # === Deletion ===
        # Single residue: p.Lys23del
        "(?P<pep_del>" + PEP_REF + COORD_START + r"del)",
        # Range:          p.Lys23_Val25del
        ("(?P<pep_del>"
         + PEP_REF + COORD_START + "_" + PEP_REF2 + COORD_END
         + r"del)"),

        # === Duplication ===
        # Single residue: p.Lys23dup
        "(?P<pep_dup>" + PEP_REF + COORD_START + r"dup)",
        # Range:          p.Lys23_Val25dup
        ("(?P<pep_dup>"
         + PEP_REF + COORD_START + "_" + PEP_REF2 + COORD_END
         + r"dup)"),

        # === Insertion ===
        # p.Lys23_Leu24insArgSerGln  /  p.Lys23_Leu24ins*  /  p.Lys23_Leu24ins[5]
        ("(?P<pep_ins>"
         + PEP_REF + COORD_START + "_" + PEP_REF2 + COORD_END
         + r"ins" + _PEP_INS_SEQ + ")"),

        # === Deletion-Insertion (delins) ===
        # Single residue: p.Cys28delinsTrpVal
        ("(?P<pep_delins>"
         + PEP_REF + COORD_START
         + r"delins" + _PEP_DELINS_SEQ + ")"),
        # Range:          p.Cys28_Lys29delinsTrp
        ("(?P<pep_delins>"
         + PEP_REF + COORD_START + "_" + PEP_REF2 + COORD_END
         + r"delins" + _PEP_DELINS_SEQ + ")"),

        # === Extension ===
        # N-terminal: p.Met1ext-5  /  p.Met1ext?
        ("(?P<pep_ext>" + PEP_REF + COORD_START
         + r"ext" + _PEP_EXT_OFFSET + ")"),
        # C-terminal: p.*110GlnextTer17  /  p.Ter110GlnextTer17
        ("(?P<pep_ext>"
         + PEP_REF + COORD_START + _PEP_EXT_AA
         + r"ext" + _PEP_EXT_TER + ")"),

        # === Repeated sequences ===
        # Single residue: p.Ala2[10]  /  p.Ala2[?]
        PEP_REF + COORD_START + r"\[(?P<rep_count>\d+|\?)\]",
        # Range:          p.Ala2_Pro5[10]
        (PEP_REF + COORD_START + "_" + PEP_REF2 + COORD_END
         + r"\[(?P<rep_count>\d+|\?)\]"),
    ]

    PEP_ALLELE_REGEXES = [re.compile("^" + regex + "$")
                          for regex in PEP_ALLELE]

    # Genomic allele syntax
    GENOMIC_ALLELE = [
        # No change
        COORD_START + DNA_REF + EQUAL,

        # Substitution
        COORD_START + DNA_REF + SUB + DNA_ALT,

        # 1bp insertion, deletion, duplication
        COORD_START + INS + DNA_ALT,
        COORD_START + DEL + DNA_REF,
        COORD_START + DUP + DNA_REF,
        COORD_START + DEL,
        COORD_START + DUP,

        # Insertion, deletion, duplication
        COORD_RANGE + INS + DNA_ALT,
        COORD_RANGE + DEL + DNA_REF,
        COORD_RANGE + DUP + DNA_REF,
        COORD_RANGE + DEL,
        COORD_RANGE + DUP,

        # Inversion
        COORD_RANGE + INV,

        # Indels
        "(?P<delins>" + COORD_START + 'del' + DNA_REF + 'ins' + DNA_ALT + ")",
        "(?P<delins>" + COORD_RANGE + 'del' + DNA_REF + 'ins' + DNA_ALT + ")",
        "(?P<delins>" + COORD_START + 'delins' + DNA_ALT + ")",
        "(?P<delins>" + COORD_RANGE + 'delins' + DNA_ALT + ")",
    ]

    GENOMIC_ALLELE_REGEXES = [re.compile("^" + regex + "$")
                              for regex in GENOMIC_ALLELE]


class ChromosomeSubset(object):
    """
    Allow direct access to a subset of the chromosome.
    """

    def __init__(self, name, genome=None):
        self.name = name
        self.genome = genome

    def __getitem__(self, key):
        """Return sequence from region [start, end)

        Coordinates are 0-based, end-exclusive."""
        if isinstance(key, slice):
            start, end = (key.start, key.stop)
            start -= self.genome.start
            end -= self.genome.start
            return self.genome.genome[self.genome.seqid][start:end]
        else:
            raise TypeError('Expected a slice object but '
                            'received a {0}.'.format(type(key)))

    def __repr__(self):
        return 'ChromosomeSubset("%s")' % self.name


class GenomeSubset(object):
    """
    Allow the direct access of a subset of the genome.
    """

    def __init__(self, genome, chrom, start, end, seqid):
        self.genome = genome
        self.chrom = chrom
        self.start = start
        self.end = end
        self.seqid = seqid
        self._chroms = {}

    def __getitem__(self, chrom):
        """Return a chromosome by its name."""
        if chrom in self._chroms:
            return self._chroms[chrom]
        else:
            chromosome = ChromosomeSubset(chrom, self)
            self._chroms[chrom] = chromosome
            return chromosome


class CDNACoord(object):
    """
    A HGVS cDNA-based coordinate.

    A cDNA coordinate can take one of these forms:

    N = nucleotide N in protein coding sequence (e.g. 11A>G)

    -N = nucleotide N 5' of the ATG translation initiation codon (e.g. -4A>G)
         NOTE: so located in the 5'UTR or 5' of the transcription initiation
         site (upstream of the gene, incl. promoter)

    *N = nucleotide N 3' of the translation stop codon (e.g. *6A>G)
         NOTE: so located in the 3'UTR or 3' of the polyA-addition site
         (including downstream of the gene)

    N+M = nucleotide M in the intron after (3' of) position N in the coding DNA
          reference sequence (e.g. 30+4A>G)

    N-M = nucleotide M in the intron before (5' of) position N in the coding
          DNA reference sequence (e.g. 301-2A>G)

    -N+M / -N-M = nucleotide in an intron in the 5'UTR (e.g. -45+4A>G)

    *N+M / *N-M = nucleotide in an intron in the 3'UTR (e.g. *212-2A>G)
    """

    def __init__(self, coord=0, offset=0, landmark=CDNA_START_CODON,
                 string=''):
        """
        coord: main coordinate along cDNA on the same strand as the transcript

        offset: an additional genomic offset from the main coordinate.  This
                allows referencing non-coding (e.g. intronic) positions.
                Offset is also interpreted on the coding strand.

        landmark: ('cdna_start', 'cdna_stop') indicating that 'coord'
                  is relative to one of these landmarks.

        string: a coordinate from an HGVS name.  If given coord, offset, and
                landmark should not be specified.
        """

        if string:
            if coord != 0 or offset != 0 or landmark != CDNA_START_CODON:
                raise ValueError("coord, offset, and landmark should not "
                                 "be given with string argument")

            self.parse(string)
        else:
            self.coord = coord
            self.offset = offset
            self.landmark = landmark

    def parse(self, coord_text):
        """
        Parse a HGVS formatted cDNA coordinate.
        """

        match = re.match(r"(|-|\*)(\d+)((-|\+)(\d+))?", coord_text)
        if not match:
            raise ValueError("unknown coordinate format '%s'" % coord_text)
        coord_prefix, coord, _, offset_prefix, offset = match.groups()

        self.coord = int(coord)
        self.offset = int(offset) if offset else 0

        if offset_prefix == '-':
            self.offset *= -1
        elif offset_prefix == '+' or offset is None:
            pass
        else:
            raise ValueError("unknown offset_prefix '%s'" % offset_prefix)

        if coord_prefix == '':
            self.landmark = CDNA_START_CODON
        elif coord_prefix == "-":
            self.coord *= -1
            self.landmark = CDNA_START_CODON
        elif coord_prefix == '*':
            self.landmark = CDNA_STOP_CODON
        else:
            raise ValueError("unknown coord_prefix '%s'" % coord_prefix)
        return self

    def __str__(self):
        """
        Return a formatted cDNA coordinate
        """
        if self.landmark == CDNA_STOP_CODON:
            coord_prefix = '*'
        else:
            coord_prefix = ''

        if self.offset < 0:
            offset = '%d' % self.offset
        elif self.offset > 0:
            offset = '+%d' % self.offset
        else:
            offset = ''

        return '%s%d%s' % (coord_prefix, self.coord, offset)

    def __eq__(self, other):
        """Equality operator."""
        return ((self.coord, self.offset, self.landmark) ==
                (other.coord, other.offset, other.landmark))

    def __repr__(self):
        """
        Returns a string representation of a cDNA coordinate.
        """
        if self.landmark != CDNA_START_CODON:
            return "CDNACoord(%d, %d, '%s')" % (
                self.coord, self.offset, self.landmark)
        else:
            return "CDNACoord(%d, %d)" % (self.coord, self.offset)


# The RefSeq standard for naming contigs/transcripts/proteins:
# http://www.ncbi.nlm.nih.gov/books/NBK21091/table/ch18.T.refseq_accession_numbers_and_mole/?report=objectonly  # nopep8
REFSEQ_PREFIXES = [
    ('AC_', 'genomic',
     'Complete genomic molecule, usually alternate assembly'),
    ('NC_', 'genomic',
     'Complete genomic molecule, usually reference assembly'),
    ('NG_', 'genomic', 'Incomplete genomic region'),
    ('NT_', 'genomic', 'Contig or scaffold, clone-based or WGS'),
    ('NW_', 'genomic', 'Contig or scaffold, primarily WGS'),
    ('NS_', 'genomic', 'Environmental sequence'),
    ('NZ_', 'genomic', 'Unfinished WGS'),
    ('NM_', 'mRNA', ''),
    ('NR_', 'RNA', ''),
    ('XM_', 'mRNA', 'Predicted model'),
    ('XR_', 'RNA', 'Predicted model'),
    ('AP_', 'Protein', 'Annotated on AC_ alternate assembly'),
    ('NP_', 'Protein', 'Associated with an NM_ or NC_ accession'),
    ('YP_', 'Protein', ''),
    ('XP_', 'Protein', 'Predicted model, associated with an XM_ accession'),
    ('ZP_', 'Protein', 'Predicted model, annotated on NZ_ genomic records'),
]

REFSEQ_PREFIX_LOOKUP = dict(
    (prefix, (kind, description))
    for prefix, kind, description in REFSEQ_PREFIXES
)


def get_refseq_type(name):
    """
    Return the RefSeq type for a refseq name.
    """
    prefix = name[:3]
    return REFSEQ_PREFIX_LOOKUP.get(prefix, (None, ''))[0]


def get_exons(transcript):
    """Yield exons in coding order."""
    transcript_strand = transcript.tx_position.is_forward_strand
    if hasattr(transcript.exons, 'select_related'):
        exons = list(transcript.exons.select_related('tx_position'))
    else:
        exons = list(transcript.exons)
    exons.sort(key=lambda exon: exon.tx_position.chrom_start)
    if not transcript_strand:
        exons.reverse()
    return exons


def get_coding_exons(transcript):
    """Yield non-empty coding exonic regions in coding order."""
    for exon in get_exons(transcript):
        region = exon.get_as_interval(coding_only=True)
        if region:
            yield region


def get_utr5p_size(transcript):
    """Return the size of the 5prime UTR of a transcript."""

    transcript_strand = transcript.tx_position.is_forward_strand
    exons = get_exons(transcript)

    # Find the exon containing the start codon.
    start_codon = (transcript.cds_position.chrom_start if transcript_strand
                   else transcript.cds_position.chrom_stop - 1)
    cdna_len = 0
    for exon in exons:
        exon_start = exon.tx_position.chrom_start
        exon_end = exon.tx_position.chrom_stop
        if exon_start <= start_codon < exon_end:
            break
        cdna_len += exon_end - exon_start
    else:
        raise ValueError("transcript contains no exons")

    if transcript_strand:
        return cdna_len + (start_codon - exon_start)
    else:
        return cdna_len + (exon_end - start_codon - 1)


def find_stop_codon(exons, cds_position):
    """Return the position along the cDNA of the base after the stop codon."""
    if cds_position.is_forward_strand:
        stop_pos = cds_position.chrom_stop
    else:
        stop_pos = cds_position.chrom_start
    cdna_pos = 0
    for exon in exons:
        exon_start = exon.tx_position.chrom_start
        exon_stop = exon.tx_position.chrom_stop

        if exon_start <= stop_pos <= exon_stop:
            if cds_position.is_forward_strand:
                return cdna_pos + stop_pos - exon_start
            else:
                return cdna_pos + exon_stop - stop_pos
        else:
            cdna_pos += exon_stop - exon_start
    raise ValueError('Stop codon is not in any of the exons')


def get_genomic_sequence(genome, chrom, start, end):
    """
    Return a sequence for the genomic region.

    start, end: 1-based, end-inclusive coordinates of the sequence.
    """
    if start > end:
        return ''
    else:
        return str(genome[str(chrom)][start - 1:end]).upper()


def cdna_to_genomic_coord(transcript, coord):
    """Convert a HGVS cDNA coordinate to a genomic coordinate."""
    transcript_strand = transcript.tx_position.is_forward_strand
    exons = get_exons(transcript)
    utr5p = (get_utr5p_size(transcript)
             if transcript.is_coding else 0)

    # compute starting position along spliced transcript.
    if coord.landmark == CDNA_START_CODON:
        if coord.coord > 0:
            pos = utr5p + coord.coord
        else:
            pos = utr5p + coord.coord + 1
    elif coord.landmark == CDNA_STOP_CODON:
        if coord.coord < 0:
            raise ValueError('CDNACoord cannot have a negative coord and '
                             'landmark CDNA_STOP_CODON')
        pos = find_stop_codon(exons, transcript.cds_position) + coord.coord
    else:
        raise ValueError('unknown CDNACoord landmark "%s"' % coord.landmark)

    # 5' flanking sequence.
    if pos < 1:
        if transcript_strand:
            return transcript.tx_position.chrom_start + pos
        else:
            return transcript.tx_position.chrom_stop - pos + 1

    # Walk along transcript until we find an exon that contains pos.
    cdna_start = 1
    cdna_end = 1
    for exon in exons:
        exon_start = exon.tx_position.chrom_start + 1
        exon_end = exon.tx_position.chrom_stop
        cdna_end = cdna_start + (exon_end - exon_start)
        if cdna_start <= pos <= cdna_end:
            break
        cdna_start = cdna_end + 1
    else:
        # 3' flanking sequence
        if transcript_strand:
            return transcript.cds_position.chrom_stop + coord.coord
        else:
            return transcript.cds_position.chrom_start + 1 - coord.coord

    # Compute genomic coordinate using offset.
    if transcript_strand:
        # Plus strand.
        return exon_start + (pos - cdna_start) + coord.offset
    else:
        # Minus strand.
        return exon_end - (pos - cdna_start) - coord.offset


def genomic_to_cdna_coord(transcript, genomic_coord):
    """Convert a genomic coordinate to a cDNA coordinate and offset.
    """
    exons = [exon.get_as_interval()
             for exon in get_exons(transcript)]

    if len(exons) == 0:
        return None

    strand = transcript.strand

    if strand == "+":
        exons.sort(key=lambda exon: exon.chrom_start)
    else:
        exons.sort(key=lambda exon: -exon.chrom_end)

    distances = [exon.distance(genomic_coord)
                 for exon in exons]
    min_distance_to_exon = min(map(abs, distances))

    coding_offset = 0
    for exon in exons:
        exon_length = exon.chrom_end - exon.chrom_start
        distance = exon.distance(genomic_coord)
        if abs(distance) == min_distance_to_exon:
            if strand == "+":
                exon_start_cds_offset = coding_offset + 1
                exon_end_cds_offset = coding_offset + exon_length
            else:
                exon_start_cds_offset = coding_offset + exon_length
                exon_end_cds_offset = coding_offset + 1
            # This is the exon we want to annotate against.
            if distance == 0:
                # Inside the exon.
                if strand == "+":
                    coord = (exon_start_cds_offset +
                             (genomic_coord -
                              (exon.chrom_start + 1)))
                else:
                    coord = (exon_end_cds_offset +
                             (exon.chrom_end -
                              genomic_coord))
                cdna_coord = CDNACoord(coord, 0)
            else:
                # Outside the exon.
                if distance > 0:
                    nearest_exonic = exon_start_cds_offset
                else:
                    nearest_exonic = exon_end_cds_offset
                if strand == "+":
                    distance *= -1

                # If outside transcript, don't use offset.
                if (genomic_coord < transcript.tx_position.chrom_start + 1 or
                        genomic_coord > transcript.tx_position.chrom_stop):
                    nearest_exonic += distance
                    distance = 0
                cdna_coord = CDNACoord(nearest_exonic, distance)
            break
        coding_offset += exon_length

    # Adjust coordinates for coding transcript.
    if transcript.is_coding:
        # Detect if position before start codon.
        utr5p = get_utr5p_size(transcript) if transcript.is_coding else 0
        cdna_coord.coord -= utr5p
        if cdna_coord.coord <= 0:
            cdna_coord.coord -= 1
        else:
            # Detect if position is after stop_codon.
            exons = get_exons(transcript)
            stop_codon = find_stop_codon(exons, transcript.cds_position)
            stop_codon -= utr5p
            if (cdna_coord.coord > stop_codon or
                    cdna_coord.coord == stop_codon and cdna_coord.offset > 0):
                cdna_coord.coord -= stop_codon
                cdna_coord.landmark = CDNA_STOP_CODON

    return cdna_coord


def get_allele(hgvs, genome, transcript=None):
    """Get an allele from a HGVSName, a genome, and a transcript."""
    chrom, start, end = hgvs.get_coords(transcript)
    _, alt = hgvs.get_ref_alt(
        transcript.tx_position.is_forward_strand if transcript else True)
    ref = get_genomic_sequence(genome, chrom, start, end)
    return chrom, start, end, ref, alt


_indel_mutation_types = set(['ins', 'del', 'dup', 'delins'])


def get_vcf_allele(hgvs, genome, transcript=None):
    """Get an VCF-style allele from a HGVSName, a genome, and a transcript."""
    chrom, start, end = hgvs.get_vcf_coords(transcript)
    _, alt = hgvs.get_ref_alt(
        transcript.tx_position.is_forward_strand if transcript else True)
    ref = get_genomic_sequence(genome, chrom, start, end)

    if hgvs.mutation_type in _indel_mutation_types:
        # Left-pad alternate allele.
        alt = ref[0] + alt
    return chrom, start, end, ref, alt


def matches_ref_allele(hgvs, genome, transcript=None):
    """Return True if reference allele matches genomic sequence.

    For duplication variants the check verifies that the sequence at the
    duplicated region in the genome matches the stated ``ref_allele``.  When
    no explicit sequence is stored (``ref_allele == ''``) the check is
    skipped and ``True`` is returned.
    """
    is_fwd = transcript.tx_position.is_forward_strand if transcript else True

    if hgvs.mutation_type == "dup":
        # For dup, get_coords now returns an empty interval (insertion point).
        # We need the actual duplicated region to verify the sequence.
        if not hgvs.ref_allele:
            # No explicit sequence to check.
            return True
        if hgvs.kind == 'g':
            dup_start = hgvs.start
            dup_end = hgvs.end
            chrom = hgvs.chrom
        elif hgvs.kind == 'c' and transcript:
            chrom = transcript.tx_position.chrom
            dup_start = cdna_to_genomic_coord(transcript, hgvs.cdna_start)
            dup_end = cdna_to_genomic_coord(transcript, hgvs.cdna_end)
            if not transcript.tx_position.is_forward_strand:
                dup_start, dup_end = dup_end, dup_start
        else:
            return True
        genome_ref = get_genomic_sequence(genome, chrom, dup_start, dup_end)
        stated_ref = hgvs.ref_allele if is_fwd else revcomp(hgvs.ref_allele)
        return genome_ref == stated_ref

    ref, alt = hgvs.get_ref_alt(is_fwd)
    chrom, start, end = hgvs.get_coords(transcript)
    genome_ref = get_genomic_sequence(genome, chrom, start, end)
    return genome_ref == ref



#: Backward-compatible alias for :class:`HGVSParseError`.
#:
#: .. deprecated::
#:     Use :class:`HGVSParseError` for new code.  This alias is retained so
#:     that existing ``except InvalidHGVSName`` handlers continue to work.
InvalidHGVSName = HGVSParseError


class HGVSName(object):
    """
    Represents a HGVS variant name.
    """

    def __init__(self, name='', prefix='', chrom='', transcript='', gene='',
                 kind='', mutation_type=None, start=0, end=0, ref_allele='',
                 ref2_allele='', alt_allele='',
                 cdna_start=None, cdna_end=None, pep_extra=''):

        # Full HGVS name.
        self.name = name

        # Name parts.
        self.prefix = prefix
        self.chrom = chrom
        self.transcript = transcript
        self.gene = gene
        self.kind = kind
        self.mutation_type = mutation_type
        self.start = start
        self.end = end
        self.ref_allele = ref_allele    # reference allele
        self.ref2_allele = ref2_allele  # reference allele at end of pep indel
        self.alt_allele = alt_allele    # alternate allele

        # cDNA-specific fields
        self.cdna_start = cdna_start if cdna_start else CDNACoord()
        self.cdna_end = cdna_end if cdna_end else CDNACoord()

        # Protein-specific fields
        self.pep_extra = pep_extra

        if name:
            self.parse(name)

    def parse(self, name):
        """Parse a HGVS name."""
        # Does HGVS name have transcript/gene prefix?
        if ':' in name:
            prefix, allele = name.split(':', 1)
        else:
            prefix = ''
            allele = name

        self.name = name

        # Parse prefix and allele.
        self.parse_allele(allele)
        self.parse_prefix(prefix, self.kind)
        self._validate()

    def parse_prefix(self, prefix, kind):
        """
        Parse a HGVS prefix (gene/transcript/chromosome).

        Some examples of full hgvs names with transcript include:
          NM_007294.3:c.2207A>C
          NM_007294.3(BRCA1):c.2207A>C
          BRCA1{NM_007294.3}:c.2207A>C
        """

        self.prefix = prefix

        # No prefix.
        if prefix == '':
            self.chrom = ''
            self.transcript = ''
            self.gene = ''
            return

        # Transcript and gene given with parens.
        # example: NM_007294.3(BRCA1):c.2207A>C
        match = re.match(r"^(?P<transcript>[^(]+)\((?P<gene>[^)]+)\)$", prefix)
        if match:
            self.transcript = match.group('transcript')
            self.gene = match.group('gene')
            return

        # Transcript and gene given with braces.
        # example: BRCA1{NM_007294.3}:c.2207A>C
        match = re.match(r"^(?P<gene>[^{]+)\{(?P<transcript>[^}]+)\}$", prefix)
        if match:
            self.transcript = match.group('transcript')
            self.gene = match.group('gene')
            return

        # Determine using Ensembl type.
        if prefix.startswith('ENST'):
            self.transcript = prefix
            return

        # Determine using refseq type.
        refseq_type = get_refseq_type(prefix)
        if refseq_type in ('mRNA', 'RNA'):
            self.transcript = prefix
            return

        # Determine using refseq type.
        if prefix.startswith(CHROM_PREFIX) or refseq_type == 'genomic':
            self.chrom = prefix
            return

        # Assume gene name.
        self.gene = prefix

    def parse_allele(self, allele):
        """
        Parse a HGVS allele description.

        Some examples include:
          cDNA substitution: c.101A>C,
          cDNA indel: c.3428delCinsTA, c.1000_1003delATG, c.1000_1001insATG
          No protein change: p.Glu1161=
          Protein change: p.Glu1161Ser
          Protein frameshift: p.Glu1161_Ser1164?fs
          Genomic substitution: g.1000100A>T
          Genomic indel: g.1000100_1000102delATG
        """
        if '.' not in allele:
            raise InvalidHGVSName(allele, 'allele',
                                  'expected kind "c.", "p.", "g.", etc')

        # Determine HGVS name kind.
        kind, details = allele.split('.', 1)
        self.kind = kind
        self.mutation_type = None

        if kind == "c":
            self.parse_cdna(details)
        elif kind == "p":
            self.parse_protein(details)
        elif kind == "g":
            self.parse_genome(details)
        else:
            raise HGVSParseError(allele, 'allele',
                                 'unknown kind; expected "c", "p", "g", etc')

    def parse_cdna(self, details):
        """
        Parse a HGVS cDNA name.

        Some examples include:
          Substitution: 101A>C,
          Indel: 3428delCinsTA, 1000_1003delATG, 1000_1001insATG
        """
        for regex in HGVSRegex.CDNA_ALLELE_REGEXES:
            match = re.match(regex, details)
            if match:
                groups = match.groupdict()

                # Parse mutation type.
                if groups.get('delins'):
                    self.mutation_type = 'delins'
                else:
                    self.mutation_type = groups['mutation_type']

                # Parse coordinates.
                self.cdna_start = CDNACoord(string=groups.get('start'))
                if groups.get('end'):
                    self.cdna_end = CDNACoord(string=groups.get('end'))
                else:
                    self.cdna_end = CDNACoord(string=groups.get('start'))

                # Parse alleles.
                self.ref_allele = groups.get('ref', '')
                self.alt_allele = groups.get('alt', '')

                # Convert numerical allelles.
                if self.ref_allele.isdigit():
                    self.ref_allele = "N" * int(self.ref_allele)
                if self.alt_allele.isdigit():
                    self.alt_allele = "N" * int(self.alt_allele)

                # Convert duplication alleles.
                # For dup, alt_allele stores the inserted (duplicated) sequence,
                # which is identical to ref_allele.  The previous convention of
                # storing ref_allele * 2 is no longer used here; get_ref_alt()
                # returns ("", ref_allele) for dup variants.
                if self.mutation_type == "dup":
                    self.alt_allele = self.ref_allele

                # Convert no match alleles.
                if self.mutation_type == "=":
                    self.alt_allele = self.ref_allele
                return

        raise InvalidHGVSName(details, 'cDNA allele')

    def parse_protein(self, details):
        """
        Parse a HGVS protein name.

        Both one-letter (e.g. ``R132H``) and three-letter (e.g.
        ``Arg132His``) amino acid notation are accepted per the HGVS stable
        recommendations (https://hgvs-nomenclature.org/stable/).  Amino acid
        alleles are stored exactly as they appear in the input string; use
        :meth:`normalize` to obtain a canonical three-letter representation.

        Supported categories:
          Substitution:        Glu1161Ser / R132H / Arg132*
          No change:           Glu1161=
          Deletion:            Lys23del / Lys23_Val25del
          Duplication:         Lys23dup / Lys23_Val25dup
          Insertion:           Lys23_Leu24insArgSerGln
          Deletion-Insertion:  Cys28delinsTrpVal / Cys28_Lys29delinsTrp
          Frameshift:          Arg97fs / Arg97ProfsTer23 / Arg97Profs*23
          Extension (N-term):  Met1ext-5
          Extension (C-term):  *110GlnextTer17
          Repeated sequences:  Ala2[10] / Ala2_Pro5[10]
          Predicted forms:     (Arg97fs) / (Lys23del)  [parens stripped]
          Range change (legacy): Glu1161_Ser1164?fs
        """
        # Handle predicted / uncertain consequence form — strip outer parens.
        # p.(Arg97fs) and p.(Lys23del) are parsed identically to the bare form;
        # the predicted nature is currently not stored in a dedicated field.
        if details.startswith('(') and details.endswith(')'):
            details = details[1:-1]

        for regex in HGVSRegex.PEP_ALLELE_REGEXES:
            match = re.match(regex, details)
            if match:
                groups = match.groupdict()

                # ----------------------------------------------------------------
                # Determine mutation type and extract fields based on which
                # named groups are present in this particular regex pattern.
                # ----------------------------------------------------------------

                if 'new_aa' in groups:
                    # Frameshift with a new first amino acid.
                    # e.g. Arg97ProfsTer23 / Arg97Profs*23 / Ile327Argfs*?
                    self.mutation_type = 'fs'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    self.end = self.start
                    self.ref2_allele = self.ref_allele
                    self.alt_allele = groups.get('new_aa', '')
                    ter_pos = groups.get('ter_pos')
                    self.pep_extra = 'fsTer' + ter_pos if ter_pos else 'fs'

                elif 'ter_pos' in groups:
                    # Frameshift without a new amino acid.
                    # e.g. Arg97fs / Arg97fsTer23 / Arg97fs*?
                    self.mutation_type = 'fs'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    self.end = self.start
                    self.ref2_allele = self.ref_allele
                    self.alt_allele = ''
                    ter_pos = groups.get('ter_pos')
                    self.pep_extra = 'fsTer' + ter_pos if ter_pos else 'fs'

                elif 'pep_del' in groups:
                    # Deletion: Lys23del / Lys23_Val25del
                    self.mutation_type = 'del'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    if groups.get('end'):
                        self.end = int(groups.get('end'))
                        self.ref2_allele = groups.get('ref2', '')
                    else:
                        self.end = self.start
                        self.ref2_allele = self.ref_allele
                    self.alt_allele = ''
                    self.pep_extra = ''

                elif 'pep_dup' in groups:
                    # Duplication: Lys23dup / Lys23_Val25dup
                    self.mutation_type = 'dup'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    if groups.get('end'):
                        self.end = int(groups.get('end'))
                        self.ref2_allele = groups.get('ref2', '')
                    else:
                        self.end = self.start
                        self.ref2_allele = self.ref_allele
                    self.alt_allele = self.ref_allele  # duplicated == ref
                    self.pep_extra = ''

                elif 'pep_ins' in groups:
                    # Insertion: Lys23_Leu24insArgSerGln
                    self.mutation_type = 'ins'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    self.end = int(groups.get('end'))
                    self.ref2_allele = groups.get('ref2', '')
                    self.alt_allele = groups.get('alt', '')
                    self.pep_extra = ''

                elif 'pep_delins' in groups:
                    # Deletion-Insertion: Cys28delinsTrpVal / Cys28_Lys29delinsTrp
                    self.mutation_type = 'delins'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    if groups.get('end'):
                        self.end = int(groups.get('end'))
                        self.ref2_allele = groups.get('ref2', '')
                    else:
                        self.end = self.start
                        self.ref2_allele = self.ref_allele
                    self.alt_allele = groups.get('alt', '')
                    self.pep_extra = ''

                elif 'pep_ext' in groups:
                    # Extension (N-terminal or C-terminal)
                    self.mutation_type = 'ext'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    self.end = self.start
                    self.ref2_allele = self.ref_allele
                    if 'ext_offset' in groups:
                        # N-terminal: Met1ext-5
                        self.alt_allele = ''
                        ext_offset = groups.get('ext_offset') or ''
                        self.pep_extra = 'ext' + ext_offset
                    else:
                        # C-terminal: *110GlnextTer17
                        self.alt_allele = groups.get('ext_aa', '')
                        ext_ter_pos = groups.get('ext_ter_pos')
                        self.pep_extra = ('extTer' + ext_ter_pos
                                          if ext_ter_pos else 'ext')

                elif 'rep_count' in groups:
                    # Repeated sequences: Ala2[10] / Ala2_Pro5[10]
                    self.mutation_type = 'rep'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    if groups.get('end'):
                        self.end = int(groups.get('end'))
                        self.ref2_allele = groups.get('ref2', '')
                    else:
                        self.end = self.start
                        self.ref2_allele = self.ref_allele
                    self.alt_allele = ''
                    rep_count = groups.get('rep_count', '')
                    self.pep_extra = '[' + rep_count + ']'

                elif groups.get('delins'):
                    # Legacy range change: Glu1161_Ser1164?fs
                    self.mutation_type = 'delins'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    self.end = int(groups.get('end'))
                    if groups.get('ref2'):
                        self.ref2_allele = groups.get('ref2')
                        self.alt_allele = groups.get('alt', '')
                    else:
                        self.ref2_allele = self.ref_allele
                        self.alt_allele = groups.get('alt', self.ref_allele)
                    self.pep_extra = groups.get('extra') or ''

                else:
                    # Substitution or no-change: Glu1161Ser / R132H / Glu1161=
                    self.mutation_type = '>'
                    self.ref_allele = groups.get('ref', '')
                    self.start = int(groups.get('start'))
                    self.end = self.start
                    if groups.get('ref2'):
                        self.ref2_allele = groups.get('ref2')
                        self.alt_allele = groups.get('alt', '')
                    else:
                        self.ref2_allele = self.ref_allele
                        self.alt_allele = groups.get('alt', self.ref_allele)
                    self.pep_extra = groups.get('extra') or ''

                return

        raise HGVSParseError(details, 'protein allele')

    def parse_genome(self, details):
        """
        Parse a HGVS genomic name.

        Some examples include:
          Substitution: 1000100A>T
          Indel: 1000100_1000102delATG
        """

        for regex in HGVSRegex.GENOMIC_ALLELE_REGEXES:
            match = re.match(regex, details)
            if match:
                groups = match.groupdict()

                # Parse mutation type.
                if groups.get('delins'):
                    self.mutation_type = 'delins'
                else:
                    self.mutation_type = groups['mutation_type']

                # Parse coordinates.
                self.start = int(groups.get('start'))
                if groups.get('end'):
                    self.end = int(groups.get('end'))
                else:
                    self.end = self.start

                # Parse alleles.
                self.ref_allele = groups.get('ref', '')
                self.alt_allele = groups.get('alt', '')

                # Convert numerical allelles.
                if self.ref_allele.isdigit():
                    self.ref_allele = "N" * int(self.ref_allele)
                if self.alt_allele.isdigit():
                    self.alt_allele = "N" * int(self.alt_allele)

                # Convert duplication alleles.
                # alt_allele stores the inserted (duplicated) sequence,
                # identical to ref_allele.
                if self.mutation_type == "dup":
                    self.alt_allele = self.ref_allele

                # Convert no match alleles.
                if self.mutation_type == "=":
                    self.alt_allele = self.ref_allele
                return

        raise HGVSParseError(details, 'genomic allele')

    def _validate(self):
        """
        Check for internal inconsistencies in representation
        """
        if self.start > self.end:
            raise HGVSParseError(reason="Coordinates are nonincreasing")

    def __repr__(self):
        try:
            return "HGVSName('%s')" % self.format()
        except (HGVSFormattingError, NotImplementedError):
            return "HGVSName('%s')" % self.name

    def __unicode__(self):
        return self.format()

    def format(self, use_prefix=True, use_gene=True, use_counsyl=False,
               use_3letter=True):
        """Generate a HGVS name as a string.

        Args:
            use_prefix: Include a transcript/gene/chromosome prefix.
            use_gene: Include the gene name alongside the transcript.
            use_counsyl: Use Counsyl-style formatting conventions.
            use_3letter: For protein (``p.``) names, use three-letter amino
                acid codes (``True``, the default) rather than single-letter.
        """

        if self.kind == 'c':
            allele = 'c.' + self.format_cdna()
        elif self.kind == 'p':
            allele = 'p.' + self.format_protein(use_3letter=use_3letter)
        elif self.kind == 'g':
            allele = 'g.' + self.format_genome()
        else:
            raise HGVSFormattingError("unsupported HGVS kind: %r" % self.kind)

        prefix = self.format_prefix(use_gene=use_gene) if use_prefix else ''

        if prefix:
            return prefix + ':' + allele
        else:
            return allele

    def format_prefix(self, use_gene=True):
        """
        Generate HGVS trancript/gene prefix.

        Some examples of full hgvs names with transcript include:
          NM_007294.3:c.2207A>C
          NM_007294.3(BRCA1):c.2207A>C
        """

        if self.kind == 'g':
            if self.chrom:
                return self.chrom

        if self.transcript:
            if use_gene and self.gene:
                return '%s(%s)' % (self.transcript, self.gene)
            else:
                return self.transcript
        else:
            if use_gene:
                return self.gene
            else:
                return ''

    def format_cdna_coords(self):
        """
        Generate HGVS cDNA coordinates string.
        """
        # Format coordinates.
        if self.cdna_start == self.cdna_end:
            return str(self.cdna_start)
        else:
            return "%s_%s" % (self.cdna_start, self.cdna_end)

    def format_dna_allele(self):
        """
        Generate HGVS DNA allele.
        """
        if self.mutation_type == '=':
            # No change.
            # example: 101A=
            return self.ref_allele + '='

        if self.mutation_type == '>':
            # SNP.
            # example: 101A>C
            return self.ref_allele + '>' + self.alt_allele

        elif self.mutation_type == 'delins':
            # Indel.
            # example: 112_117delAGGTCAinsTG, 112_117delinsTG
            return 'del' + self.ref_allele + 'ins' + self.alt_allele

        elif self.mutation_type in ('del', 'dup'):
            # Delete, duplication.
            # example: 1000_1003delATG, 1000_1003dupATG
            return self.mutation_type + self.ref_allele

        elif self.mutation_type == 'ins':
            # Insert.
            # example: 1000_1001insATG
            return self.mutation_type + self.alt_allele

        elif self.mutation_type == 'inv':
            return self.mutation_type

        else:
            raise HGVSFormattingError(
                "unknown mutation type: %r" % self.mutation_type)

    def format_cdna(self):
        """
        Generate HGVS cDNA allele.

        Some examples include:
          Substitution: 101A>C,
          Indel: 3428delCinsTA, 1000_1003delATG, 1000_1001insATG
        """
        return self.format_cdna_coords() + self.format_dna_allele()

    def format_protein(self, use_3letter=True):
        """
        Generate HGVS protein name.

        Args:
            use_3letter: If ``True`` (default), amino acid alleles are
                emitted using three-letter codes (e.g. ``Glu``, ``Ser``).
                If ``False``, single-letter codes are used instead
                (e.g. ``E``, ``S``).

        Supported categories (HGVS stable recommendations):
          Substitution:        Glu1161Ser  /  R132H
          No change:           Glu1161=
          Deletion:            Lys23del  /  Lys23_Val25del
          Duplication:         Lys23dup  /  Lys23_Val25dup
          Insertion:           Lys23_Leu24insArgSerGln
          Deletion-Insertion:  Cys28delinsTrpVal  /  Cys28_Lys29delinsTrp
          Frameshift:          Arg97fs  /  Arg97ProfsTer23
          Extension:           Met1ext-5  /  *110GlnextTer17
          Repeated sequences:  Ala2[10]
          Range change (legacy): Glu1161_Ser1164?fs
        """

        def _fmt(allele):
            """Normalise *allele* according to the requested notation."""
            return normalize_aa_allele(allele, use_3letter=use_3letter)

        def _norm_ter(s):
            """Convert Ter ↔ * inside a suffix string based on use_3letter."""
            if not s:
                return s
            if use_3letter:
                return s.replace('*', 'Ter')
            else:
                return s.replace('Ter', '*')

        # --- New protein mutation types (HGVS stable) ---

        if self.mutation_type == 'fs':
            # Frameshift: Arg97fs / Arg97ProfsTer23 / Arg97Profs*23
            result = _fmt(self.ref_allele) + str(self.start)
            if self.alt_allele:
                result += _fmt(self.alt_allele)
            result += _norm_ter(self.pep_extra or 'fs')
            return result

        if self.mutation_type == 'del':
            # Deletion: Lys23del / Lys23_Val25del
            result = _fmt(self.ref_allele) + str(self.start)
            if self.start != self.end:
                result += '_' + _fmt(self.ref2_allele) + str(self.end)
            return result + 'del'

        if self.mutation_type == 'dup':
            # Duplication: Lys23dup / Lys23_Val25dup
            result = _fmt(self.ref_allele) + str(self.start)
            if self.start != self.end:
                result += '_' + _fmt(self.ref2_allele) + str(self.end)
            return result + 'dup'

        if self.mutation_type == 'ins':
            # Insertion: Lys23_Leu24insArgSerGln
            return (_fmt(self.ref_allele) + str(self.start)
                    + '_' + _fmt(self.ref2_allele) + str(self.end)
                    + 'ins' + _fmt(self.alt_allele))

        if self.mutation_type == 'rep':
            # Repeated sequences: Ala2[10] / Ala2_Pro5[10]
            result = _fmt(self.ref_allele) + str(self.start)
            if self.start != self.end:
                result += '_' + _fmt(self.ref2_allele) + str(self.end)
            return result + (self.pep_extra or '')

        if self.mutation_type == 'ext':
            # Extension: Met1ext-5 (N-term) / *110GlnextTer17 (C-term)
            result = _fmt(self.ref_allele) + str(self.start)
            if self.alt_allele:
                result += _fmt(self.alt_allele)
            return result + _norm_ter(self.pep_extra or 'ext')

        if self.mutation_type == 'delins':
            # Deletion-Insertion (both protein delins and legacy range change)
            if self.start != self.end:
                base = (_fmt(self.ref_allele) + str(self.start)
                        + '_' + _fmt(self.ref2_allele) + str(self.end))
                if self.alt_allele:
                    # New range delins: Cys28_Lys29delinsTrp
                    return base + 'delins' + _fmt(self.alt_allele)
                else:
                    # Legacy range change: Glu1161_Ser1164?fs
                    return base + (self.pep_extra or '')
            else:
                # Single-residue delins: Cys28delinsTrpVal
                return (_fmt(self.ref_allele) + str(self.start)
                        + 'delins' + _fmt(self.alt_allele))

        # --- Original logic for substitution and no-change ---

        if (self.start == self.end and
                self.ref_allele == self.ref2_allele == self.alt_allele):
            # Match.
            # Example: Glu1161=
            pep_extra = self.pep_extra if self.pep_extra else '='
            return _fmt(self.ref_allele) + str(self.start) + pep_extra

        if (self.start == self.end and
                self.ref_allele == self.ref2_allele and
                self.ref_allele != self.alt_allele):
            # Change.
            # Example: Glu1161Ser
            return (_fmt(self.ref_allele) + str(self.start) +
                    _fmt(self.alt_allele) + (self.pep_extra or ''))

        if self.start != self.end:
            # Fallback range change (manually-constructed names).
            return (_fmt(self.ref_allele) + str(self.start) + '_' +
                    _fmt(self.ref2_allele) + str(self.end) +
                    (self.pep_extra or ''))

        raise HGVSFormattingError('cannot format protein name with these fields')

    def normalize(self):
        """Return a new :class:`HGVSName` with amino acid alleles normalised to
        three-letter codes.

        For non-protein HGVS names the object is returned unchanged (a copy is
        not made — the same object is returned because there is nothing to
        normalise).

        Returns:
            A :class:`HGVSName` instance where protein allele fields use
            three-letter amino acid codes.
        """
        if self.kind != 'p':
            return self

        import copy
        norm = copy.copy(self)
        norm.ref_allele = normalize_aa_allele(self.ref_allele, use_3letter=True)
        norm.ref2_allele = normalize_aa_allele(self.ref2_allele, use_3letter=True)
        norm.alt_allele = normalize_aa_allele(self.alt_allele, use_3letter=True)
        # Normalise Ter/Star notation inside pep_extra so that equivalence
        # comparisons treat p.Arg97Profs*23 and p.Arg97ProfsTer23 as equal.
        if self.pep_extra:
            norm.pep_extra = self.pep_extra.replace('*', 'Ter')
        return norm

    def equivalent(self, other):
        """Return ``True`` if *other* represents the same variant as this name.

        Semantic equivalence is determined by normalising both names to their
        canonical three-letter amino-acid form before comparison.  For cDNA and
        genomic names, field-level equality is used directly.

        The comparison does **not** require the transcript prefixes to match so
        that, for example, a bare ``p.R132H`` can be compared against a fully
        qualified ``NM_004380.2(IDH1):p.Arg132His``.

        Args:
            other: Another :class:`HGVSName` instance to compare against.

        Returns:
            ``bool``
        """
        if not isinstance(other, HGVSName):
            return NotImplemented
        if self.kind != other.kind:
            return False

        a = self.normalize()
        b = other.normalize()

        if self.kind == 'p':
            return (a.start == b.start and
                    a.end == b.end and
                    a.ref_allele == b.ref_allele and
                    a.ref2_allele == b.ref2_allele and
                    a.alt_allele == b.alt_allele and
                    a.mutation_type == b.mutation_type and
                    # Normalise pep_extra: treat '' and None as equal
                    (a.pep_extra or '') == (b.pep_extra or ''))

        if self.kind == 'c':
            return (a.cdna_start == b.cdna_start and
                    a.cdna_end == b.cdna_end and
                    a.ref_allele == b.ref_allele and
                    a.alt_allele == b.alt_allele and
                    a.mutation_type == b.mutation_type)

        # Genomic
        return (a.chrom == b.chrom and
                a.start == b.start and
                a.end == b.end and
                a.ref_allele == b.ref_allele and
                a.alt_allele == b.alt_allele and
                a.mutation_type == b.mutation_type)

    def format_coords(self):
        """
        Generate HGVS cDNA coordinates string.
        """
        # Format coordinates.
        if self.start == self.end:
            return str(self.start)
        else:
            return "%s_%s" % (self.start, self.end)

    def format_genome(self):
        """
        Generate HGVS genomic allele.

        Some examples include:
          Substitution: 1000100A>T
          Indel: 1000100_1000102delATG
        """
        return self.format_coords() + self.format_dna_allele()

    def get_coords(self, transcript=None):
        """Return genomic coordinates of reference allele.

        For insertions (``ins``) and duplications (``dup``) the coordinates
        describe an *empty* interval representing the insertion point.  The
        empty interval is encoded as ``(start, start - 1)`` so that
        :meth:`get_vcf_coords` can left-pad by subtracting 1 from *start*.

        For cDNA duplications the convention is consistent with historical
        behaviour: the empty interval is placed at the start of the duplicated
        region, so that after VCF left-padding and normalisation the result
        matches the canonical left-normalised VCF form.

        For genomic duplications the insertion point is placed *after* the
        last base of the duplicated region (HGVS 3′ convention), which allows
        the correct single-base VCF anchor to be computed.
        """
        if self.kind == 'c':
            chrom = transcript.tx_position.chrom
            start = cdna_to_genomic_coord(transcript, self.cdna_start)
            end = cdna_to_genomic_coord(transcript, self.cdna_end)

            if not transcript.tx_position.is_forward_strand:
                if end > start:
                    raise HGVSParseError(
                        reason="cdna_start cannot be greater than cdna_end")
                start, end = end, start
            else:
                if start > end:
                    raise HGVSParseError(
                        reason="cdna_start cannot be greater than cdna_end")

            if self.mutation_type == "ins":
                # Inserts have empty interval.
                if start < end:
                    start += 1
                    end -= 1
                else:
                    end = start - 1

            elif self.mutation_type == "dup":
                # Empty interval at the start of the duplicated region.
                # After get_vcf_coords subtracts 1, the anchor base lands at
                # genomic(cdna_start) - 1, which together with VCF
                # normalisation gives the canonical left-normalised result.
                end = start - 1

        elif self.kind == 'g':
            chrom = self.chrom
            start = self.start
            end = self.end

            if self.mutation_type == "dup":
                # For genomic dup, the insertion is AFTER the end of the dup
                # region (HGVS 3′ convention).  Use an empty interval so that
                # get_vcf_coords gives ref = seq(end, end) = 1 base.
                start = end + 1
                # end remains unchanged (last base of duplicated region).

        else:
            raise HGVSFormattingError(
                'Coordinates are not available for HGVS kind "%s"' % self.kind)

        return chrom, start, end

    def get_vcf_coords(self, transcript=None):
        """Return genomic coordinates of reference allele in VCF-style.

        Insertions, deletions and duplications require one base of left-padding
        (the VCF anchor base).  Inversions use their full coordinate range.
        """
        chrom, start, end = self.get_coords(transcript)

        # Inserts and deletes require left-padding by 1 base
        if self.mutation_type in ("=", ">"):
            pass
        elif self.mutation_type in ("del", "ins", "dup", "delins"):
            # Indels have left-padding.
            start -= 1
        elif self.mutation_type == "inv":
            # Inversions span their full range; no padding needed.
            pass
        else:
            raise HGVSFormattingError(
                "Unknown mutation_type %r" % self.mutation_type)
        return chrom, start, end

    def get_ref_alt(self, is_forward_strand=True):
        """Return reference and alternate alleles.

        For duplication (``dup``) variants, returns an empty reference string
        and the duplicated sequence as the alternate allele, representing the
        event as a pure insertion.  This is consistent regardless of whether
        the sequence is stored explicitly or not.
        """
        if self.kind == 'p':
            raise NotImplementedError(
                'get_ref_alt is not implemented for protein HGVS names')
        alleles = [self.ref_allele, self.alt_allele]

        # Represent duplications as inserts: ref is empty, alt is the
        # duplicated sequence (stored in ref_allele).
        if self.mutation_type == "dup":
            alleles[0] = ""
            alleles[1] = self.ref_allele

        if is_forward_strand:
            return alleles
        else:
            return tuple(map(revcomp, alleles))


def hgvs_justify_dup(chrom, offset, ref, alt, genome):
    """
    Determines if allele is a duplication and justifies.

    chrom: Chromosome name.
    offset: 1-index genomic coordinate.
    ref: Reference allele (no padding).
    alt: Alternate allele (no padding).
    genome: pygr compatible genome object.

    For a duplication, ``ref`` and ``alt`` are both set to the duplicated
    sequence (``indel_seq``).  The mutation_type is set to ``'dup'``.
    For a plain insertion, mutation_type is ``'ins'``.

    Returns ``(chrom, offset, ref, alt, mutation_type)``.
    """

    if len(ref) == len(alt) == 0:
        # it's a SNP, just return.
        return chrom, offset, ref, alt, '>'

    if len(ref) > 0 and len(alt) > 0:
        # complex indel, don't know how to dup check
        return chrom, offset, ref, alt, 'delins'

    if len(ref) > len(alt):
        # deletion -- don't dup check
        return chrom, offset, ref, alt, 'del'

    indel_seq = alt
    indel_length = len(indel_seq)

    # Convert offset to 0-index.
    offset -= 1

    # Get genomic sequence around the lesion.
    prev_seq = str(
        genome[str(chrom)][offset - indel_length:offset]).upper()
    next_seq = str(
        genome[str(chrom)][offset:offset + indel_length]).upper()

    # Convert offset back to 1-index.
    offset += 1

    if prev_seq == indel_seq:
        offset = offset - indel_length
        mutation_type = 'dup'
        ref = indel_seq
        # alt equals ref for dup: both represent the duplicated sequence.
        alt = indel_seq
    elif next_seq == indel_seq:
        mutation_type = 'dup'
        ref = indel_seq
        alt = indel_seq
    else:
        mutation_type = 'ins'

    return chrom, offset, ref, alt, mutation_type


def hgvs_justify_indel(chrom, offset, ref, alt, strand, genome):
    """
    3' justify an indel according to the HGVS standard.

    Returns (chrom, offset, ref, alt).
    """
    if len(ref) == len(alt) == 0:
        # It's a SNP, just return.
        return chrom, offset, ref, alt

    if len(ref) > 0 and len(alt) > 0:
        # Complex indel, don't know how to justify.
        return chrom, offset, ref, alt

    # Get genomic sequence around the lesion.
    start = max(offset - 100, 0)
    end = offset + 100
    seq = str(genome[str(chrom)][start - 1:end]).upper()
    cds_offset = offset - start

    # indel -- strip off the ref base to get the actual lesion sequence
    is_insert = len(alt) > 0
    if is_insert:
        indel_seq = alt
        cds_offset_end = cds_offset
    else:
        indel_seq = ref
        cds_offset_end = cds_offset + len(indel_seq)

    # Now 3' justify (vs. cDNA not genome) the offset
    justify = 'right' if strand == '+' else 'left'
    offset, _, indel_seq = justify_indel(
        cds_offset, cds_offset_end, indel_seq, seq, justify)
    offset += start

    if is_insert:
        alt = indel_seq
    else:
        ref = indel_seq

    return chrom, offset, ref, alt


def hgvs_normalize_variant(chrom, offset, ref, alt, genome, transcript=None):
    """Convert VCF-style variant to HGVS-style."""
    if len(ref) == len(alt) == 1:
        if ref == alt:
            mutation_type = '='
        else:
            mutation_type = '>'
    else:
        # Remove 1bp padding
        offset += 1
        ref = ref[1:]
        alt = alt[1:]

        # 3' justify allele.
        strand = transcript.strand if transcript else '+'
        chrom, offset, ref, alt = hgvs_justify_indel(
            chrom, offset, ref, alt, strand, genome)

        # Represent as duplication if possible.
        chrom, offset, ref, alt, mutation_type = hgvs_justify_dup(
            chrom, offset, ref, alt, genome)
    return chrom, offset, ref, alt, mutation_type


def parse_hgvs_name(hgvs_name, genome, transcript=None,
                    get_transcript=lambda name: None,
                    flank_length=30, normalize=True, lazy=False):
    """
    Parse an HGVS name into (chrom, start, end, ref, alt)

    hgvs_name: HGVS name to parse.
    genome: pygr compatible genome object.
    transcript: Transcript corresponding to HGVS name.
    normalize: If True, normalize allele according to VCF standard.
    lazy: If True, discard version information from incoming transcript/gene.
    """
    hgvs = HGVSName(hgvs_name)

    # Determine transcript.
    if hgvs.kind == 'c' and not transcript:
        if '.' in hgvs.transcript and lazy:
            hgvs.transcript, version = hgvs.transcript.split('.')
        elif '.' in hgvs.gene and lazy:
            hgvs.gene, version = hgvs.gene.split('.')
        if get_transcript:
            if hgvs.transcript:
                transcript = get_transcript(hgvs.transcript)
            elif hgvs.gene:
                transcript = get_transcript(hgvs.gene)
        if not transcript:
            raise ValueError('transcript is required')

    if transcript and hgvs.transcript in genome:
        # Reference sequence is directly known, use it.
        genome = GenomeSubset(genome, transcript.tx_position.chrom,
                              transcript.tx_position.chrom_start,
                              transcript.tx_position.chrom_stop,
                              hgvs.transcript)

    chrom, start, end, ref, alt = get_vcf_allele(hgvs, genome, transcript)
    if normalize:
        chrom, start, ref, [alt] = normalize_variant(
            chrom, start, ref, [alt], genome,
            flank_length=flank_length).variant
    return (chrom, start, ref, alt)


def variant_to_hgvs_name(chrom, offset, ref, alt, genome, transcript,
                         max_allele_length=4, use_counsyl=False):
    """
    Populate a HGVSName from a genomic coordinate.

    chrom: Chromosome name.
    offset: Genomic offset of allele.
    ref: Reference allele.
    alt: Alternate allele.
    genome: pygr compatible genome object.
    transcript: Transcript corresponding to allele.
    max_allele_length: If allele is greater than this use allele length.
    """
    # Convert VCF-style variant to HGVS-style.
    chrom, offset, ref, [alt] = normalize_variant(
        chrom, offset, ref, [alt], genome).variant
    chrom, offset, ref, alt, mutation_type = hgvs_normalize_variant(
        chrom, offset, ref, alt, genome, transcript)

    # Populate HGVSName parse tree.
    hgvs = HGVSName()

    # Populate coordinates.
    if not transcript:
        # Use genomic coordinate when no transcript is available.
        hgvs.kind = 'g'
        hgvs.start = offset
        hgvs.end = offset + len(ref) - 1
    else:
        # Use cDNA coordinates.
        hgvs.kind = 'c'
        is_single_base_indel = (
            (mutation_type == 'ins' and len(alt) == 1) or
            (mutation_type in ('del', 'delins', 'dup') and len(ref) == 1))

        if mutation_type == '>' or (use_counsyl and is_single_base_indel):
            # Use a single coordinate.
            hgvs.cdna_start = genomic_to_cdna_coord(transcript, offset)
            hgvs.cdna_end = hgvs.cdna_start
        else:
            # Use a range of coordinates.
            if mutation_type == 'ins':
                # Insert uses coordinates around the insert point.
                offset_start = offset - 1
                offset_end = offset
            else:
                offset_start = offset
                offset_end = offset + len(ref) - 1
            if transcript.strand == '-':
                offset_start, offset_end = offset_end, offset_start
            hgvs.cdna_start = genomic_to_cdna_coord(transcript, offset_start)
            hgvs.cdna_end = genomic_to_cdna_coord(transcript, offset_end)

    # Populate prefix.
    if transcript:
        hgvs.transcript = transcript.full_name
        hgvs.gene = transcript.gene.name

    # Convert alleles to transcript strand.
    if transcript and transcript.strand == '-':
        ref = revcomp(ref)
        alt = revcomp(alt)

    # Convert to allele length if alleles are long.
    ref_len = len(ref)
    alt_len = len(alt)
    if ((mutation_type == 'dup' and ref_len > max_allele_length) or
            (mutation_type != 'dup' and
             (ref_len > max_allele_length or alt_len > max_allele_length))):
        ref = str(ref_len)
        alt = str(alt_len)

    # Populate alleles.
    hgvs.mutation_type = mutation_type
    hgvs.ref_allele = ref
    hgvs.alt_allele = alt

    return hgvs


def format_hgvs_name(chrom, offset, ref, alt, genome, transcript,
                     use_prefix=True, use_gene=True, use_counsyl=False,
                     max_allele_length=4):
    """
    Generate a HGVS name from a genomic coordinate.

    chrom: Chromosome name.
    offset: Genomic offset of allele.
    ref: Reference allele.
    alt: Alternate allele.
    genome: pygr compatible genome object.
    transcript: Transcript corresponding to allele.
    use_prefix: Include a transcript/gene/chromosome prefix in HGVS name.
    use_gene: Include gene name in HGVS prefix.
    max_allele_length: If allele is greater than this use allele length.
    """
    hgvs = variant_to_hgvs_name(chrom, offset, ref, alt, genome, transcript,
                                max_allele_length=max_allele_length,
                                use_counsyl=use_counsyl)
    return hgvs.format(use_prefix=use_prefix, use_gene=use_gene,
                       use_counsyl=use_counsyl)


# ---------------------------------------------------------------------------
# Public HGVS normalisation and equivalence API
# ---------------------------------------------------------------------------

def normalize_hgvs_name(hgvs_name_str, use_3letter=True):
    """Return a canonical string form of *hgvs_name_str*.

    For protein names this normalises amino acid codes to the three-letter
    (or, if *use_3letter* is ``False``, single-letter) representation.
    cDNA and genomic names are reformatted using the standard
    :meth:`HGVSName.format` logic.

    Args:
        hgvs_name_str: A raw HGVS string, e.g. ``'NM_004380.2:p.R132H'``
            or ``'p.Arg132His'``.
        use_3letter: Controls amino acid notation for protein names.

    Returns:
        A canonical string representation of the HGVS name.

    Raises:
        HGVSParseError: If *hgvs_name_str* cannot be parsed.
    """
    parsed = HGVSName(hgvs_name_str)
    return parsed.format(use_3letter=use_3letter)


def hgvs_names_equal(name1, name2):
    """Return ``True`` if *name1* and *name2* represent the same variant.

    Both arguments may be HGVS strings or :class:`HGVSName` instances.
    Amino acid notation differences (1-letter vs 3-letter, stop codon
    representations ``*`` vs ``Ter``) are normalised before comparison so
    that, for example::

        hgvs_names_equal('p.R132H', 'p.Arg132His')  # True
        hgvs_names_equal('p.Arg132*', 'p.Arg132Ter')  # True

    Transcript prefixes are **ignored** for the purpose of this comparison.

    Args:
        name1: HGVS string or :class:`HGVSName`.
        name2: HGVS string or :class:`HGVSName`.

    Returns:
        ``bool``

    Raises:
        HGVSParseError: If either argument is a string that cannot be parsed.
    """
    if isinstance(name1, str):
        name1 = HGVSName(name1)
    if isinstance(name2, str):
        name2 = HGVSName(name2)
    return name1.equivalent(name2)
