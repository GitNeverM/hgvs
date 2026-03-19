HGVS variant name parsing and generation
========================================

The Human Genome Variation Society (HGVS) promotes the discovery and
sharing of genetic variation in the human population.  As part of facilitating
variant sharing, the society has produced a series of recommendations for
how to name and refer to variants within research publications and clinical
settings.  A compilation of these recommendations is available on their
[website](http://www.hgvs.org/mutnomen/recs.html).

This library provides a simple Python API for parsing, formatting, and
normalizing HGVS names.  Surprisingly, there are many non-trivial steps
necessary in handling HGVS names and therefore there is a need for well tested
libraries that encapsulate these steps.

## HGVS name example

In most next-generation sequencing applications, variants are first
discovered and described in terms of their genomic coordinates such as
chromosome 7, position 117,199,563 with reference allele `G` and
alternative allele `T`.  According to the HGVS standard, we can
describe this variant as `NC_000007.13:g.117199563G>T`.  The first
part of the name is a RefSeq ID `NC_000007.13` for chromosome 7
version 13.  The `g.` denotes that this is a variant described in
genomic (i.e. chromosomal) coordinates.  Lastly, the chromosomal position,
reference allele, and alternative allele are indicated.  For simple
single nucleotide changes the `>` character is used.

More commonly, a variant will be described using a cDNA or protein
style HGVS name.  In the example above, the variant in cDNA style is
named `NM_000492.3:c.1438G>T`.  Here again, the first part of the name
refers to a RefSeq sequence, this time mRNA transcript `NM_000492`
version `3`.  Optionally, the gene name can also be given as
`NM_000492.3(CFTR)`.  The `c.` indicates that this is a cDNA name, and
the coordinate indicates that this mutation occurs at position 1438
along the coding portion of the spliced transcript (i.e. position 1 is
the first base of `ATG` translation start codon).  Briefly, the
protein style of the variant name is `NP_000483.3:p.Gly480Cys` which
indicates the change in amino-acid coordinates (`480`) along an
amino-acid sequence (`NP_000483.3`) and gives the reference and
alternative amino-acid alleles (`Gly` and `Cys`, respectively).

The standard also specifies custom name formats for many mutation
categories such as insertions (`NM_000492.3:c.1438_1439insA`),
deletions (`NM_000492.3:c.1438_1440delGGT`),
duplications (`NM_000492.3:c.1438_1440dupGGT`), and several
other more complex genomic rearrangements.

While many of these names appear to be simple to parse or generate,
there are many corner cases, especially with cDNA HGVS names.  For
example, variants before the start codon should have negative cDNA
coordinates (`NM_000492.3:c.-4G>C`), and variants after the stop codon
also have their own format (`NM_000492.3:c.*33C>T`).  Variants within
introns are indicated by the closest exonic base with an additional
genomic offset such as `NM_000492.3:4243-20A>G` (the variant is 20
bases in the 5' direction of the cDNA coordinate 4243).  Lastly, all
coordinates and alleles are specified on the strand of the
transcript.  This library properly handles all logic necessary to
convert genomic coordinates to and from HGVS cDNA coordinates.

Another important consideration of any library that handles HGVS names
is variant normalization.  The HGVS standard aims to provide "uniform
and unequivocal" description of variants.  Namely, two people
discovering a variant should be able to arrive at the same name for
it.  Such a property is very useful for checking whether a variant has
been seen before and connecting all known relevant information.  For
SNPs, this property is fairly easy to achieve.  However, for
insertions and deletions (indels) near repetitive regions, many indels
are equivalent (e.g. it doesn't matter which `AT` in a run of
`ATATATAT` was deleted). The VCF file format has chosen to uniquely
specify such indels by using the most left-aligned genomic coordinate.
Therefore, compliant variant callers that output VCF will have applied
this normalization.  The HGVS standard also specifies a normalization
for such indels. However, it states that indels should use the most 3'
position in a transcript.  For genes on the positive strand, this is
the opposite direction specified by VCF.  This library properly
implements both kinds of variant normalization and allows easy
conversion between HGVS and VCF style variants.  It also handles
many other cases of normalization (e.g. the HGVS standard recommends
indicating an insertion with the `dup` notation instead of `ins`
if it can be represented as a tandem duplication).

## Example usage

Below is a minimal example of parsing and formatting HGVS names.  In
addition to the name itself, two other pieces of information are
needed: the genome sequence (needed for normalization), and the
transcript model or a callback for fetching the transcript model
(needed for transcript coordinate calculations).  This library makes
as few assumptions as possible about how this external data is stored.
In this example, the genome sequence is read using the `pyfaidx` library
and transcripts are read from a RefSeqGenes flat-file using methods
provided by `hgvs`.

```python
import pyhgvs as hgvs
import hgvs.utils as hgvs_utils
from pyfaidx import Fasta

# Read genome sequence using pyfaidx.
genome = Fasta('hg19.fa')

# Read RefSeq transcripts into a python dict.
with open('hgvs/data/genes.refGene') as infile:
    transcripts = hgvs_utils.read_transcripts(infile)

# Provide a callback for fetching a transcript by its name.
def get_transcript(name):
    return transcripts.get(name)

# Parse the HGVS name into genomic coordinates and alleles.
chrom, offset, ref, alt = hgvs.parse_hgvs_name(
    'NM_000352.3:c.215A>G', genome, get_transcript=get_transcript)
# Returns variant in VCF style: ('chr11', 17496508, 'T', 'C')
# Notice that since the transcript is on the negative strand, the alleles
# are reverse complemented during conversion.

# Format an HGVS name.
chrom, offset, ref, alt = ('chr11', 17496508, 'T', 'C')
transcript = get_transcript('NM_000352.3')
hgvs_name = hgvs.format_hgvs_name(
    chrom, offset, ref, alt, genome, transcript)
# Returns 'NM_000352.3(ABCC8):c.215A>G'
```

The `hgvs` library can also perform just the parsing step and provide
a parse tree of the HGVS name.

```python
import pyhgvs as hgvs

hgvs_name = hgvs.HGVSName('NM_000352.3:c.215-10A>G')

# fields of the HGVS name are available as attributes:
#
# hgvs_name.transcript = 'NM_000352.3'
# hgvs_name.kind = 'c'
# hgvs_name.mutation_type = '>'
# hgvs_name.cdna_start = hgvs.CDNACoord(215, -10)
# hgvs_name.cdna_end = hgvs.CDNACoord(215, -10)
# hgvs_name.ref_allele = 'A'
# hgvs_name.alt_allele = 'G'
```

## Install

This library can be installed using the `setup.py` file as follows:

```sh
python setup.py install
```

## Tests

Test cases can be run by running

```sh
python setup.py nosetests
```

## Requirements

This library requires at least Python 2.6, but otherwise has no
external dependencies.

The library does assume that genome sequence is available through a `pyfaidx`
compatible `Fasta` object. For an example of writing a wrapper for
a different genome sequence back-end, see
[hgvs.tests.genome.MockGenome](pyhgvs/tests/genome.py).

---

## New features (GitNeverM fork)

### Protein HGVS: 1-letter and 3-letter amino acid support

Both single-letter (`p.R132H`) and three-letter (`p.Arg132His`) amino acid
notation are now parsed and can be compared for semantic equivalence:

```python
import pyhgvs as hgvs

# Parse single-letter notation
n1 = hgvs.HGVSName('p.R132H')
print(n1.ref_allele)   # 'R'
print(n1.alt_allele)   # 'H'

# Parse three-letter notation
n2 = hgvs.HGVSName('p.Arg132His')
print(n2.ref_allele)   # 'Arg'

# Semantic equivalence comparison (ignores notation and transcript prefix)
hgvs.hgvs_names_equal('p.R132H', 'p.Arg132His')              # True
hgvs.hgvs_names_equal('p.Arg132*', 'p.Arg132Ter')            # True
hgvs.hgvs_names_equal('NM_004380.2:p.R132H', 'p.Arg132His')  # True

# Direct method comparison
n1.equivalent(n2)  # True
```

### Canonical HGVS normalisation

```python
hgvs.normalize_hgvs_name('p.R132H')                    # 'p.Arg132His'
hgvs.normalize_hgvs_name('p.Arg132His', use_3letter=False)  # 'p.R132H'
hgvs.normalize_hgvs_name('c.395G>A')                   # 'c.395G>A' (unchanged)
```

### Configurable protein formatting

```python
n = hgvs.HGVSName('p.Arg132His')
n.format()                  # 'p.Arg132His'   (default: 3-letter)
n.format(use_3letter=False) # 'p.R132H'       (1-letter output)

# Works from 1-letter parsed input too
n2 = hgvs.HGVSName('p.R132H')
n2.format(use_3letter=True) # 'p.Arg132His'
```

### Amino acid conversion utilities

```python
from pyhgvs import aa1_to_aa3, aa3_to_aa1, normalize_aa_allele

aa1_to_aa3('R')   # 'Arg'
aa3_to_aa1('His') # 'H'
aa1_to_aa3('*')   # 'Ter'  (stop codon)
aa3_to_aa1('Ter') # '*'

normalize_aa_allele('R',   use_3letter=True)  # 'Arg'
normalize_aa_allele('Arg', use_3letter=False) # 'R'
```

### Structured exception hierarchy

All library errors are now subclasses of `HGVSError(ValueError)`:

| Class | Meaning |
|---|---|
| `HGVSError` | Base class for all library errors |
| `HGVSParseError` | Parse failure (alias: `InvalidHGVSName`) |
| `HGVSTranscriptError` | Transcript resolution failure |
| `HGVSFormattingError` | Formatting failure |
| `HGVSNormalizationError` | Normalisation failure |
| `HGVSInvalidAminoAcidError` | Unrecognised amino acid code |

Existing code that catches `InvalidHGVSName` or `ValueError` continues to work.

```python
from pyhgvs import HGVSParseError, InvalidHGVSName  # same object

try:
    hgvs.HGVSName('p.ZZZ')
except InvalidHGVSName as e:   # or HGVSParseError, or ValueError
    print(e)
```

### GRCh38 and MANE-aware transcript lookup

The `TranscriptLookup` class provides a lightweight, policy-driven transcript
store that supports version-aware lookups and optional MANE annotation.

#### Downloading data files

**GRCh38 RefSeq transcripts** (from UCSC):
```sh
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz
gunzip refGene.txt.gz
```

**MANE summary** (from NCBI — check for the latest version):
```sh
wget https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.3.summary.txt.gz
gunzip MANE.GRCh38.v1.3.summary.txt.gz
```

#### Usage

```python
from pyhgvs.utils import TranscriptLookup

store = TranscriptLookup()

# Load GRCh38 RefSeq transcripts
with open('refGene.txt') as f:
    store.load_refgene(f, genome_build='GRCh38')

# Optionally annotate with MANE status
with open('MANE.GRCh38.v1.3.summary.txt') as f:
    store.load_mane_summary(f)

# Lookup policies
tx = store.get('NM_004380')          # latest version
tx = store.get('NM_004380.2')        # exact version
tx = store.get('NM_004380', policy='mane_select')  # prefer MANE Select
tx = store.get('NM_004380', policy='exact')        # versioned only

# Ensembl transcript lookup (available after load_mane_summary)
tx = store.get('ENST00000415669')    # bare Ensembl ID
tx = store.get('ENST00000415669.8')  # versioned Ensembl ID

# Direct MANE Select lookup by gene symbol
mane_tx = store.get_mane_select('IDH1')
```

After calling `load_mane_summary`, Ensembl transcript accessions (``ENST…``)
become valid lookup keys because the MANE summary file maps each RefSeq
transcript to its Ensembl counterpart.  Both versioned (``ENST00000415669.8``)
and bare (``ENST00000415669``) forms are accepted.

```python
# Use an Ensembl transcript with parse_hgvs_name
chrom, start, ref, alt = hgvs.parse_hgvs_name(
    'ENST00000357654:c.2207A>C',
    genome,
    get_transcript=store.get)
```

The `TranscriptLookup` object can be used as the `get_transcript` callback in
`parse_hgvs_name`:

```python
import pyhgvs as hgvs

chrom, start, ref, alt = hgvs.parse_hgvs_name(
    'NM_004380.2:c.395G>A',
    genome,
    get_transcript=store.get)
```

### Duplication (dup) internal representation

When a `dup` variant is parsed (e.g. `c.101dupA` or `c.1000_1002dupATG`),
the library stores:

| Field | Value |
|---|---|
| `mutation_type` | `'dup'` |
| `ref_allele` | The duplicated sequence (`'A'`, `'ATG'`, or `''` if absent) |
| `alt_allele` | Same as `ref_allele` (the inserted sequence) |

`get_ref_alt()` returns `('', ref_allele)` — an empty reference and the
duplicated sequence as the alternate — representing the event semantically
as a pure insertion.  This is then used by `get_vcf_allele()` to build the
left-padded VCF allele pair.

When no explicit sequence is given (e.g. `c.101dup`), both `ref_allele` and
`alt_allele` are empty strings and the VCF allele cannot be computed without
a genome reference.

Inversions (`c.100_102inv`, `g.100_102inv`) are now also parsed and
formatted correctly.

## Tests

Run the test suite with:

```sh
python -m pytest pyhgvs/tests/
```

The `test_protein_matching.py` module covers all new protein parsing,
formatting, normalisation and equivalence features.
