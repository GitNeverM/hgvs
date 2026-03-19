"""
Helper functions.
"""

from __future__ import absolute_import
from __future__ import unicode_literals

from .models import Exon
from .models import Position
from .models import Transcript


def read_refgene(infile):
    """
    Iterate through a refGene file.

    GenePred extension format:
    http://genome.ucsc.edu/FAQ/FAQformat.html#GenePredExt

    Column definitions:
    0. uint undocumented id
    1. string name;             "Name of gene (usually transcript_id from GTF)"
    2. string chrom;                "Chromosome name"
    3. char[1] strand;              "+ or - for strand"
    4. uint txStart;                "Transcription start position"
    5. uint txEnd;                  "Transcription end position"
    6. uint cdsStart;               "Coding region start"
    7. uint cdsEnd;                 "Coding region end"
    8. uint exonCount;              "Number of exons"
    9. uint[exonCount] exonStarts;  "Exon start positions"
    10. uint[exonCount] exonEnds;   "Exon end positions"
    11. uint id;                    "Unique identifier"
    12. string name2;               "Alternate name (e.g. gene_id from GTF)"
    13. string cdsStartStat;        "enum('none','unk','incmpl','cmpl')"
    14. string cdsEndStat;          "enum('none','unk','incmpl','cmpl')"
    15. lstring exonFrames;         "Exon frame offsets {0,1,2}"
    """
    for line in infile:
        # Skip comments.
        if line.startswith('#'):
            continue
        row = line.rstrip('\n').split('\t')
        if len(row) != 16:
            raise ValueError(
                'File has incorrect number of columns '
                'in at least one line.')

        # Skip trailing ,
        exon_starts = list(map(int, row[9].split(',')[:-1]))
        exon_ends = list(map(int, row[10].split(',')[:-1]))
        exon_frames = list(map(int, row[15].split(',')[:-1]))
        exons = list(zip(exon_starts, exon_ends))

        yield {
            'chrom': row[2],
            'start': int(row[4]),
            'end': int(row[5]),
            'id': row[1],
            'strand': row[3],
            'cds_start': int(row[6]),
            'cds_end': int(row[7]),
            'gene_name': row[12],
            'exons': exons,
            'exon_frames': exon_frames
        }


def make_transcript(transcript_json):
    """
    Make a Transcript form a JSON object.
    """

    transcript_name = transcript_json['id']
    if '.' in transcript_name:
        name, version = transcript_name.split('.')
    else:
        name, version = transcript_name, None

    transcript = Transcript(
        name=name,
        version=int(version) if version is not None else None,
        gene=transcript_json['gene_name'],
        tx_position=Position(
            transcript_json['chrom'],
            transcript_json['start'],
            transcript_json['end'],
            transcript_json['strand'] == '+'),
        cds_position=Position(
            transcript_json['chrom'],
            transcript_json['cds_start'],
            transcript_json['cds_end'],
            transcript_json['strand'] == '+'))

    exons = transcript_json['exons']
    if not transcript.tx_position.is_forward_strand:
        exons = reversed(exons)

    for exon_number, (exon_start, exon_end) in enumerate(exons, 1):
        transcript.exons.append(
            Exon(transcript=transcript,
                 tx_position=Position(
                     transcript_json['chrom'],
                     exon_start,
                     exon_end,
                     transcript_json['strand'] == '+'),
                 exon_number=exon_number))

    # Carry over optional provenance metadata when present.
    for field in ('genome_build', 'source', 'is_mane_select',
                  'is_mane_plus_clinical', 'ensembl_transcript'):
        if field in transcript_json:
            setattr(transcript, field, transcript_json[field])

    return transcript


def read_transcripts(refgene_file):
    """
    Read all transcripts in a RefGene file.
    """
    transcripts = {}
    for trans in (make_transcript(record)
                  for record in read_refgene(refgene_file)):
        transcripts[trans.name] = trans
        transcripts[trans.full_name] = trans

    return transcripts


# ---------------------------------------------------------------------------
# Transcript lookup policies
# ---------------------------------------------------------------------------

class TranscriptLookup(object):
    """A lightweight, policy-driven transcript store.

    Supports lookup by full versioned name (``NM_004380.2``), by bare name
    (``NM_004380``), and optionally by MANE Select status when metadata is
    available.

    Usage::

        store = TranscriptLookup()
        store.load_refgene(open('hg38_refGene.txt'))
        tx = store.get('NM_004380')        # version-independent
        tx = store.get('NM_004380.2')      # exact version
        tx = store.get_mane_select('IDH1') # MANE Select for gene

    Loading GRCh38 RefSeq transcripts
    ----------------------------------
    Download ``hg38.refGene.txt.gz`` from UCSC::

        wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz
        gunzip refGene.txt.gz

    Then load with ``store.load_refgene(open('refGene.txt'))``.

    Loading MANE data
    -----------------
    Download the MANE summary file from NCBI::

        wget https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v*.summary.txt.gz

    Then call ``store.load_mane_summary(open('MANE.GRCh38.vX.summary.txt'))``.
    """

    def __init__(self):
        # Maps full versioned name → Transcript
        self._by_full_name = {}
        # Maps bare name → list of Transcripts (highest version first)
        self._by_name = {}

    def add(self, transcript):
        """Add a :class:`~pyhgvs.models.Transcript` to the store."""
        self._by_full_name[transcript.full_name] = transcript
        self._by_name.setdefault(transcript.name, []).append(transcript)
        # Keep highest version first
        self._by_name[transcript.name].sort(
            key=lambda t: t.version if t.version is not None else -1,
            reverse=True)

    def load_refgene(self, infile, genome_build=None):
        """Load transcripts from a RefGene (GenePred extension) file.

        Args:
            infile: Readable file-like object or path.
            genome_build: Optional string label, e.g. ``'GRCh38'`` or
                ``'hg19'``, stored as ``transcript.genome_build`` metadata.
        """
        if isinstance(infile, str):
            with open(infile) as f:
                return self.load_refgene(f, genome_build=genome_build)

        for record in read_refgene(infile):
            if genome_build is not None:
                record['genome_build'] = genome_build
            self.add(make_transcript(record))

    def load_mane_summary(self, infile):
        """Annotate loaded transcripts with MANE Select / Plus Clinical status.

        Call *after* :meth:`load_refgene` so that transcripts are already
        present in the store.

        The MANE summary TSV (available from NCBI) has at least these columns:
        ``#NCBI_GeneID``, ``Ensembl_Gene``, ``HGNC_ID``, ``symbol``,
        ``name``, ``RefSeq_nuc``, ``RefSeq_prot``, ``Ensembl_nuc``,
        ``Ensembl_prot``, ``MANE_status``.

        Args:
            infile: Readable file-like object or path.
        """
        if isinstance(infile, str):
            with open(infile) as f:
                return self.load_mane_summary(f)

        # header is populated from the first comment line (starts with '#').
        header = None
        for line in infile:
            if line.startswith('#'):
                # Parse column names from the header line
                header = line.lstrip('#').rstrip('\n').split('\t')
                continue
            if header is None:
                raise ValueError(
                    "MANE summary file has no header line (expected a line "
                    "beginning with '#' listing column names).")
            row = dict(zip(header, line.rstrip('\n').split('\t')))
            refseq_nuc = row.get('RefSeq_nuc', '').strip()
            ensembl_nuc = row.get('Ensembl_nuc', '').strip()
            mane_status = row.get('MANE_status', '').strip()
            if not refseq_nuc:
                continue

            is_select = mane_status == 'MANE Select'
            is_plus = mane_status == 'MANE Plus Clinical'

            # Look up the versioned transcript
            tx = self._by_full_name.get(refseq_nuc)
            if tx is None:
                # Try without version
                bare = refseq_nuc.split('.')[0]
                txs = self._by_name.get(bare, [])
                tx = txs[0] if txs else None

            if tx is not None:
                tx.is_mane_select = is_select
                tx.is_mane_plus_clinical = is_plus
                if ensembl_nuc:
                    tx.ensembl_transcript = ensembl_nuc

    def get(self, name, policy='latest'):
        """Retrieve a transcript by name.

        Args:
            name: Transcript accession, with or without version suffix
                (e.g. ``'NM_004380'`` or ``'NM_004380.2'``).
            policy: Lookup policy when the version is not specified:

                - ``'latest'`` (default): return the highest-version
                  transcript present in the store.
                - ``'mane_select'``: prefer the MANE Select transcript if
                  one is annotated; fall back to ``'latest'``.
                - ``'exact'``: only return a transcript when the full
                  versioned name matches exactly; return ``None`` otherwise.

        Returns:
            A :class:`~pyhgvs.models.Transcript` or ``None``.
        """
        # Exact versioned lookup (e.g. 'NM_004380.2')
        if '.' in name:
            tx = self._by_full_name.get(name)
            if tx is not None:
                return tx
            if policy == 'exact':
                return None
            # Fall through to bare-name lookup
            name = name.split('.')[0]

        if policy == 'exact':
            return None

        candidates = self._by_name.get(name, [])
        if not candidates:
            return None

        if policy == 'mane_select':
            for tx in candidates:
                if getattr(tx, 'is_mane_select', False):
                    return tx
            # Fall back to latest
            return candidates[0]

        # 'latest'
        return candidates[0]

    def get_mane_select(self, gene_symbol):
        """Return the MANE Select transcript for *gene_symbol*, or ``None``.

        This only works after :meth:`load_mane_summary` has been called.
        """
        for tx in self._by_full_name.values():
            if (getattr(tx, 'is_mane_select', False) and
                    tx.gene.name == gene_symbol):
                return tx
        return None

    def __contains__(self, name):
        return name in self._by_full_name or name in self._by_name

    def __len__(self):
        return len(self._by_full_name)
