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
    (``NM_004380``), by Ensembl transcript accession (``ENST00000123456`` or
    ``ENST00000123456.1``), and optionally by MANE Select status when metadata
    is available.

    Usage::

        store = TranscriptLookup()
        store.load_refgene(open('hg38_refGene.txt'))
        tx = store.get('NM_004380')            # version-independent
        tx = store.get('NM_004380.2')          # exact version
        tx = store.get_mane_select('IDH1')     # MANE Select for gene

        # Retrieve all transcripts for a gene, MANE Select first:
        txs = store.get_gene_transcripts('IDH1')
        # Retrieve transcript accessions only:
        ids = store.get_gene_transcripts('IDH1', return_id=True)

    After loading a MANE summary, Ensembl accessions are also accepted::

        store.load_mane_summary(open('MANE.GRCh38.vX.summary.txt'))
        tx = store.get('ENST00000415669')      # bare Ensembl ID
        tx = store.get('ENST00000415669.8')    # versioned Ensembl ID

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
        # Maps Ensembl accession (versioned and bare) → Transcript
        # Populated by load_mane_summary().
        self._by_ensembl = {}
        # Maps gene symbol → list of Transcripts (all versions).
        # Populated by add(); provides O(1) gene-level lookup without a
        # full-store linear scan.
        self._by_gene = {}

    def add(self, transcript):
        """Add a :class:`~pyhgvs.models.Transcript` to the store."""
        self._by_full_name[transcript.full_name] = transcript
        self._by_name.setdefault(transcript.name, []).append(transcript)
        # Keep highest version first
        self._by_name[transcript.name].sort(
            key=lambda t: t.version if t.version is not None else -1,
            reverse=True)
        # Index by gene symbol for get_gene_transcripts()
        gene_sym = transcript.gene.name
        self._by_gene.setdefault(gene_sym, []).append(transcript)

    def _index_ensembl(self, ensembl_nuc, transcript):
        """Register *transcript* under its Ensembl accession.

        Both the full versioned accession (``ENST00000123456.1``) and the
        bare accession (``ENST00000123456``) are stored.
        """
        if not ensembl_nuc:
            return
        self._by_ensembl[ensembl_nuc] = transcript
        bare = ensembl_nuc.split('.')[0]
        if bare != ensembl_nuc:
            # Register the bare (unversioned) form only if no mapping exists
            # yet for that bare ID.  First-registered entry wins so that
            # callers who load a single MANE summary get a stable result.
            existing = self._by_ensembl.get(bare)
            if existing is None:
                self._by_ensembl[bare] = transcript
            # else: keep the existing mapping

    def _transcript_length(self, transcript):
        """Return a stable transcript length for sorting.

        Prefer summed exon length (spliced transcript length). Fall back to
        transcript genomic span when exons are unavailable.
        """
        if getattr(transcript, 'exons', None):
            return sum(
                exon.tx_position.chrom_stop - exon.tx_position.chrom_start
                for exon in transcript.exons
            )
        return (
            transcript.tx_position.chrom_stop -
            transcript.tx_position.chrom_start
        )
    
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

        After this call, transcripts can also be retrieved by their Ensembl
        accession via :meth:`get`.

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
                    # Index by Ensembl accession so callers can look up
                    # transcripts using ENST IDs after loading MANE data.
                    self._index_ensembl(ensembl_nuc, tx)

    def get(self, name, policy='latest'):
        """Retrieve a transcript by name.

        Supports RefSeq accessions (``NM_004380``, ``NM_004380.2``) and
        Ensembl transcript accessions (``ENST00000123456``,
        ``ENST00000123456.1``).  Ensembl lookup is only available after
        :meth:`load_mane_summary` has been called.

        Args:
            name: Transcript accession, with or without version suffix.
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
        # ------------------------------------------------------------------
        # Ensembl transcript lookup (ENST accessions)
        # ------------------------------------------------------------------
        if name.startswith('ENST'):
            tx = self._by_ensembl.get(name)
            if tx is not None:
                return tx
            if policy == 'exact':
                return None
            # Try bare (without version)
            bare = name.split('.')[0]
            return self._by_ensembl.get(bare)

        # ------------------------------------------------------------------
        # RefSeq / generic lookup
        # ------------------------------------------------------------------
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

    def get_by_ensembl(self, ensembl_id):
        """Return the transcript associated with an Ensembl transcript ID.

        Accepts both versioned (``ENST00000123456.1``) and unversioned
        (``ENST00000123456``) identifiers.  Returns ``None`` when no match
        is found.

        .. note::
            Ensembl mappings are populated by :meth:`load_mane_summary`.
            Only transcripts present in the loaded RefGene data **and**
            referenced in the MANE summary are reachable via this method.
        """
        tx = self._by_ensembl.get(ensembl_id)
        if tx is not None:
            return tx
        # Try without version
        bare = ensembl_id.split('.')[0]
        return self._by_ensembl.get(bare)

    def get_mane_select(self, gene_symbol):
        """Return the MANE Select transcript for *gene_symbol*, or ``None``.

        This only works after :meth:`load_mane_summary` has been called.
        """
        for tx in self._by_full_name.values():
            if (getattr(tx, 'is_mane_select', False) and
                    tx.gene.name == gene_symbol):
                return tx
        return None

    def get_gene_transcripts(self, gene_name, sort_policy="mane",
                             return_id=False):
        """Return all transcripts for *gene_name*.

        Args:
            gene_name: HGNC gene symbol (e.g. ``'BRCA1'``).
            sort_policy: Controls the order of the returned list.

                - ``'mane'`` (default): MANE Select transcripts first, then
                  MANE Plus Clinical, then the rest; within each tier,
                  transcripts are sorted by genomic span (largest first).
                - ``'longest'``: Sort purely by genomic span, largest first.
                  MANE status is ignored.
                - ``None``: No ordering is applied; the list is returned
                  in insertion order.  Use this when you only need all
                  transcripts and do not care about their order (avoids the
                  sorting overhead).

            return_id: If ``True``, return a list of transcript accessions
                (``full_name``, e.g. ``'NM_007294.3'``) instead of
                :class:`~pyhgvs.models.Transcript` objects.

        Returns:
            A list of :class:`~pyhgvs.models.Transcript` objects when
            *return_id* is ``False``, or a list of ``str`` accessions when
            *return_id* is ``True``.  Returns an empty list when *gene_name*
            is not found in the store.

        Raises:
            ValueError: If *sort_policy* is not one of ``'mane'``,
                ``'longest'``, or ``'random'``.
        """
        _VALID_POLICIES = ('mane', 'longest', None)
        if sort_policy not in _VALID_POLICIES:
            raise ValueError(
                "sort_policy must be one of %r; got %r"
                % (_VALID_POLICIES, sort_policy))

        transcripts = self._by_gene.get(gene_name, [])
        if not transcripts:
            return []

        if sort_policy == 'mane':
            def _mane_key(tx):
                # Tier: 0 = MANE Select, 1 = MANE Plus Clinical, 2 = other.
                if getattr(tx, 'is_mane_select', False):
                    tier = 0
                elif getattr(tx, 'is_mane_plus_clinical', False):
                    tier = 1
                else:
                    tier = 2
                # Within the same tier, longer transcripts come first
                # (negate span so that larger values sort earlier).
                return (tier, -self._transcript_length(tx))
            transcripts = sorted(transcripts, key=_mane_key)

        elif sort_policy == 'longest':
            transcripts = sorted(transcripts, key=self._transcript_length, reverse=True)

        else:
            # sort_policy == None: return in insertion order; no sort needed.
            # Return a copy so callers cannot accidentally mutate the internal
            # index list.
            transcripts = list(transcripts)

        if return_id:
            return [tx.full_name for tx in transcripts]
        return transcripts

    def __contains__(self, name):
        if name.startswith('ENST'):
            return name in self._by_ensembl or name.split('.')[0] in self._by_ensembl
        return name in self._by_full_name or name in self._by_name

    def __len__(self):
        return len(self._by_full_name)
