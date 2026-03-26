"""Tecplot ASCII file reader and writer."""

# --------------------------------------------------
# import necessary libraries
# --------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
import logging
import re
import numpy as np
from pathlib import Path
from pprint import pformat

logger = logging.getLogger(__name__)

# --------------------------------------------------
# data class for tecplot zone information
# --------------------------------------------------



@dataclass
class TecplotZone:
    """Metadata for a single Tecplot zone (dimensions, packing, etc.)."""
    name: str
    I: int
    J: int
    K: int
    datapacking: str  # "POINT" expected
    dt: list[str]  # list of Tecplot DT tokens, e.g. ["DOUBLE", ...]
    zone_type: str | None = None



# --------------------------------------------------
# data class for tecplot data
# --------------------------------------------------



@dataclass
class TecplotData:
    """Parsed Tecplot ASCII dataset (header + zone + data array)."""
    # title string from TITLE
    title: str
    # list of variable names from VARIABLES
    variables: list[str]
    # zone information (see TecplotZone class above)
    zone: TecplotZone
    # data shaped as (K, J, I, nvars)
    data: np.ndarray
    # a quick lookup from variable name
    var_index: dict[str, int]
    # user-extensible alias map: canonical -> actual header name
    aliases: dict[str, str] = field(default_factory=dict)
    
    # --------------------------------------------------
    # methods
    # --------------------------------------------------
    
    # post initialization to build normalized header map
    # -> this is automatically called after the dataclass TecplotData is generated

    def __post_init__(self) -> None:
        
        # build a normalized name dictionary -> original header map for resilient lookups
        self._norm_to_header: dict[str, str] = {}
        
        # loop over all variable headers and populate the map
        for h in self.variables:
            
            # refer normalized name to original header
            # (note: normalized names have been lowercased and stripped of spaces/underscores/dots)
            self._norm_to_header[_normalize(h)] = h
            
        
        # add some default aliases for convenience
        self.add_default_aliases()
        

    
    # safely bind canonical -> first matching header (if any) present in this file

    def safely_bind(self, canonical: str, *candidates: str) -> None:
        
        ncanon = _normalize(canonical)
        
        # loop over candidates in order
        for cand in candidates:
            
            # is this candidate header present in this file (using normalized header map)?
            h = self._norm_to_header.get(_normalize(cand))
            
            if h:
                # register the alias: now tp.field(canonical) resolves to header h
                self.aliases[ncanon] = h
                
                return
            
        

    
    # add some default aliases for convenience
    
    # example: alpi will be mapped to -im(alpha), imag(alpha), alpha_i if that header is present

    def add_default_aliases(self) -> None:
        
        # seed a few convenient short names
        # note: the variable names are normalized before matching
        # therefore "X-Location", "x-location", "x location" all map to the same normalized name -> xlocation
        # cases are ignored
        
        self.safely_bind("s", "s", "x", "x-location")
        self.safely_bind("freq", "freq.", "frequency")
        self.safely_bind("beta", "beta")
        self.safely_bind("alpi", "-im(alpha)", "imag(alpha)", "alpha_i", "alpha_imag")
        self.safely_bind("alpr", "re(alpha)", "real(alpha)", "alpha_r", "alpha_real")
        self.safely_bind("ampl", "amp", "amplitude")
        self.safely_bind("nfac", "nfac", "nfactor", "n-factor", "n")
        self.safely_bind("nfac2", "nfac2")
        self.safely_bind("nfac3", "nfac3")

    
    # add an alias from canonical name to actual header name (can be done by user to add to defaults)

    def add_alias(self, canonical: str, actual_header: str) -> None:
        
        self.aliases[_normalize(canonical)] = actual_header

    
    # return a formatted table of canonical aliases -> actual header (and index)

    def aliases_table(self) -> str:
        
        # build rows: canonical, header, col_idx
        rows = []
        
        for canonical in sorted(self.aliases.keys()):
            header = self.aliases[canonical]
            idx = self.var_index.get(header, -1)
            rows.append((canonical, header, idx))
        
        # Column widths
        w0 = max([len("alias")] + [len(r[0]) for r in rows]) if rows else len("alias")
        w1 = max([len("header")] + [len(r[1]) for r in rows]) if rows else len("header")
        w2 = len("col")
        
        # Header + divider
        lines = []
        lines.append(f"{'alias'.ljust(w0)}  {'header'.ljust(w1)}  {'col'}")
        lines.append(f"{'-' * w0}  {'-' * w1}  {'-' * w2}")
        
        # rows
        for a, h, c in rows:
            lines.append(f"{a.ljust(w0)}  {h.ljust(w1)}  {c}")
            
        
        return "\n".join(lines)

    
    # return a formatted table of all headers and their column indices (for context)

    def headers_table(self) -> str:
        
        w0 = (
            max([len("header")] + [len(h) for h in self.variables])
            if self.variables
            else len("header")
        )
        
        lines = []
        lines.append(f"{'header'.ljust(w0)}  col")
        lines.append(f"{'-' * w0}  ---")
        
        for h in self.variables:
            lines.append(f"{h.ljust(w0)}  {self.var_index.get(h, -1)}")
        return "\n".join(lines)

    
    # print headers and alias tables (for debugging)

    def debug_aliases(self, file=None) -> None:
        
        if file is None:
            import sys as _sys

            file = _sys.stdout
        
        print("\n[TecplotData] headers:", file=file)
        
        print(self.headers_table(), file=file)
        
        if self.aliases:
            print("\n[TecplotData] aliases:", file=file)
            print(self.aliases_table(), file=file)
        else:
            print("\n[TecplotData] aliases: <none>", file=file)
        

    
    # resolve a requested variable name to an actual header name
    # when tp.field() or tp.fields() method is called

    def _resolve(self, key: str) -> str:
        """Resolve a requested variable name to an actual header name.
        Tries: exact header, alias, normalized-header, then fuzzy suggestions.
        """
        # 1) Exact match first
        if key in self.var_index:
            return key
        # 2) Alias map (normalized canonical -> actual header)
        nkey = _normalize(key)
        if nkey in self.aliases:
            return self.aliases[nkey]
        # 3) Normalized header map
        if nkey in self._norm_to_header:
            return self._norm_to_header[nkey]
        # 4) Fuzzy (very light) suggestions
        lower = key.lower()
        candidates = []
        for h in self.variables:
            hl = h.lower()
            if hl.startswith(lower) or lower in hl:
                candidates.append(h)
        hint = ("; did you mean: " + ", ".join(candidates[:5])) if candidates else ""
        raise KeyError(f"Variable '{key}' not found in Tecplot data{hint}.")

    def field(self, key: str) -> np.ndarray:
        """Return a (K,J,I) view for a single variable name (zero-copy)."""
        header = self._resolve(key)
        idx = self.var_index.get(header)
        if idx is None:
            raise KeyError(
                f"Resolved header '{header}' has no index—this should not happen."
            )
        return self.data[..., idx]

    def fields(self, *keys: str) -> np.ndarray:
        """Return a (K,J,I,N) view stacking multiple variables along the last axis."""
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = tuple(keys[0])
        arrays = [self.field(k) for k in keys]
        return np.stack(arrays, axis=-1)

    



# --------------------------------------------------
# compile regex patterns for parsing (regex pattern compilation is expensive so it is done once and used repatedly)
# --------------------------------------------------

# regex to match quoted strings, e.g. "Velocity", "freq", etc.
# it passes in a literal string (r'...') so that backslashes and spaces are treated literally
# "([^"]*)" explanation:
# " -> matches the opening double quote as usual in tecplot ascii files
# ([^"]*) -> capturing group:
#   [...] -> capture character class
#   ^" -> inside the brackets = “anything except a double quote”
#   * -> zero or more of those.
# meaning: “capture all characters until the next double-quote”
# " -> matches the closing double-quote

# useful regex expression for parsing TITLE, VARIABLES and ZONE header information for tecplot ascii files
_RE_QUOTED_STRING = re.compile(r'"([^"]*)"')

# --------------------------------------------------
# helper: normalize variable names for resilient lookup
# --------------------------------------------------



def _normalize(s: str) -> str:
    """Lowercase, strip spaces/underscores/dots and common punctuation for robust matching."""
    return re.sub(r"[\s_\.\-\(\)]+", "", s.strip().lower())



# --------------------------------------------------
# parse VARIABLES header information (replaced with _parse_variables_block)
# --------------------------------------------------



def _parse_variables_block(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """
    Parse a VARIABLES header that may span multiple lines.

    Returns (variables, next_index_to_read_from) where next_index_to_read_from
    should point at the first non-VARIABLES header line (typically the ZONE line).
    """
    
    # --------------------------------------------------
    # start with the VARIABLES line
    # --------------------------------------------------
    
    var_lines = lines[start_idx].strip()
    
    i = start_idx + 1
    
    # accumulate subsequent lines until we hit the ZONE header or run out of lines
    while i < len(lines):
        
        L = lines[i].strip()
        
        if L.upper().startswith("ZONE"):
            
            break
        
        # Keep appending (this allows one-name-per-line style)
        var_lines += " " + L
        
        i += 1
    
    # find all quoted variable names across the whole variables block stored in var_lines
    # this uses the compiled regex pattern defined above
    
    variables = _RE_QUOTED_STRING.findall(var_lines)
    
    # fallback if the previous regex found nothing
    if not variables:
        
        # remove the leading "VARIABLES = " token from var_lines
        var_lines_split = var_lines.split("=", 1)[-1]
        
        # split on commas or whitespace
        tokens = re.split(r"[,\s]+", var_lines_split.strip())
        
        # loop over tokens and keep those that are not empty and not the word VARIABLES
        variables = [t for t in tokens if t and t.upper() != "VARIABLES"]
    
    # return variables list and the next index to read from (after variables block)
    return variables, i



# --------------------------------------------------
# parse ZONE header information
# --------------------------------------------------



def _parse_zone_header(lines: list[str], idx_start: int) -> tuple[TecplotZone, int]:
    
    """
    Parse ZONE ... plus following lines that may include I=,J=,K=, DATAPACKING=, DT=(...)
    Returns (zone, next_index_to_read_from).
    """
    
    zone_lines = lines[idx_start].strip()
    
    # ZONE T="LST-FORTRAN (nfactors)" ...
    m = re.search(r'ZONE\s+T="([^"]*)"', zone_lines, re.IGNORECASE)
    
    name = m.group(1) if m else "ZONE"
    
    # intialize defaults
    I = J = K = 1
    
    datapacking = "POINT"
    
    dt: list[str] = []
    
    # store the entire zone description for parsing
    i = idx_start + 1
    
    while i < len(lines):
        
        L = lines[i].strip()
        
        # numeric lines typically start with digit, sign, or decimal
        if re.match(r"^[\s\+\-0-9\.]", L):
            break
        zone_lines += " " + L
        
        i += 1
    
    # parse I/J/K
    dims = {}
    for key in ("I", "J", "K"):
        mm = re.search(rf"\b{key}\s*=\s*(\d+)", zone_lines, re.IGNORECASE)
        if mm:
            dims[key] = int(mm.group(1))

    I = dims.get("I", 1)
    J = dims.get("J", 1)
    K = dims.get("K", 1)

    # DATAPACKING
    mdp = re.search(r"DATAPACKING\s*=\s*(\w+)", zone_lines, re.IGNORECASE)
    if mdp:
        datapacking = mdp.group(1).upper()

    # DT=(DOUBLE DOUBLE ...)
    mdt = re.search(r"DT\s*=\s*\(([^)]*)\)", zone_lines, re.IGNORECASE)
    if mdt:
        dt = [tok.strip().upper() for tok in mdt.group(1).split() if tok.strip()]
    
    # create zone object
    zone = TecplotZone(name=name, I=I, J=J, K=K, datapacking=datapacking, dt=dt)
    
    # return zone object and next index to read from (start of data)
    return zone, i



# --------------------------------------------------
# main function to read tecplot ascii file
# --------------------------------------------------




# --------------------------------------------------
# helper function to read plain ASCII files (no Tecplot headers)
# --------------------------------------------------
def _read_plain_ascii(
    lines: list[str], path: str | Path
) -> TecplotData:
    """Read plain ASCII data file without Tecplot headers (like Eigenvalues_*)"""

    # Filter out empty lines and comments
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data found in {path}")

    # Parse first line to determine number of columns
    first_row = data_lines[0].split()
    ncols = len(first_row)
    nrows = len(data_lines)

    logger.debug("parsing %d rows, %d columns", nrows, ncols)

    # Parse all data
    data_array = np.zeros((nrows, ncols))
    for i, line in enumerate(data_lines):
        try:
            values = [float(x) for x in line.split()]
            if len(values) != ncols:
                raise ValueError(
                    f"Row {i + 1} has {len(values)} values, expected {ncols}"
                )
            data_array[i, :] = values
        except ValueError as e:
            raise ValueError(f"Error parsing row {i + 1} in {path}: {e}")

    # Create variable names v1, v2, v3, etc.
    variables = [f"v{i + 1}" for i in range(ncols)]

    # Create TecplotData structure
    # Reshape data to (K=1, J=1, I=nrows, nvars=ncols) format
    data_reshaped = data_array.reshape(1, 1, nrows, ncols)

    # Create zone info
    zone = TecplotZone(
        name="PlainASCII", I=nrows, J=1, K=1, datapacking="POINT", dt=["DOUBLE"] * ncols
    )

    # Build variable index
    var_index = {var: i for i, var in enumerate(variables)}

    return TecplotData(
        title=f"Plain ASCII: {Path(path).name}",
        variables=variables,
        zone=zone,
        data=data_reshaped,
        var_index=var_index,
    )



# --------------------------------------------------
# main function to read tecplot ascii file
# --------------------------------------------------
def read_tecplot_ascii(path: str | Path) -> TecplotData:
    
    """
    read a Tecplot ASCII file with one ordered ZONE
    returns TecplotData with data shaped (K, J, I, nvars).
    """
    
    logger.info("reading tecplot ascii file '%s'", path)
    
    # --------------------------------------------------
    # store path in p for convenience -> also typecasting to path object providing useful methods
    # --------------------------------------------------
    p = Path(path)
    
    # --------------------------------------------------
    # some sanity checks
    # --------------------------------------------------
    if p.suffix.lower() not in {".dat", ".plt", ".tec"}:
        
        logger.warning("File extension '%s' not recognized as tecplot ascii (.dat, .plt, .tec)", p.suffix)
        
    
    # try and read entire file as text, catching decode errors
    try:
        txt = p.read_text()
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1,
            f"File does not appear to be ascii text: {p}"
        )
    
    # --------------------------------------------------
    # split into lines
    # --------------------------------------------------
    
    lines = txt.splitlines()

    # --------------------------------------------------
    # check if this is a plain ASCII file (no Tecplot headers)
    # --------------------------------------------------
    
    has_title = any(L.strip().upper().startswith("TITLE") for L in lines)
    has_variables = any(L.strip().upper().startswith("VARIABLES") for L in lines)
    has_zone = any(L.strip().upper().startswith("ZONE") for L in lines)

    if not (has_title or has_variables or has_zone):
        # This is a plain ASCII data file (like Eigenvalues_*)
        logger.debug("detected plain ASCII file (no Tecplot headers)")
        return _read_plain_ascii(lines, path)
    
    # --------------------------------------------------
    # parse TITLE
    # --------------------------------------------------
    
    title_line = next((L for L in lines if L.strip().upper().startswith("TITLE")), "")
    
    # regex explanation for r'TITLE\s*=\s*"([^"]*)"'
    # r -> raw string (so backslashes and spaces are literal)
    # TITLE -> matches the literal word TITLE (due to re.IGNORECASE it can match title, Title, etc.)
    # \s*:
    # \s -> any whitespace (spaces, tabs, etc.)
    # * -> zero or more repetitions -> match any amount of whitespace after TITLE.
    # = -> Matches a literal = sign.
    # \s* -> as before zero or more white spaces
    # " -> matches the opening double quote as usual in tecplot ascii files
    # ([^"]*) -> capturing group:
    #   [...] -> capture character class
    #   ^" -> inside the brackets = “anything except a double quote”.
    #   * -> zero or more of those.
    # meaning: “capture all characters until the next double-quote.”
    # " -> matches the closing double-quote.
    # returns captured object m or None if no match
    
    m = re.search(r'TITLE\s*=\s*"([^"]*)"', title_line, re.IGNORECASE)
    
    # the matchin object has the following atributes:
    # m.string = the string passed to re.search()
    # m.re = the compiled regex pattern
    # m.pos = starting position passed to search()
    # m.endpos = ending position passed to search()
    # m.lastindex = integer index of last matched capturing group
    # m.lastgroup = name of last matched capturing group
    # m.group() = the entire matched string
    # m.start() = starting position of the match
    # m.end() = ending position of the match
    # m.span() = (m.start(), m.end())
    # m.groups() = tuple of all captured groups
    # m.group(n) = nth captured group -> note because we used re.search with one capturing group, m.group(1) is the title string
    
    title = m.group(1) if m else ""
    
    # print output for user
    logger.info("TITLE = '%s'", title)
    
    # --------------------------------------------------
    # parse VARIABLES
    # --------------------------------------------------
    
    # find the VARIABLES line
    var_idx = next(
        (i for i, L in enumerate(lines) if L.strip().upper().startswith("VARIABLES")),
        None,
    )
    
    if var_idx is None:
        
        logger.error("VARIABLES header not found")
        raise ValueError("VARIABLES header not found")
    
    # parse the VARIABLES block (which may span multiple lines) and get the next index to read from
    variables, idx_continue = _parse_variables_block(lines, var_idx)
    
    # number of variables
    nvars = len(variables)
    
    logger.debug("found %d variables:", nvars)
    for i, v in enumerate(variables):
        logger.debug("  var %d: '%s'", i, v)
    
    # --------------------------------------------------
    # create name to column index mapping
    # --------------------------------------------------
    
    name_to_col = {v: i for i, v in enumerate(variables)}
    
    # --------------------------------------------------
    # parse ZONE
    # --------------------------------------------------
    
    # start with first line after the VARIABLES block if it is a ZONE line
    if idx_continue < len(lines) and lines[idx_continue].strip().upper().startswith(
        "ZONE"
    ):
        zone_idx = idx_continue
    else:
        zone_idx = next(
            (
                i
                for i in range(idx_continue, len(lines))
                if lines[i].strip().upper().startswith("ZONE")
            ),
            None,
        )
    
    # error if no ZONE line found
    if zone_idx is None:
        raise ValueError("ZONE header not found")
    
    # parse the ZONE block (which may span multiple lines) and get the next index to read from
    zone, idx_data_start = _parse_zone_header(lines, zone_idx)
    
    # debug output
    logger.debug("zone information:")
    logger.debug(pformat(zone))
    
    # compute number of points
    npts = zone.I * zone.J * zone.K
    
    # --------------------------------------------------
    # read numerical data from idx_data_start (ignore line breaks)
    # --------------------------------------------------
    
    # handle scientific notation, signs, etc.
    numeric_tokens: list[float] = []
    
    # loop over lines starting at idx_data_start until we have enough tokens
    for L in lines[idx_data_start:]:
        
        # strip leading/trailing whitespace from current line
        L = L.strip()
        
        # skip empty lines
        if not L:
            continue

        # stop if we hit a new zone header or other non-data keyword
        L_upper = L.upper()
        if L_upper.startswith("ZONE") or L_upper.startswith("TITLE") or L_upper.startswith("VARIABLES"):
            break
        
        # split by whitespace or commas (both are common delimiters in tecplot ascii files)
        parts = re.split(r"[,\s]+", L.strip())
        
        # convert each part to float; be strict and fail on non-numeric tokens
        for part in parts:
            try:
                numeric_tokens.append(float(part))
            except ValueError:
                # handle Fortran-style floats with missing 'E'
                # e.g. '0.901560349186166-100' -> '0.901560349186166E-100'
                fixed = re.sub(r"(\d)([+-])(\d)", r"\1E\2\3", part, count=1)
                try:
                    numeric_tokens.append(float(fixed))
                except ValueError as e:
                    raise ValueError(
                        f"Non-numeric token '{part}' encountered in data section; "
                        "Tecplot ASCII is expected to be whitespace-separated numeric values. "
                        "please re-export or clean the file."
                    ) from e

    expected = npts * nvars
    
    if len(numeric_tokens) < expected:
        raise ValueError(
            f"Not enough data: expected {expected} floats, found {len(numeric_tokens)}"
        )
    if len(numeric_tokens) > expected:
        # be lenient — trim extras (some writers append newlines or extra zeros)
        numeric_tokens = numeric_tokens[:expected]
    
    arr = np.asarray(numeric_tokens, dtype=float).reshape(npts, nvars)
    
    # Tecplot Ordered POINT layout default is i fastest, then j, then k.
    # Reshape into (K, J, I, nvars)
    data_reshaped = arr.reshape(zone.K, zone.J, zone.I, nvars, order="C")
    
    
    # --------------------------------------------------
    # return the tecplot data class
    # --------------------------------------------------
    
    return TecplotData(
        title=title,
        variables=variables,
        zone=zone,
        data=data_reshaped,
        var_index=name_to_col,
    )



# --------------------------------------------------
# write Tecplot ASCII structured-zone file
# --------------------------------------------------



def write_tecplot_ascii(
    path: str | Path,
    variables: dict[str, np.ndarray],
    *,
    title: str = "debug",
    zone: str = "zone",
    fmt: str = ".10e",
) -> Path:
    """Write a single-zone Tecplot ASCII file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    variables : dict[str, ndarray]
        Ordered mapping of variable names to arrays.  All arrays must
        share the same shape.  1-D arrays produce an I-ordered zone;
        2-D arrays (J, I) produce a structured IJ zone.
    title : str, optional
        Tecplot TITLE string.
    zone : str, optional
        Tecplot ZONE T string.
    fmt : str, optional
        Format spec applied to every value (default ``".10e"``).

    Returns
    -------
    Path
        The path that was written.
    """
    path = Path(path)
    names = list(variables.keys())
    arrays = list(variables.values())

    # validate shapes
    shape = arrays[0].shape
    for name, arr in zip(names, arrays):
        if arr.shape != shape:
            raise ValueError(
                f"shape mismatch: '{names[0]}' is {shape}, "
                f"'{name}' is {arr.shape}"
            )

    # build header
    var_line = " ".join(f'"{n}"' for n in names)

    with open(path, "w") as f:
        f.write(f'TITLE = "{title}"\n')
        f.write(f"VARIABLES = {var_line}\n")

        if arrays[0].ndim == 1:
            ni = shape[0]
            f.write(f'ZONE T="{zone}", I={ni}\n')
            for i in range(ni):
                vals = " ".join(f"{a[i]:{fmt}}" for a in arrays)
                f.write(vals + "\n")

        elif arrays[0].ndim == 2:
            nj, ni = shape
            f.write(f'ZONE T="{zone}", I={ni}, J={nj}\n')
            for j in range(nj):
                for i in range(ni):
                    vals = " ".join(f"{a[j, i]:{fmt}}" for a in arrays)
                    f.write(vals + "\n")
        else:
            raise ValueError(f"expected 1-D or 2-D arrays, got ndim={arrays[0].ndim}")

    logger.debug("wrote tecplot file: %s", path)
    return path
