"""Parse a saved tournament TSV back into the data the results screen shows."""

_HEADER_KEYS = {'Date': 'date', 'User': 'user', 'GIFs': 'gifs', 'ROI': 'roi',
                'OTF radius': 'r_o', 'Total comparisons': 'comparisons'}


def parse_result_tsv(text):
    """Return header fields, the ranked candidates and their metric values.

    `ranking` is a list of {'full_label', 'values'} in rank order and
    `metric_names` names the value columns. Files without a ranking table
    (a tournament saved with no metrics) give an empty ranking.
    """
    out = {'date': '', 'user': '', 'gifs': [], 'roi': '', 'r_o': '',
           'comparisons': '', 'metric_names': [], 'ranking': []}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('Rank\tCandidate'):
            out['metric_names'] = line.split('\t')[2:]
            i += 1
            while i < len(lines) and lines[i].strip():
                cells = lines[i].split('\t')
                try:
                    values = [float(v) for v in cells[2:]]
                except ValueError:
                    values = []
                out['ranking'].append({'full_label': cells[1] if len(cells) > 1 else '',
                                       'values': values})
                i += 1
            break
        key, sep, value = line.partition(':')
        if sep and key in _HEADER_KEYS:
            value = value.strip()
            if key == 'GIFs':
                value = [g.strip() for g in value.split(',') if g.strip()]
            out[_HEADER_KEYS[key]] = value
        i += 1
    if out['roi'] == 'none':
        out['roi'] = ''
    return out


def split_full_label(full_label):
    """'<gif path> Ep.85' -> ('<gif path>', 'Ep.85')."""
    filename, _, epoch_label = full_label.rpartition(' ')
    return filename, epoch_label
